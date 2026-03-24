"""
qa_agent/nodes/self_heal.py
─────────────────────────────
SelfHealingAgent — node 7 of 12.

Decision boundary
─────────────────
Only answers: "Can I fix this failure autonomously without human intervention?"
Never files bug tickets, never evaluates coverage, never makes merge decisions.

Inputs consumed from state
──────────────────────────
  state["execution_report"]   ExecutionReport — latest cycle (not full history)
  state["generated_tests"]    List[TestScript]
  state["test_scenarios"]     List[TestScenario]
  state["retry_count"]        int

State keys written
──────────────────
  generated_tests            List[TestScript]  — patched scripts
  repair_attempts            List[RepairAttempt]  — accumulated via operator.add
  retry_count                int               — incremented
  human_escalation_required  bool
  escalation_reason          str | None
  current_phase              str
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from qa_agent.healing.classifier import ClassificationResult, classify_failure, is_auto_healable
from qa_agent.healing.prompts import (
    SELECTOR_REPAIR_SYSTEM,
    SELECTOR_REPAIR_USER,
    SELF_HEALING_SYSTEM,
    SELF_HEALING_USER,
)
from qa_agent.observability import log_node_execution
from qa_agent.state import (
    AgentError,
    EscalationRequest,
    ExecutionReport,
    FailureType,
    QAAgentState,
    RepairAttempt,
    TestResult,
    TestScenario,
    TestScript,
)
from qa_agent.tools.spec_tools import (
    extract_spec_for_endpoint,
    find_spec_files,
    parse_openapi_spec,
)

_VOCAREUM_API_KEY  = os.environ.get("VOCAREUM_API_KEY",  "voc-placeholder")
_VOCAREUM_BASE_URL = os.environ.get("VOCAREUM_BASE_URL", "https://openai.vocareum.com/v1")
_NODE_NAME = "self_heal"


# ──────────────────────────────────────────────────────────────────────────────
# LLM output model — matches the JSON contract in healing/prompts.py
# ──────────────────────────────────────────────────────────────────────────────

class _HealPatch(BaseModel):
    action:            str                  # "patch" | "escalate"
    patched_content:   Optional[str] = None
    patch_explanation: str           = ""
    escalation_reason: Optional[str] = None


class _SelectorPatch(BaseModel):
    """Extended output for SELECTOR failures (includes alternative_locators)."""
    action:              str
    patched_content:     Optional[str]             = None
    patch_explanation:   str                       = ""
    alternative_locators: List[dict]               = []
    escalation_reason:   Optional[str]             = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_llm(output_schema) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        api_key=_VOCAREUM_API_KEY,
        base_url=_VOCAREUM_BASE_URL,
        temperature=0,
    ).with_structured_output(output_schema)


def _get_spec_section(scenario: Optional[TestScenario]) -> str:
    """
    Attempt to extract the OpenAPI spec section for the scenario's endpoint.
    Falls back to scenario expected-behaviour text when spec is unavailable.
    """
    if scenario is None:
        return "(spec not available)"

    # Try to load and parse the spec from disk
    spec_files = find_spec_files(".")
    if spec_files:
        try:
            parsed = parse_openapi_spec(spec_files[0], validate=False)
            section = extract_spec_for_endpoint(
                method=scenario.method,
                path=scenario.endpoint,
                parsed_spec=parsed,
            )
            if section.raw_section:
                return section.raw_section
        except Exception:
            pass

    # Fallback: use scenario fields as proxy for spec
    return (
        f"endpoint: {scenario.endpoint}\n"
        f"method: {scenario.method}\n"
        f"expected_status: {scenario.expected_status}\n"
        f"expected_behavior: {scenario.expected_behavior}\n"
        f"expected_body: {scenario.expected_body}\n"
        f"preconditions: {scenario.preconditions}\n"
    )


def _extract_element_description(result: TestResult) -> str:
    """
    Try to extract a human-readable element description from the failure output.
    Used as context for SELECTOR_REPAIR_USER injection.
    """
    combined = (result.error_message or "") + (result.stdout or "")
    # Look for common Playwright error patterns mentioning the element
    for line in combined.splitlines():
        if any(k in line.lower() for k in
               ("locator(", "get_by_", "waiting for", "element")):
            return line.strip()[:200]
    return "(could not determine element — see error message)"


def _find_script_for_result(
    result: TestResult,
    scripts: List[TestScript],
) -> Optional[TestScript]:
    """Find the TestScript that corresponds to a failed TestResult."""
    # Match by scenario_id first (most reliable)
    if result.scenario_id:
        for s in scripts:
            if s.scenario_id == result.scenario_id:
                return s
    # Fall back: match by test_id substring
    for s in scripts:
        if s.scenario_id and s.scenario_id in (result.test_id or ""):
            return s
    return None


def _find_scenario(
    script: TestScript,
    scenarios_by_id: Dict[str, TestScenario],
) -> Optional[TestScenario]:
    return scenarios_by_id.get(script.scenario_id)


def _apply_patch(
    scripts: List[TestScript],
    original: TestScript,
    new_content: str,
) -> List[TestScript]:
    """Return scripts list with the target script's content replaced."""
    return [
        (s.model_copy(update={"content": new_content})
         if s.scenario_id == original.scenario_id else s)
        for s in scripts
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Per-failure healing logic
# ──────────────────────────────────────────────────────────────────────────────

def _heal_one(
    result:        TestResult,
    script:        TestScript,
    classification: ClassificationResult,
    scenario:      Optional[TestScenario],
    attempt_number: int,
) -> tuple[Optional[str], Optional[str], RepairAttempt]:
    """
    Attempt to heal one failure.  Returns (new_content, escalation_reason, attempt).
    new_content is None on escalation or LLM failure.
    """
    failure_type = classification.failure_type
    spec_section = _get_spec_section(scenario)

    # ── SELECTOR: use specialist prompt ───────────────────────────────────────
    if failure_type == FailureType.SELECTOR:
        llm = _make_llm(_SelectorPatch)
        user_msg = SELECTOR_REPAIR_USER.format(
            error_message=result.error_message or result.stdout[:500],
            element_description=_extract_element_description(result),
            test_content=script.content,
            file_path=script.file_path,
        )
        messages = [
            {"role": "system", "content": SELECTOR_REPAIR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
        patch_cls = _SelectorPatch

    # ── ASSERTION / SCHEMA: use generic healing prompt ────────────────────────
    else:
        llm = _make_llm(_HealPatch)
        user_msg = SELF_HEALING_USER.format(
            failure_type=failure_type.value,
            error_message=result.error_message or result.stdout[:500],
            test_content=script.content,
            file_path=script.file_path,
            spec_section=spec_section,
        )
        messages = [
            {"role": "system", "content": SELF_HEALING_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
        patch_cls = _HealPatch

    try:
        patch = llm.invoke(messages)
    except Exception as exc:
        reason = f"LLM call failed: {type(exc).__name__}: {exc}"
        attempt = RepairAttempt(
            test_id=result.test_id or script.scenario_id,
            attempt_number=attempt_number,
            failure_type=failure_type,
            patch_applied=reason,
            success=False,
        )
        return None, reason, attempt

    success = (patch.action == "patch" and bool(patch.patched_content))
    attempt = RepairAttempt(
        test_id=result.test_id or script.scenario_id,
        attempt_number=attempt_number,
        failure_type=failure_type,
        patch_applied=patch.patch_explanation[:400] if patch.patch_explanation else "",
        success=success,
    )

    if success:
        return patch.patched_content, None, attempt
    else:
        return None, patch.escalation_reason or "LLM chose to escalate", attempt


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def self_heal(state: QAAgentState) -> dict:
    """
    SelfHealingAgent: for each failed test in the latest execution cycle,
    classify the failure and attempt autonomous repair.

    ENVIRONMENT and AUTH failures → immediate escalation (no LLM call).
    ASSERTION, SCHEMA, SELECTOR → LLM generates a minimal patch.
    If any repair fails or escalation is required → human_review flagged.

    Pipeline position: Node 7 (conditional — only reached when failures exist).
    Input:  state["execution_report"], state["generated_tests"],
            state["test_scenarios"], state["retry_count"]
    Output: state["generated_tests"] (patched), state["repair_attempts"],
            state["retry_count"], state["human_escalation_required"],
            state["escalation_reason"]

    LLM calls: one per auto-healable failure (with_structured_output).
    """
    start_ts  = time.time()
    report: ExecutionReport = state.get("execution_report") or ExecutionReport(
        total=0, passed=0, failed=0, errors=0, skipped=0
    )
    scripts       = list(state.get("generated_tests") or [])
    scenarios     = state.get("test_scenarios") or []
    retry_count   = state.get("retry_count", 0)
    errors: list  = []

    scenarios_by_id: Dict[str, TestScenario] = {
        s.scenario_id: s for s in scenarios
    }

    # ── Identify failures from LATEST execution cycle ─────────────────────────
    failed_results = [
        r for r in report.per_test_results
        if r.status in ("failed", "error")
    ]

    if not failed_results:
        log_node_execution(
            node_name=_NODE_NAME,
            input_summary={"failed_results": 0},
            output_summary={"note": "no failures — nothing to heal"},
            routing_hint="execute_tests",
            pr_number=state["pr_metadata"].pr_number,
        )
        return {
            "retry_count":                retry_count + 1,
            "human_escalation_required":  False,
            "escalation_reason":          None,
            "current_phase":              _NODE_NAME,
        }

    # ── Process each failure ───────────────────────────────────────────────────
    new_attempts:      List[RepairAttempt] = []
    escalation_needed  = False
    escalation_reasons: List[str] = []
    patched_count      = 0
    escalated_count    = 0

    for result in failed_results:
        # 1. Classify (may already be set from execute_tests node)
        if result.failure_type:
            failure_type = result.failure_type
            classification = ClassificationResult(
                failure_type=failure_type,
                confidence=0.9,
                method="pre_classified",
                reasoning=f"Pre-classified by execute_tests: {failure_type.value}",
            )
        else:
            classification = classify_failure(
                error_message=result.error_message or "",
                stderr=result.stderr or "",
                stdout=result.stdout or "",
            )
            failure_type = classification.failure_type

        # 2. Non-healable types → escalate immediately without LLM call
        if not is_auto_healable(failure_type):
            reason = (
                f"Non-healable failure type '{failure_type.value}' on "
                f"test '{result.test_id}': {(result.error_message or '')[:200]}"
            )
            escalation_needed = True
            escalation_reasons.append(reason)
            escalated_count += 1

            new_attempts.append(RepairAttempt(
                test_id=result.test_id or "unknown",
                attempt_number=retry_count + 1,
                failure_type=failure_type,
                patch_applied="escalated — non-healable type",
                success=False,
            ))
            continue

        # 3. Find the corresponding script
        script = _find_script_for_result(result, scripts)
        if script is None:
            reason = f"Could not find script for test_id '{result.test_id}'"
            escalation_reasons.append(reason)
            escalated_count += 1
            continue

        # 4. Attempt LLM-generated patch
        scenario = _find_scenario(script, scenarios_by_id)
        new_content, esc_reason, attempt = _heal_one(
            result=result,
            script=script,
            classification=classification,
            scenario=scenario,
            attempt_number=retry_count + 1,
        )
        new_attempts.append(attempt)

        if new_content:
            scripts = _apply_patch(scripts, script, new_content)
            patched_count += 1
        else:
            escalation_needed = True
            if esc_reason:
                escalation_reasons.append(esc_reason)
            escalated_count += 1

    # ── Log before returning ──────────────────────────────────────────────────
    routing_hint = "execute_tests" if not escalation_needed else "human_review"
    duration_ms  = (time.time() - start_ts) * 1000

    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "failed_results":  len(failed_results),
            "retry_count":     retry_count,
        },
        output_summary={
            "patched":           patched_count,
            "escalated":         escalated_count,
            "escalation_needed": escalation_needed,
        },
        routing_hint=routing_hint,
        pr_number=state["pr_metadata"].pr_number,
        duration_ms=duration_ms,
    )

    escalation_reason_str = (
        "; ".join(escalation_reasons[:3]) if escalation_reasons else None
    )

    result_dict: dict = {
        "generated_tests":           scripts,
        "repair_attempts":           new_attempts,   # operator.add — accumulated
        "retry_count":               retry_count + 1,
        "human_escalation_required": escalation_needed,
        "escalation_reason":         escalation_reason_str,
        "current_phase":             _NODE_NAME,
    }
    if errors:
        result_dict["errors"] = errors
    return result_dict
