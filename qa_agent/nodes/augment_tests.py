"""
qa_agent/nodes/augment_tests.py
──────────────────────────────────
Test augmentation (gap-filling) — node 9 of 12.

Decision boundary
─────────────────
Answers: "What additional tests address the specific gaps identified by the judge?"
Reuses TestGeneratorAgent's prompt in gap-filling mode (AUGMENT_TESTS_USER).
Never evaluates coverage, never executes tests.

Inputs consumed from state
──────────────────────────
  state["coverage_report"]   CoverageReport — .gaps contains gap descriptions
  state["test_scenarios"]    List[TestScenario]
  state["generated_tests"]   List[TestScript]  — to avoid duplication
  state["augmentation_cycle"]  int

State keys written
──────────────────
  generated_tests      List[TestScript]  — original + new augmentation scripts
  augmentation_cycle   int               — incremented
  current_phase        str
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from qa_agent.observability import log_node_execution
from qa_agent.prompts.test_generator import AUGMENT_TESTS_USER, TEST_GENERATOR_SYSTEM
from qa_agent.state import AgentError, QAAgentState, TestScript
from qa_agent.tools.spec_tools import find_spec_files, parse_openapi_spec, get_spec_summary

_VOCAREUM_API_KEY  = os.environ.get("VOCAREUM_API_KEY",  "voc-placeholder")
_VOCAREUM_BASE_URL = os.environ.get("VOCAREUM_BASE_URL", "https://openai.vocareum.com/v1")
_NODE_NAME = "augment_tests"


# ──────────────────────────────────────────────────────────────────────────────
# LLM output model (same shape as generate_tests node)
# ──────────────────────────────────────────────────────────────────────────────

class _AugmentedScript(BaseModel):
    scenario_id:  str
    framework:    str       = "pytest"
    content:      str
    dependencies: List[str] = Field(default_factory=list)


class _AugmentedTestBatch(BaseModel):
    tests: List[_AugmentedScript]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_gaps(gaps: List[str]) -> str:
    if not gaps:
        return "(no specific gaps identified)"
    return "\n".join(f"  - {g}" for g in gaps)


def _format_existing_scenarios(existing_tests: List[TestScript]) -> str:
    """Compact listing of already-generated scenarios to prevent duplication."""
    if not existing_tests:
        return "[]"
    return json.dumps(
        [{"scenario_id": t.scenario_id, "framework": t.framework} for t in existing_tests],
        indent=2,
    )


def _load_spec_sections(gaps: List[str]) -> str:
    """
    Attempt to load relevant spec sections for the gap endpoints.
    Falls back to full spec summary if endpoint parsing fails.
    """
    spec_files = find_spec_files(".")
    if not spec_files:
        return "(No OpenAPI spec found)"

    try:
        parsed = parse_openapi_spec(spec_files[0], validate=False)
        return get_spec_summary(parsed)[:6000]
    except Exception:
        return "(Spec parsing failed — generate tests based on gap descriptions only)"


def _augmented_to_test_script(
    gen:    _AugmentedScript,
    cycle:  int,
    base_dir: str = "/tmp/generated_tests",
) -> TestScript:
    """Convert LLM output → canonical TestScript with augmentation prefix."""
    # Ensure scenario_id carries the augmentation prefix (guard against LLM drift)
    scenario_id = gen.scenario_id
    if not scenario_id.startswith(f"augment_{cycle}_"):
        scenario_id = f"augment_{cycle}_{scenario_id}"

    filename = f"{scenario_id[:60]}.py".replace(" ", "_")

    return TestScript(
        scenario_id=scenario_id,
        file_path=f"{base_dir}/{filename}",
        framework=gen.framework.lower(),
        content=gen.content,
        dependencies=gen.dependencies,
        chromadb_id=None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def augment_tests(state: QAAgentState) -> dict:
    """
    Generate additional test scripts that specifically address the coverage gaps
    reported by CoverageEvaluatorAgent.

    Uses the AUGMENT_TESTS_USER prompt (gap-filling mode) with the same
    TEST_GENERATOR_SYSTEM role definition as generate_tests.

    Pipeline position: Node 9 (conditional — only reached when coverage score < 6.0).
    Input:  state["coverage_report"], state["test_scenarios"],
            state["generated_tests"], state["augmentation_cycle"]
    Output: state["generated_tests"] (appended), state["augmentation_cycle"]

    LLM call: yes — with_structured_output(_AugmentedTestBatch) via Vocareum.
    """
    start_ts       = time.time()
    coverage       = state.get("coverage_report")
    existing_tests = list(state.get("generated_tests") or [])
    scenarios      = state.get("test_scenarios") or []
    aug_cycle      = state.get("augmentation_cycle", 0)
    errors: list   = []

    gaps = coverage.gaps if coverage else []

    if not gaps:
        log_node_execution(
            node_name=_NODE_NAME,
            input_summary={"gaps": 0, "aug_cycle": aug_cycle},
            output_summary={"new_tests": 0, "note": "no gaps to fill"},
            routing_hint="execute_tests",
            pr_number=state["pr_metadata"].pr_number,
        )
        return {
            "augmentation_cycle": aug_cycle + 1,
            "current_phase":      _NODE_NAME,
        }

    # ── Build prompt context ──────────────────────────────────────────────────
    gaps_text          = _format_gaps(gaps)
    existing_scenarios = _format_existing_scenarios(existing_tests)
    spec_sections      = _load_spec_sections(gaps)

    user_message = AUGMENT_TESTS_USER.format(
        gaps_list=gaps_text,
        spec_sections=spec_sections[:6000],
        existing_scenarios=existing_scenarios[:3000],
        augmentation_cycle=aug_cycle + 1,
    )

    # ── LLM call ─────────────────────────────────────────────────────────────
    error_str: Optional[str] = None
    new_scripts: List[TestScript] = []

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=_VOCAREUM_API_KEY,
            base_url=_VOCAREUM_BASE_URL,
            temperature=0,
        ).with_structured_output(_AugmentedTestBatch)

        result: _AugmentedTestBatch = llm.invoke([
            {"role": "system", "content": TEST_GENERATOR_SYSTEM},
            {"role": "user",   "content": user_message},
        ])

        for gen in result.tests:
            new_scripts.append(_augmented_to_test_script(gen, aug_cycle + 1))

    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        errors.append(AgentError(
            agent="AugmentTestsAgent",
            phase="augment_tests",
            error=error_str,
            recoverable=True,
        ))
        # No fallback stubs for augmentation — gaps remain unfilled but pipeline continues

    # ── Merge new scripts with existing (deduplicate by scenario_id) ──────────
    existing_ids = {t.scenario_id for t in existing_tests}
    unique_new   = [s for s in new_scripts if s.scenario_id not in existing_ids]
    all_scripts  = existing_tests + unique_new

    # ── Log before returning ──────────────────────────────────────────────────
    duration_ms = (time.time() - start_ts) * 1000
    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "gaps":            len(gaps),
            "existing_tests":  len(existing_tests),
            "aug_cycle":       aug_cycle,
        },
        output_summary={
            "new_tests_added": len(unique_new),
            "duplicates_dropped": len(new_scripts) - len(unique_new),
            "total_tests":     len(all_scripts),
        },
        routing_hint="execute_tests",
        pr_number=state["pr_metadata"].pr_number,
        duration_ms=duration_ms,
        error=error_str,
    )

    result_dict: dict = {
        "generated_tests":   all_scripts,
        "augmentation_cycle": aug_cycle + 1,
        "current_phase":     _NODE_NAME,
    }
    if errors:
        result_dict["errors"] = errors
    return result_dict
