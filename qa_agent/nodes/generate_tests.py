"""
qa_agent/nodes/generate_tests.py
──────────────────────────────────
TestGeneratorAgent — node 4 of 12.

Decision boundary
─────────────────
Only answers: "What executable Python code tests these behaviours?"
Never executes tests, never classifies failures, never evaluates coverage.

Inputs consumed from state
──────────────────────────
  state["test_scenarios"]    List[TestScenario]
  state["historical_tests"]  List[TestScript]    — few-shot context

State keys written
──────────────────
  generated_tests   List[TestScript]
  current_phase     str
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
from qa_agent.prompts.test_generator import (
    HISTORICAL_CONTEXT_BLOCK,
    TEST_GENERATOR_SYSTEM,
    TEST_GENERATOR_USER,
)
from qa_agent.state import AgentError, QAAgentState, TestScript

_VOCAREUM_API_KEY  = os.environ.get("VOCAREUM_API_KEY",  "voc-placeholder")
_VOCAREUM_BASE_URL = os.environ.get("VOCAREUM_BASE_URL", "https://openai.vocareum.com/v1")
_NODE_NAME = "generate_tests"

# Batch size: generate tests for N scenarios per LLM call to stay within
# token limits.  Larger batches = fewer API calls but risk context overflow.
_BATCH_SIZE = 5


# ──────────────────────────────────────────────────────────────────────────────
# LLM output models
# ──────────────────────────────────────────────────────────────────────────────

class _GeneratedScript(BaseModel):
    """One test file produced by the LLM per scenario."""
    scenario_id:  str
    framework:    str           = "pytest"   # pytest | playwright
    content:      str           # full executable Python source
    dependencies: List[str]     = Field(default_factory=list)


class _GeneratedTestBatch(BaseModel):
    """Wrapper — with_structured_output cannot return a bare list."""
    tests: List[_GeneratedScript]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_historical_context(historical: List[TestScript]) -> str:
    """
    Format up to 3 historical tests as few-shot examples.
    Capped at 3 to limit token usage; highest similarity come first
    (already sorted by retrieve_history node).
    """
    if not historical:
        return ""

    examples: List[str] = []
    for script in historical[:3]:
        header = (
            f"# scenario_id: {script.scenario_id}\n"
            f"# framework:   {script.framework}\n"
            f"# file_path:   {script.file_path}\n"
        )
        truncated_content = script.content[:1500]
        if len(script.content) > 1500:
            truncated_content += "\n# ... [truncated for context] ..."
        examples.append(header + truncated_content)

    historical_block = "\n\n---\n\n".join(examples)
    return HISTORICAL_CONTEXT_BLOCK.format(historical_tests=historical_block)


def _scenarios_to_json(scenarios: list) -> str:
    """Serialise TestScenario list to compact JSON for the LLM prompt."""
    return json.dumps(
        [
            {
                "scenario_id":      s.scenario_id,
                "endpoint":         s.endpoint,
                "method":           s.method,
                "scenario_type":    s.scenario_type.value,
                "description":      s.description,
                "preconditions":    s.preconditions,
                "expected_behavior": s.expected_behavior,
                "expected_status":  s.expected_status,
                "expected_body":    s.expected_body,
                "priority":         s.priority,
            }
            for s in scenarios
        ],
        indent=2,
    )


def _script_to_test_script(
    gen: _GeneratedScript,
    base_dir: str = "/tmp/generated_tests",
) -> TestScript:
    """Convert LLM _GeneratedScript → canonical TestScript."""
    ext = ".py"
    filename = f"{gen.scenario_id[:60]}{ext}".replace(" ", "_")
    file_path = f"{base_dir}/{filename}"

    return TestScript(
        scenario_id=gen.scenario_id,
        file_path=file_path,
        framework=gen.framework.lower(),
        content=gen.content,
        dependencies=gen.dependencies,
        chromadb_id=None,   # set by store_tests node
    )


def _fallback_scripts(scenarios: list) -> List[TestScript]:
    """
    Minimal stub tests when the LLM call fails.
    These will execute and report as passed (trivially), allowing the pipeline
    to continue to coverage evaluation with a low score.
    """
    stubs: List[TestScript] = []
    for scenario in scenarios:
        stub_content = (
            f"import pytest\n\n"
            f"def test_{scenario.scenario_id[:40].replace('-', '_')}():\n"
            f"    \"\"\"[STUB] {scenario.description}\"\"\"\n"
            f"    # LLM generation failed — this stub always passes.\n"
            f"    # Replace with a real test before merging.\n"
            f"    pytest.skip('TestGeneratorAgent LLM call failed — stub test')\n"
        )
        stubs.append(TestScript(
            scenario_id=scenario.scenario_id,
            file_path=f"/tmp/generated_tests/stub_{scenario.scenario_id[:40]}.py",
            framework="pytest",
            content=stub_content,
            dependencies=[],
            chromadb_id=None,
        ))
    return stubs


def _call_llm_batch(
    llm: ChatOpenAI,
    batch: list,
    historical_context: str,
) -> List[TestScript]:
    """
    Call the LLM for one batch of scenarios.  Returns TestScript list.
    Raises on LLM failure — caller handles the exception.
    """
    scenarios_json = _scenarios_to_json(batch)
    user_message   = TEST_GENERATOR_USER.format(
        historical_context=historical_context,
        scenarios_json=scenarios_json,
    )

    result: _GeneratedTestBatch = llm.invoke([
        {"role": "system", "content": TEST_GENERATOR_SYSTEM},
        {"role": "user",   "content": user_message},
    ])

    # Map results back to TestScripts; fall back to stub for any missing IDs
    scenario_ids = {s.scenario_id for s in batch}
    generated_ids: set = set()
    scripts: List[TestScript] = []

    for gen in result.tests:
        scripts.append(_script_to_test_script(gen))
        generated_ids.add(gen.scenario_id)

    # Stub any scenarios the LLM silently dropped
    for scenario in batch:
        if scenario.scenario_id not in generated_ids:
            scripts.extend(_fallback_scripts([scenario]))

    return scripts


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def generate_tests(state: QAAgentState) -> dict:
    """
    TestGeneratorAgent: generate executable pytest/playwright test files for
    every TestScenario, optionally using historical tests as few-shot context.

    Pipeline position: Node 4.
    Input:  state["test_scenarios"], state["historical_tests"]
    Output: state["generated_tests"]

    LLM call: yes — with_structured_output(_GeneratedTestBatch) via Vocareum.
              Batched in groups of _BATCH_SIZE scenarios per call.
    """
    start_ts  = time.time()
    scenarios = state.get("test_scenarios") or []
    historical = state.get("historical_tests") or []
    errors: list = []

    if not scenarios:
        log_node_execution(
            node_name=_NODE_NAME,
            input_summary={"scenarios": 0},
            output_summary={"generated_tests": 0, "note": "no scenarios"},
            routing_hint="store_tests",
            pr_number=state["pr_metadata"].pr_number,
        )
        return {"generated_tests": [], "current_phase": _NODE_NAME}

    # ── Sort by priority (1=critical first) ─────────────────────────────────
    sorted_scenarios = sorted(scenarios, key=lambda s: s.priority)

    historical_context = _build_historical_context(historical)

    # ── Initialise LLM ───────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=_VOCAREUM_API_KEY,
        base_url=_VOCAREUM_BASE_URL,
        temperature=0,
    ).with_structured_output(_GeneratedTestBatch)

    # ── Batch LLM calls ──────────────────────────────────────────────────────
    all_scripts: List[TestScript] = []
    error_str: Optional[str] = None

    for batch_start in range(0, len(sorted_scenarios), _BATCH_SIZE):
        batch = sorted_scenarios[batch_start : batch_start + _BATCH_SIZE]

        try:
            batch_scripts = _call_llm_batch(llm, batch, historical_context)
            all_scripts.extend(batch_scripts)

        except Exception as exc:
            error_str = f"Batch {batch_start // _BATCH_SIZE}: {type(exc).__name__}: {exc}"
            errors.append(AgentError(
                agent="TestGeneratorAgent",
                phase="generate_tests",
                error=error_str,
                recoverable=True,
            ))
            # Stub the failed batch — continue with remaining batches
            all_scripts.extend(_fallback_scripts(batch))

    # ── Log before returning ─────────────────────────────────────────────────
    pytest_count     = sum(1 for s in all_scripts if s.framework == "pytest")
    playwright_count = sum(1 for s in all_scripts if s.framework == "playwright")
    duration_ms      = (time.time() - start_ts) * 1000

    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "scenarios":         len(scenarios),
            "historical_tests":  len(historical),
            "batches":           -(-len(scenarios) // _BATCH_SIZE),
        },
        output_summary={
            "generated_tests":    len(all_scripts),
            "pytest_scripts":     pytest_count,
            "playwright_scripts": playwright_count,
            "stubs_generated":    sum(1 for s in all_scripts if "STUB" in s.content),
        },
        routing_hint="store_tests",
        pr_number=state["pr_metadata"].pr_number,
        duration_ms=duration_ms,
        error=error_str,
    )

    result: dict = {"generated_tests": all_scripts, "current_phase": _NODE_NAME}
    if errors:
        result["errors"] = errors
    return result
