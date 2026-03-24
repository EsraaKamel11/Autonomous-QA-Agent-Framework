"""
qa_agent/nodes/store_tests.py
───────────────────────────────
Test memory storage — node 5 of 12.

Decision boundary
─────────────────
Only answers: "Store these test scripts in ChromaDB for future retrieval."
Pure tool call — zero LLM calls.  Continues gracefully if ChromaDB is down.

Inputs consumed from state
──────────────────────────
  state["generated_tests"]   List[TestScript]
  state["test_scenarios"]    List[TestScenario]  — for scenario descriptions
  state["pr_metadata"]       PRMetadata          — pr_number for provenance

State keys written
──────────────────
  generated_tests   List[TestScript]  — same list with chromadb_id set on each
  current_phase     str
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from qa_agent.observability import log_node_execution
from qa_agent.state import QAAgentState, TestScenario, TestScript
from qa_agent.tools.memory_tools import MemoryStoreResult, store_test_in_memory

_NODE_NAME = "store_tests"


def _get_scenario_description(
    scenario_id: str,
    scenarios_by_id: Dict[str, TestScenario],
) -> str:
    """Return the description for a scenario, or a safe default."""
    scenario = scenarios_by_id.get(scenario_id)
    return scenario.description if scenario else f"Test for {scenario_id}"


def store_tests(state: QAAgentState) -> dict:
    """
    Upsert every generated test script into ChromaDB so that future pipeline
    runs can retrieve them as few-shot context via retrieve_history.

    Scripts are stored with passed=False at this stage (before execution).
    The store_test_in_memory tool uses upsert semantics, so re-running is safe.

    Pipeline position: Node 5.
    Input:  state["generated_tests"], state["test_scenarios"], state["pr_metadata"]
    Output: state["generated_tests"] (same list, chromadb_id populated per script)

    LLM calls: NONE.
    Tool calls: memory_tools.store_test_in_memory (ChromaDB upsert).
    Failure mode: on ChromaDB error, chromadb_id stays None — pipeline continues.
    """
    start_ts       = time.time()
    generated      = state.get("generated_tests") or []
    scenarios      = state.get("test_scenarios")  or []
    pr_metadata    = state["pr_metadata"]

    if not generated:
        log_node_execution(
            node_name=_NODE_NAME,
            input_summary={"generated_tests": 0},
            output_summary={"stored": 0, "note": "nothing to store"},
            routing_hint="execute_tests",
            pr_number=pr_metadata.pr_number,
        )
        return {"generated_tests": [], "current_phase": _NODE_NAME}

    # Build scenario lookup: scenario_id → TestScenario.description
    scenarios_by_id: Dict[str, TestScenario] = {
        s.scenario_id: s for s in scenarios
    }

    # ── Upsert each script ────────────────────────────────────────────────────
    stored_count  = 0
    failed_count  = 0
    updated_scripts: List[TestScript] = []

    for script in generated:
        description = _get_scenario_description(
            script.scenario_id, scenarios_by_id
        )
        try:
            result: MemoryStoreResult = store_test_in_memory(
                test_script=script,
                scenario_description=description,
                passed=False,       # not yet executed
                pr_number=pr_metadata.pr_number,
            )
            # Stamp the chromadb_id onto the script for downstream traceability
            updated_scripts.append(
                script.model_copy(update={"chromadb_id": result.chroma_id})
            )
            stored_count += 1

        except Exception:
            # ChromaDB unavailable or upsert failed — keep original script
            updated_scripts.append(script)
            failed_count += 1

    # ── Log before returning ──────────────────────────────────────────────────
    duration_ms = (time.time() - start_ts) * 1000
    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={"generated_tests": len(generated)},
        output_summary={
            "stored_successfully": stored_count,
            "storage_failures":    failed_count,
        },
        routing_hint="execute_tests",
        pr_number=pr_metadata.pr_number,
        duration_ms=duration_ms,
    )

    return {"generated_tests": updated_scripts, "current_phase": _NODE_NAME}
