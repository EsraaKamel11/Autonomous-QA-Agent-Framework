"""
qa_agent/nodes/retrieve_history.py
─────────────────────────────────────
Historical test retrieval — node 3 of 12.

Decision boundary
─────────────────
Only answers: "Have we successfully tested similar scenarios before?"
Pure tool call — zero LLM calls.  Degrades silently if ChromaDB is down.

Inputs consumed from state
──────────────────────────
  state["test_scenarios"]   List[TestScenario]

State keys written
──────────────────
  historical_tests   List[TestScript]   — top-5 per scenario, similarity > 0.75
  current_phase      str
"""

from __future__ import annotations

import os
import time
from typing import List, Set

from qa_agent.observability import log_node_execution
from qa_agent.state import QAAgentState, TestScript
from qa_agent.tools.memory_tools import (
    RetrievedTest,
    retrieve_similar_tests,
)

_NODE_NAME = "retrieve_history"

# Maximum number of historical tests to carry forward into the generator.
# Capped to avoid overwhelming the TestGeneratorAgent's context window.
_MAX_TOTAL_HISTORICAL = 15


def retrieve_history(state: QAAgentState) -> dict:
    """
    For each TestScenario, query ChromaDB for historically successful tests
    with cosine similarity ≥ 0.75.  Deduplicates by chromadb_id.

    Pipeline position: Node 3.
    Input:  state["test_scenarios"]
    Output: state["historical_tests"]

    LLM calls: NONE.
    Tool calls: memory_tools.retrieve_similar_tests (ChromaDB HTTP query).
    Failure mode: returns empty list — never crashes the pipeline.
    """
    start_ts  = time.time()
    scenarios = state.get("test_scenarios") or []

    if not scenarios:
        log_node_execution(
            node_name=_NODE_NAME,
            input_summary={"scenarios": 0},
            output_summary={"historical_tests": 0, "note": "no scenarios to query"},
            routing_hint="generate_tests",
            pr_number=state["pr_metadata"].pr_number,
        )
        return {"historical_tests": []}

    # ── Query ChromaDB for each scenario ─────────────────────────────────────
    collected:   List[RetrievedTest] = []
    seen_ids:    Set[str]            = set()

    for scenario in scenarios:
        try:
            results: List[RetrievedTest] = retrieve_similar_tests(
                scenario_description=scenario.description,
                endpoint=scenario.endpoint,
                n_results=5,              # top-5 per scenario
                similarity_threshold=0.75,
            )
        except Exception:
            # ChromaDB unreachable or query failed — skip this scenario
            results = []

        for r in results:
            if r.chroma_id not in seen_ids:
                seen_ids.add(r.chroma_id)
                collected.append(r)

            if len(collected) >= _MAX_TOTAL_HISTORICAL:
                break

        if len(collected) >= _MAX_TOTAL_HISTORICAL:
            break

    # Sort by similarity (highest first) and extract TestScript objects
    collected.sort(key=lambda r: r.similarity, reverse=True)
    historical: List[TestScript] = [r.test_script for r in collected]

    # ── Log before returning ──────────────────────────────────────────────────
    duration_ms = (time.time() - start_ts) * 1000
    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "scenarios_queried": len(scenarios),
            "chromadb_available": len(collected) > 0 or len(scenarios) == 0,
        },
        output_summary={
            "historical_tests_found": len(historical),
            "top_similarity":         round(collected[0].similarity, 3) if collected else 0.0,
            "deduplication_removed":  sum(
                1 for r in collected if collected.count(r) > 1
            ),
        },
        routing_hint="generate_tests",
        pr_number=state["pr_metadata"].pr_number,
        duration_ms=duration_ms,
    )

    # current_phase intentionally omitted: retrieve_history runs in parallel with
    # generate_tests and both writing the same scalar key in one step causes
    # INVALID_CONCURRENT_GRAPH_UPDATE.  generate_tests sets current_phase instead.
    return {"historical_tests": historical}
