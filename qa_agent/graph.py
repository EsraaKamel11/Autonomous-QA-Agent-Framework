"""
qa_agent/graph.py
──────────────────
Full LangGraph state machine assembly for the Autonomous QA Agent Framework.

Node inventory (12 nodes)
──────────────────────────
 1. analyze_pr         — CodeAnalystAgent: PR diff → ChangeManifest
 2. parse_specs        — SpecParserAgent: OpenAPI → TestScenarios
 3. retrieve_history   — ChromaDB retrieval (no LLM)               ┐ parallel
 4. generate_tests     — TestGeneratorAgent: Scenarios → TestScripts┘ fan-out
 5. store_tests        — ChromaDB upsert (no LLM)                   ← fan-in
 6. execute_tests      — pytest / Playwright runner
 7. self_heal          — SelfHealingAgent: patches failing tests      (conditional)
 8. evaluate_coverage  — CoverageEvaluatorAgent: LLM-as-judge
 9. augment_tests      — TestGeneratorAgent gap-fill                  (conditional)
10. generate_defects   — DefectReporterAgent: Defects + Jira tickets
11. human_review       — HITL interrupt checkpoint                    (conditional)
12. finalize           — GitHub PR comment + commit status

Edge topology
──────────────
START → analyze_pr → parse_specs
parse_specs → retrieve_history ─┐
parse_specs → generate_tests ───┼─→ store_tests → execute_tests
                                 ┘
execute_tests ──[route_after_execution]──→ self_heal
                                       └─→ evaluate_coverage
self_heal ──[route_after_healing]──→ execute_tests   (retry loop)
                                 └─→ human_review
evaluate_coverage ──[route_after_coverage]──→ augment_tests
                                          └─→ generate_defects
augment_tests → execute_tests   (re-run after gap-fill)
generate_defects ──[route_after_defects]──→ human_review
                                        └─→ finalize
human_review ──[route_after_human_review]──→ finalize
                                         └─→ END
                                         └─→ generate_tests  (modify loop)
generate_tests → store_tests   (shared edge — also serves the modify loop)
finalize → END

Checkpointing
──────────────
SqliteSaver persists full state at ./checkpoints/qa_agent.db after every node.
The graph is compiled with interrupt_before=["human_review"] so the HITL
checkpoint is always written before the node suspends via interrupt().

Resuming a suspended graph
───────────────────────────
  from langgraph.types import Command

  graph.invoke(
      Command(resume={"action": "approve", "comment": "LGTM"}),
      config={"configurable": {"thread_id": "<thread_id>"}},
  )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from qa_agent.state import QAAgentState

# ── Node imports ──────────────────────────────────────────────────────────────
from qa_agent.nodes.analyze_pr        import analyze_pr
from qa_agent.nodes.parse_specs       import parse_specs
from qa_agent.nodes.retrieve_history  import retrieve_history
from qa_agent.nodes.generate_tests    import generate_tests
from qa_agent.nodes.store_tests       import store_tests
from qa_agent.nodes.execute_tests     import execute_tests
from qa_agent.nodes.self_heal         import self_heal
from qa_agent.nodes.evaluate_coverage import evaluate_coverage
from qa_agent.nodes.augment_tests     import augment_tests
from qa_agent.nodes.generate_defects  import generate_defects
from qa_agent.nodes.human_review      import human_review
from qa_agent.nodes.finalize          import finalize

# ── Routing imports ───────────────────────────────────────────────────────────
from qa_agent.routing import (
    COVERAGE_PATH_MAP,
    DEFECTS_PATH_MAP,
    EXECUTION_PATH_MAP,
    HEALING_PATH_MAP,
    HUMAN_REVIEW_PATH_MAP,
    route_after_coverage,
    route_after_defects,
    route_after_execution,
    route_after_healing,
    route_after_human_review,
)

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = os.environ.get("QA_AGENT_CHECKPOINT_DIR", "./checkpoints")
_CHECKPOINT_DB  = os.path.join(_CHECKPOINT_DIR, "qa_agent.db")


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    """
    Assemble and compile the full QA Agent LangGraph state machine.

    Called once at module load time.  Returns a compiled CompiledGraph
    with SqliteSaver checkpointing and interrupt_before=["human_review"].

    On SqliteSaver failure (e.g. missing langgraph-checkpoint-sqlite package)
    the function falls back to MemorySaver and logs a warning — the pipeline
    remains fully functional but checkpoint state is lost on process restart.
    """
    # ── Ensure checkpoint directory exists ───────────────────────────────────
    Path(_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Graph builder ─────────────────────────────────────────────────────────
    builder = StateGraph(QAAgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("analyze_pr",        analyze_pr)
    builder.add_node("parse_specs",       parse_specs)
    builder.add_node("retrieve_history",  retrieve_history)
    builder.add_node("generate_tests",    generate_tests)
    builder.add_node("store_tests",       store_tests)
    builder.add_node("execute_tests",     execute_tests)
    builder.add_node("self_heal",         self_heal)
    builder.add_node("evaluate_coverage", evaluate_coverage)
    builder.add_node("augment_tests",     augment_tests)
    builder.add_node("generate_defects",  generate_defects)
    builder.add_node("human_review",      human_review)
    builder.add_node("finalize",          finalize)

    # ── Phase 1: Ingestion ────────────────────────────────────────────────────
    builder.add_edge(START,        "analyze_pr")
    builder.add_edge("analyze_pr", "parse_specs")

    # ── Phase 2: Parallel fan-out / fan-in ────────────────────────────────────
    # parse_specs fans out to both retrieve_history and generate_tests
    # simultaneously.  store_tests is the fan-in point; LangGraph >= 0.2
    # waits for ALL predecessor branches within the same execution wave before
    # executing the converging node.
    builder.add_edge("parse_specs",      "retrieve_history")
    builder.add_edge("parse_specs",      "generate_tests")
    builder.add_edge("retrieve_history", "store_tests")
    builder.add_edge("generate_tests",   "store_tests")

    # ── Phase 3: Execution entry ──────────────────────────────────────────────
    # Two entry points into execute_tests:
    #   a) Initial run after store_tests
    #   b) Re-run after augment_tests adds gap-filling tests
    #   c) Retry after self_heal patches failing tests (wired below via HEALING_PATH_MAP)
    builder.add_edge("store_tests",   "execute_tests")
    builder.add_edge("augment_tests", "execute_tests")

    # ── Phase 3: Execution / self-healing loop ────────────────────────────────
    # execute_tests → self_heal            failures > 0 AND retry_count < max_retries
    #              → evaluate_coverage     else
    builder.add_conditional_edges(
        "execute_tests",
        route_after_execution,
        EXECUTION_PATH_MAP,
    )

    # self_heal → human_review     human_escalation_required is True
    #           → execute_tests    patches applied (retry loop)
    builder.add_conditional_edges(
        "self_heal",
        route_after_healing,
        HEALING_PATH_MAP,
    )

    # ── Phase 4: Coverage evaluation and gap-filling ──────────────────────────
    # evaluate_coverage → augment_tests    score < 7.0 AND aug_cycle < max_aug_cycles
    #                  → generate_defects  else
    builder.add_conditional_edges(
        "evaluate_coverage",
        route_after_coverage,
        COVERAGE_PATH_MAP,
    )

    # ── Phase 5: Defect reporting ─────────────────────────────────────────────
    # generate_defects → human_review    any CRITICAL defect OR ENVIRONMENT failure
    #                 → finalize         else
    builder.add_conditional_edges(
        "generate_defects",
        route_after_defects,
        DEFECTS_PATH_MAP,
    )

    # ── Phase 6: Human-in-the-Loop review ────────────────────────────────────
    # human_review → finalize         commit_status == "success"  (APPROVE)
    #             → END               commit_status == "failure"  (REJECT)
    #             → generate_tests    commit_status == "pending"  (MODIFY)
    #
    # The MODIFY path re-uses the existing generate_tests → store_tests edge
    # already wired in Phase 2.  Historical tests remain in state via
    # operator.add — no need to re-query ChromaDB.
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        HUMAN_REVIEW_PATH_MAP,
    )

    # ── Phase 7: Finalization ─────────────────────────────────────────────────
    builder.add_edge("finalize", END)

    # ── Checkpointer ─────────────────────────────────────────────────────────
    checkpointer = _make_checkpointer()

    # ── Compile ───────────────────────────────────────────────────────────────
    # interrupt_before=["human_review"] ensures the full graph state is
    # serialised to SQLite BEFORE the human_review node calls interrupt().
    # This allows the graph to be resumed from a different process or thread.
    compiled = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )

    logger.info(
        "QA Agent graph compiled — %d nodes, checkpoint: %s, "
        "interrupt_before=['human_review']",
        12,
        _CHECKPOINT_DB,
    )
    return compiled


def _make_checkpointer() -> Any:
    """
    Create a SqliteSaver instance for ./checkpoints/qa_agent.db.

    Falls back to MemorySaver if the sqlite checkpoint package is not
    installed (langgraph-checkpoint-sqlite).  MemorySaver is non-persistent
    across process restarts — appropriate for development only.
    """
    try:
        import sqlite3 as _sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import]
        conn = _sqlite3.connect(_CHECKPOINT_DB, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.debug("Checkpointer: SqliteSaver at %s", _CHECKPOINT_DB)
        return checkpointer
    except ImportError:
        pass  # fall through to MemorySaver
    except Exception as exc:
        logger.warning(
            "SqliteSaver initialisation failed (%s) — falling back to MemorySaver", exc
        )

    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import]
    logger.warning(
        "Using MemorySaver — state will NOT persist across process restarts. "
        "Install langgraph-checkpoint-sqlite for production use."
    )
    return MemorySaver()


# ──────────────────────────────────────────────────────────────────────────────
# Module-level compiled graph — the primary import target
# ──────────────────────────────────────────────────────────────────────────────

#: Importable compiled graph.  Usage:
#:
#:   from qa_agent.graph import compiled_graph
#:
#:   result = compiled_graph.invoke(
#:       initial_state(pr_metadata),
#:       config={"configurable": {"thread_id": str(pr_number)}},
#:   )
compiled_graph = _build_graph()


# ──────────────────────────────────────────────────────────────────────────────
# Debugging helper
# ──────────────────────────────────────────────────────────────────────────────

def print_graph_structure() -> None:
    """
    Print a human-readable summary of the graph topology to stdout.

    Covers: node names + responsibilities, all linear edges, all conditional
    edges with their routing functions, and checkpoint configuration.

    Optionally dumps the Mermaid diagram produced by LangGraph if the
    compiled graph exposes a get_graph() method (LangGraph >= 0.2.x).
    """
    _SEP = "═" * 72

    print(f"\n{_SEP}")
    print("  Autonomous QA Agent — LangGraph Topology")
    print(_SEP)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    print("\n  Nodes (12)\n  " + "─" * 40)
    _node_rows = [
        (" 1", "analyze_pr",        "CodeAnalystAgent — PR diff → ChangeManifest"),
        (" 2", "parse_specs",       "SpecParserAgent — OpenAPI → TestScenarios"),
        (" 3", "retrieve_history",  "ChromaDB retrieval (no LLM)  [parallel A]"),
        (" 4", "generate_tests",    "TestGeneratorAgent           [parallel B]"),
        (" 5", "store_tests",       "ChromaDB upsert (no LLM)     [fan-in]"),
        (" 6", "execute_tests",     "pytest / Playwright runner"),
        (" 7", "self_heal",         "SelfHealingAgent — patches failing tests"),
        (" 8", "evaluate_coverage", "CoverageEvaluatorAgent (LLM-as-judge)"),
        (" 9", "augment_tests",     "TestGeneratorAgent gap-fill"),
        ("10", "generate_defects",  "DefectReporterAgent — Defects + Jira"),
        ("11", "human_review",      "HITL interrupt checkpoint"),
        ("12", "finalize",          "GitHub PR comment + commit status"),
    ]
    for num, name, desc in _node_rows:
        print(f"  {num}. {name:<22} {desc}")

    # ── Linear edges ──────────────────────────────────────────────────────────
    print("\n  Linear Edges\n  " + "─" * 40)
    _linear = [
        "START            → analyze_pr",
        "analyze_pr       → parse_specs",
        "parse_specs      → retrieve_history   [fan-out]",
        "parse_specs      → generate_tests     [fan-out]",
        "retrieve_history → store_tests         [fan-in]",
        "generate_tests   → store_tests         [fan-in]",
        "store_tests      → execute_tests",
        "augment_tests    → execute_tests       [re-run after gap-fill]",
        "finalize         → END",
    ]
    for edge in _linear:
        print(f"  {edge}")

    # ── Conditional edges ─────────────────────────────────────────────────────
    print("\n  Conditional Edges\n  " + "─" * 40)
    _conditional = [
        ("execute_tests",     "route_after_execution",    "self_heal | evaluate_coverage"),
        ("self_heal",         "route_after_healing",      "execute_tests | human_review"),
        ("evaluate_coverage", "route_after_coverage",     "augment_tests | generate_defects"),
        ("generate_defects",  "route_after_defects",      "human_review | finalize"),
        ("human_review",      "route_after_human_review", "finalize | END | generate_tests"),
    ]
    for source, fn, targets in _conditional:
        print(f"  {source:<22} [{fn}]")
        print(f"  {'':22}  → {targets}")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    print("\n  Checkpointing\n  " + "─" * 40)
    print(f"  Backend         : SqliteSaver (MemorySaver fallback)")
    print(f"  DB path         : {_CHECKPOINT_DB}")
    print(f"  interrupt_before: ['human_review']")
    print(f"  Resume via      : graph.invoke(Command(resume={{...}}), config=...)")

    # ── Mermaid diagram (LangGraph >= 0.2) ────────────────────────────────────
    print("\n  Mermaid Diagram\n  " + "─" * 40)
    try:
        mermaid = compiled_graph.get_graph().draw_mermaid()
        print(mermaid)
    except Exception as exc:
        print(f"  (mermaid unavailable: {exc})")

    print(f"\n{_SEP}\n")
