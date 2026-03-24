"""
qa_agent/nodes/human_review.py
────────────────────────────────
Human-in-the-Loop checkpoint — node 11 of 12.

Decision boundary
─────────────────
Suspends the graph and hands control to a human reviewer.
Answers: "Is this escalation resolved?" — after the human responds.
Never fixes code, never evaluates coverage, never generates tests.

Inputs consumed from state
──────────────────────────
  state["escalation_reason"]         str | None
  state["defects"]                   List[Defect]  — for critical-defect escalations
  state["coverage_report"]           CoverageReport | None
  state["errors"]                    List[AgentError]
  state["pr_metadata"]               PRMetadata

State keys written
──────────────────
  commit_status   str   — "success" | "failure" based on human decision
  current_phase   str

Checkpoint behaviour
────────────────────
LangGraph's interrupt() suspends graph execution and serialises the full
state to the SqliteSaver checkpoint.  The graph resumes when the caller
invokes graph.invoke(Command(resume=decision_dict), config=config).

Human decision dict schema
──────────────────────────
  {
    "action": "approve" | "reject" | "modify",
    "comment": "<optional free-text>"
  }

Escalation JSONL log
────────────────────
Every escalation is appended to QA_AGENT_LOG_DIR/escalations.jsonl in
addition to the normal decision log — creating an append-only audit trail.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

from qa_agent.observability import log_node_execution
from qa_agent.state import (
    Defect,
    QAAgentState,
    Severity,
)

_NODE_NAME = "human_review"
_LOG_DIR   = os.environ.get("QA_AGENT_LOG_DIR", "./logs")
_ESC_LOG   = os.path.join(_LOG_DIR, "escalations.jsonl")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _critical_defects(defects: List[Defect]) -> List[Defect]:
    return [d for d in defects if d.severity == Severity.CRITICAL]


def _build_escalation_payload(state: QAAgentState) -> Dict[str, Any]:
    """
    Construct the payload passed to interrupt().
    This is what a human reviewer sees when they inspect the suspended graph.
    """
    pr_meta  = state["pr_metadata"]
    coverage = state.get("coverage_report")
    defects  = state.get("defects") or []
    errors   = state.get("errors")  or []
    critical = _critical_defects(defects)

    return {
        "escalation_reason": state.get("escalation_reason") or "Unspecified escalation",
        "pr_number":         pr_meta.pr_number,
        "repo":              pr_meta.repo,
        "head_sha":          pr_meta.head_sha,
        "coverage_score":    coverage.overall_score if coverage else None,
        "merge_recommendation": coverage.merge_recommendation.value if coverage else None,
        "critical_defects":  [
            {
                "title":    d.title,
                "severity": d.severity.value,
                "endpoint": d.affected_endpoint,
                "jira_key": d.jira_key,
            }
            for d in critical
        ],
        "agent_errors": [
            {
                "agent":       e.agent,
                "error":       e.error[:200],
                "recoverable": e.recoverable,
            }
            for e in errors[-5:]   # last 5 errors for context
        ],
        "retry_count":    state.get("retry_count", 0),
        "allowed_actions": ["approve", "reject", "modify"],
        "instructions": (
            "Review the escalation above and resume the graph with:\n"
            "  graph.invoke(Command(resume={'action': 'approve'|'reject'|'modify',\n"
            "                              'comment': '<optional note>'}), config=config)"
        ),
    }


def _append_escalation_log(
    payload:    Dict[str, Any],
    start_ts:   float,
) -> None:
    """Append one entry to the append-only escalations.jsonl audit log."""
    try:
        Path(_LOG_DIR).mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "type":         "human_escalation",
            "pr_number":    payload.get("pr_number"),
            "repo":         payload.get("repo"),
            "reason":       payload.get("escalation_reason"),
            "coverage":     payload.get("coverage_score"),
            "critical_defects": len(payload.get("critical_defects", [])),
            "payload_snapshot": {
                k: str(v)[:200]
                for k, v in payload.items()
                if k != "instructions"
            },
        }
        with open(_ESC_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass   # never crash on logging failure


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def human_review(state: QAAgentState) -> dict:
    """
    HITL checkpoint: suspend graph execution until a human reviewer responds.

    The full graph state is checkpointed via SqliteSaver before interrupt()
    is called.  The graph can be resumed from a different process or thread
    using graph.invoke(Command(resume=decision), config=checkpoint_config).

    Pipeline position: Node 11 (conditional — reached on critical failures,
    environment errors, max retries exceeded, or BLOCK verdict).
    Input:  state["escalation_reason"], state["defects"], state["coverage_report"]
    Output: state["commit_status"], state["current_phase"]

    LLM calls: NONE.
    interrupt() call: YES — suspends graph here.
    """
    start_ts = time.time()

    # ── Build escalation payload ─────────────────────────────────────────────
    payload = _build_escalation_payload(state)

    # ── Write escalation to append-only audit log BEFORE interrupt() ─────────
    _append_escalation_log(payload, start_ts)

    # ── Log to decision JSONL BEFORE interrupt() ──────────────────────────────
    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "escalation_reason":  (state.get("escalation_reason") or "")[:150],
            "critical_defects":   len(_critical_defects(state.get("defects") or [])),
            "retry_count":        state.get("retry_count", 0),
            "coverage_score":     (
                state["coverage_report"].overall_score
                if state.get("coverage_report") else None
            ),
        },
        output_summary={
            "action": "SUSPENDED — awaiting human decision",
            "escalation_log": _ESC_LOG,
        },
        routing_hint="finalize (after human resumes)",
        pr_number=state["pr_metadata"].pr_number,
        duration_ms=(time.time() - start_ts) * 1000,
    )

    # ── Suspend graph — control returns here only after graph.invoke(resume=…) ─
    human_decision: Dict[str, Any] = interrupt(payload)

    # ── Process human decision ────────────────────────────────────────────────
    action  = (human_decision or {}).get("action", "reject")
    comment = (human_decision or {}).get("comment", "")

    if action == "approve":
        commit_status = "success"
    elif action == "modify":
        # "modify" means the human is taking manual action; treat as pending
        commit_status = "pending"
    else:
        # "reject" or unknown → failure
        commit_status = "failure"

    # Log post-resume
    log_node_execution(
        node_name=f"{_NODE_NAME}.resumed",
        input_summary={"human_action": action, "comment": comment[:150]},
        output_summary={"commit_status": commit_status},
        routing_hint="finalize",
        pr_number=state["pr_metadata"].pr_number,
        duration_ms=(time.time() - start_ts) * 1000,
    )

    return {
        "commit_status": commit_status,
        "current_phase": "human_resolved",
    }
