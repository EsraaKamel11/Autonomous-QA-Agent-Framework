"""
qa_agent/routing.py
─────────────────────
All conditional edge functions for the QA Agent LangGraph state machine.

Design principles
─────────────────
1.  Each function reads ONLY from state — never modifies it.
2.  Every return value is a string (or END sentinel) that exists as a key
    in the path_map passed to add_conditional_edges.  No routing function
    ever returns an unmapped string.
3.  All routing decisions read from execution_report (latest cycle), NOT
    from test_results (accumulated history), to avoid counting stale failures.
4.  The self_heal → execute_tests retry loop is bounded here, not inside the
    self_heal node — the node patches; the router decides whether to retry.
5.  Every decision is logged at INFO level for the JSONL decision log audit.

Routing map summary
───────────────────
execute_tests → self_heal            failures > 0 AND retry_count < max_retries
             → evaluate_coverage     else

self_heal     → human_review         human_escalation_required is True
             → execute_tests         patches applied (retry loop)

evaluate_coverage → augment_tests    score < 7.0 AND aug_cycle < max_aug_cycles
                 → generate_defects  else

generate_defects → human_review      any CRITICAL defect OR ENVIRONMENT failure
                → finalize           else

human_review  → finalize             commit_status == "success"  (APPROVE)
             → END                   commit_status == "failure"  (REJECT)
             → generate_tests        commit_status == "pending"  (MODIFY)
"""

from __future__ import annotations

import logging

from langgraph.graph import END

from qa_agent.state import (
    FailureType,
    MergeRecommendation,
    QAAgentState,
    Severity,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  After execute_tests
# ──────────────────────────────────────────────────────────────────────────────

def route_after_execution(state: QAAgentState) -> str:
    """
    Decide whether to attempt self-healing or proceed to coverage evaluation.

    Reads from state["execution_report"] (latest cycle) — NOT state["test_results"]
    (accumulated history) — so that prior-cycle failures do not trigger
    unnecessary retries after a successful re-run.

    Returns
    -------
    "self_heal"         failures > 0 AND retry_count < max_retries
    "evaluate_coverage" all tests passed  OR  max retries already reached
    """
    report      = state.get("execution_report")
    retry_count = state.get("retry_count",  0)
    max_retries = state.get("max_retries",  3)

    # No report at all — safest path is to evaluate what we have
    if report is None:
        logger.warning(
            "route_after_execution [pr=%s]: no execution_report — defaulting to evaluate_coverage",
            state["pr_metadata"].pr_number,
        )
        return "evaluate_coverage"

    failed_count = report.failed + report.errors

    if failed_count > 0 and retry_count < max_retries:
        logger.info(
            "route_after_execution [pr=%s]: %d failures, retry %d/%d → self_heal",
            state["pr_metadata"].pr_number, failed_count, retry_count, max_retries,
        )
        return "self_heal"

    if failed_count > 0:
        logger.info(
            "route_after_execution [pr=%s]: %d failures but max retries (%d) reached "
            "→ evaluate_coverage (with failures)",
            state["pr_metadata"].pr_number, failed_count, max_retries,
        )
    else:
        logger.info(
            "route_after_execution [pr=%s]: all %d tests passed → evaluate_coverage",
            state["pr_metadata"].pr_number, report.total,
        )

    return "evaluate_coverage"


# ──────────────────────────────────────────────────────────────────────────────
# 2.  After self_heal
# ──────────────────────────────────────────────────────────────────────────────

def route_after_healing(state: QAAgentState) -> str:
    """
    Decide whether to retry execution with patched tests or escalate to humans.

    The self_heal node sets human_escalation_required=True when it encounters
    a non-healable failure type (ENVIRONMENT, AUTH) or when the LLM chooses
    to escalate rather than patch.  This check takes priority over all others.

    If no escalation is needed, we always retry execution — even if not all
    patches succeeded — because a partial patch may unblock other tests.

    Returns
    -------
    "human_review"  human_escalation_required is True
    "execute_tests" patches applied; re-run the test suite
    """
    if state.get("human_escalation_required", False):
        reason = (state.get("escalation_reason") or "unspecified")[:120]
        logger.info(
            "route_after_healing [pr=%s]: escalation required (%s) → human_review",
            state["pr_metadata"].pr_number, reason,
        )
        return "human_review"

    # Sanity-check: were any repair attempts actually made?
    retry_count     = state.get("retry_count", 0)
    repair_attempts = state.get("repair_attempts") or []
    latest_attempts = [a for a in repair_attempts if a.attempt_number == retry_count]
    patched         = [a for a in latest_attempts if a.success]

    logger.info(
        "route_after_healing [pr=%s]: retry=%d, attempts=%d, patched=%d → execute_tests",
        state["pr_metadata"].pr_number, retry_count,
        len(latest_attempts), len(patched),
    )
    return "execute_tests"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  After evaluate_coverage
# ──────────────────────────────────────────────────────────────────────────────

def route_after_coverage(state: QAAgentState) -> str:
    """
    Decide whether to augment the test suite or move on to defect reporting.

    Augmentation is attempted when the judge score is below 7.0 AND we have
    not yet exhausted the augmentation budget (default: 2 cycles).

    A BLOCK verdict alone does NOT trigger a third path here — it passes through
    to generate_defects, which then routes to human_review via route_after_defects
    (critical defects → human_review).  This keeps the routing tree shallow.

    Returns
    -------
    "augment_tests"    coverage score < 7.0 AND aug_cycle < max_augmentation_cycles
    "generate_defects" else (score ≥ 7.0, or augmentation budget exhausted)
    """
    coverage  = state.get("coverage_report")
    aug_cycle = state.get("augmentation_cycle",      0)
    max_aug   = state.get("max_augmentation_cycles", 2)

    if coverage is None:
        logger.warning(
            "route_after_coverage [pr=%s]: no coverage_report → generate_defects",
            state["pr_metadata"].pr_number,
        )
        return "generate_defects"

    score   = coverage.overall_score
    verdict = coverage.merge_recommendation.value

    if score < 7.0 and aug_cycle < max_aug:
        logger.info(
            "route_after_coverage [pr=%s]: score=%.2f < 7.0, aug_cycle=%d/%d → augment_tests",
            state["pr_metadata"].pr_number, score, aug_cycle, max_aug,
        )
        return "augment_tests"

    logger.info(
        "route_after_coverage [pr=%s]: score=%.2f, verdict=%s, aug_cycle=%d/%d → generate_defects",
        state["pr_metadata"].pr_number, score, verdict, aug_cycle, max_aug,
    )
    return "generate_defects"


# ──────────────────────────────────────────────────────────────────────────────
# 4.  After generate_defects
# ──────────────────────────────────────────────────────────────────────────────

def route_after_defects(state: QAAgentState) -> str:
    """
    Escalate to human review when critical defects or infrastructure failures exist.

    Two triggers for human escalation:
      a) Any Defect with severity == CRITICAL
         (auth failures, data loss, system unavailable)
      b) Any ENVIRONMENT failure in the latest execution report
         (target service is down; no code change will fix it)

    Returns
    -------
    "human_review"  CRITICAL defect exists OR ENVIRONMENT failure in latest report
    "finalize"      else
    """
    defects = state.get("defects") or []
    report  = state.get("execution_report")
    pr_num  = state["pr_metadata"].pr_number

    # (a) Critical severity defects
    critical = [d for d in defects if d.severity == Severity.CRITICAL]
    if critical:
        logger.info(
            "route_after_defects [pr=%s]: %d critical defect(s) → human_review "
            "(titles: %s)",
            pr_num, len(critical),
            ", ".join(d.title[:40] for d in critical[:3]),
        )
        return "human_review"

    # (b) Environment failures in latest execution cycle
    if report:
        env_failures = [
            r for r in report.per_test_results
            if r.failure_type == FailureType.ENVIRONMENT
            and r.status in ("failed", "error")
        ]
        if env_failures:
            logger.info(
                "route_after_defects [pr=%s]: %d environment failure(s) in latest run "
                "→ human_review",
                pr_num, len(env_failures),
            )
            return "human_review"

    logger.info(
        "route_after_defects [pr=%s]: no critical issues → finalize",
        pr_num,
    )
    return "finalize"


# ──────────────────────────────────────────────────────────────────────────────
# 5.  After human_review
# ──────────────────────────────────────────────────────────────────────────────

def route_after_human_review(state: QAAgentState) -> str:
    """
    Resume the pipeline based on the human reviewer's decision.

    The human_review node translates the human's action into commit_status:
      "approve" → commit_status = "success"
      "reject"  → commit_status = "failure"
      "modify"  → commit_status = "pending"

    Returns
    -------
    "finalize"       commit_status == "success"  (human approved)
    END              commit_status == "failure"  (human rejected — terminate)
    "generate_tests" commit_status == "pending"  (human requested modifications)
    "finalize"       default — always publish results even on ambiguous state
    """
    commit_status = state.get("commit_status", "failure")
    pr_num        = state["pr_metadata"].pr_number

    if commit_status == "success":
        logger.info(
            "route_after_human_review [pr=%s]: human approved → finalize",
            pr_num,
        )
        return "finalize"

    if commit_status == "failure":
        logger.info(
            "route_after_human_review [pr=%s]: human rejected → END",
            pr_num,
        )
        return END

    if commit_status == "pending":
        logger.info(
            "route_after_human_review [pr=%s]: human requested modifications → generate_tests",
            pr_num,
        )
        return "generate_tests"

    # Defensive default — unknown status, still publish what we have
    logger.warning(
        "route_after_human_review [pr=%s]: unexpected commit_status=%r → finalize (default)",
        pr_num, commit_status,
    )
    return "finalize"


# ──────────────────────────────────────────────────────────────────────────────
# Exported path maps — used in graph.py add_conditional_edges calls
# These must be kept in sync with the return values above.
# ──────────────────────────────────────────────────────────────────────────────

EXECUTION_PATH_MAP: dict = {
    "self_heal":        "self_heal",
    "evaluate_coverage":"evaluate_coverage",
}

HEALING_PATH_MAP: dict = {
    "execute_tests": "execute_tests",
    "human_review":  "human_review",
}

COVERAGE_PATH_MAP: dict = {
    "augment_tests":    "augment_tests",
    "generate_defects": "generate_defects",
}

DEFECTS_PATH_MAP: dict = {
    "human_review": "human_review",
    "finalize":     "finalize",
}

HUMAN_REVIEW_PATH_MAP: dict = {
    "finalize":       "finalize",
    END:               END,
    "generate_tests": "generate_tests",
}
