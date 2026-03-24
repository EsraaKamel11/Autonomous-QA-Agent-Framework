"""
tests/test_routing.py
──────────────────────
Unit tests for all 5 routing functions in qa_agent/routing.py.

Design
──────
- Pure logic tests: no LLM calls, no I/O, no external services.
- Each test builds a minimal QAAgentState using initial_state() and
  overrides only the fields the routing function under test reads.
- Parametrized with pytest.mark.parametrize for concise coverage.
"""

from __future__ import annotations

import pytest

from qa_agent.state import (
    CoverageReport,
    Defect,
    ExecutionReport,
    FailureType,
    MergeRecommendation,
    PRMetadata,
    QAAgentState,
    RepairAttempt,
    Severity,
    TestResult,
    initial_state,
)
from qa_agent.routing import (
    route_after_coverage,
    route_after_defects,
    route_after_execution,
    route_after_healing,
    route_after_human_review,
)
from langgraph.graph import END


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_PR_META = PRMetadata(
    pr_number=1,
    repo="test/repo",
    head_sha="abc123",
    base_sha="def456",
    author="tester",
    title="Test PR",
)


def _state(**overrides) -> QAAgentState:
    """Return initial_state() with specific fields overridden."""
    state = initial_state(_PR_META)
    state.update(overrides)
    return state


def _report(failed: int = 0, errors: int = 0, passed: int = 5) -> ExecutionReport:
    """Build a minimal ExecutionReport with the requested failure counts."""
    total = passed + failed + errors
    results = []
    for i in range(failed):
        results.append(TestResult(
            test_id=f"test::failure_{i}",
            status="failed",
            failure_type=FailureType.ASSERTION,
            error_message=f"AssertionError: mock failure {i}",
        ))
    for i in range(errors):
        results.append(TestResult(
            test_id=f"test::error_{i}",
            status="error",
            failure_type=FailureType.ENVIRONMENT,
            error_message=f"ConnectionError: mock error {i}",
        ))
    for i in range(passed):
        results.append(TestResult(test_id=f"test::pass_{i}", status="passed"))
    return ExecutionReport(
        total=total, passed=passed, failed=failed, errors=errors,
        per_test_results=results,
    )


def _coverage(score: float, verdict: MergeRecommendation = MergeRecommendation.APPROVE
              ) -> CoverageReport:
    return CoverageReport(
        completeness_score=score,
        scenario_depth_score=score,
        assertion_quality_score=score,
        regression_risk_score=score,
        overall_score=score,
        merge_recommendation=verdict,
        gaps=[f"gap_{i}" for i in range(3)],
        judge_reasoning="test reasoning",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. route_after_execution
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("failed,errors,retry,max_ret,expected", [
    # ── Happy path: all tests passed → evaluate_coverage ─────────────────────
    (0, 0, 0, 3, "evaluate_coverage"),
    # ── Failures present + retries available → self_heal ─────────────────────
    (2, 0, 0, 3, "self_heal"),
    (0, 1, 0, 3, "self_heal"),          # error counts as failure
    (3, 2, 1, 3, "self_heal"),          # retry_count < max_retries
    # ── Failures present + max retries exhausted → evaluate_coverage ─────────
    (2, 0, 3, 3, "evaluate_coverage"),  # retry_count == max_retries
    (1, 1, 5, 3, "evaluate_coverage"),  # retry_count > max_retries
    # ── No execution report → default to evaluate_coverage ───────────────────
    (None, None, 0, 3, "evaluate_coverage"),  # report is None
])
def test_route_after_execution(failed, errors, retry, max_ret, expected):
    if failed is None:
        state = _state(execution_report=None, retry_count=retry, max_retries=max_ret)
    else:
        state = _state(
            execution_report=_report(failed=failed, errors=errors),
            retry_count=retry,
            max_retries=max_ret,
        )
    assert route_after_execution(state) == expected


# ──────────────────────────────────────────────────────────────────────────────
# 2. route_after_healing
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("escalation,expected", [
    # ── No escalation → retry execute_tests ──────────────────────────────────
    (False, "execute_tests"),
    # ── Escalation required → human_review ───────────────────────────────────
    (True,  "human_review"),
])
def test_route_after_healing_escalation(escalation, expected):
    state = _state(
        human_escalation_required=escalation,
        escalation_reason="ENVIRONMENT failure" if escalation else None,
        retry_count=1,
    )
    assert route_after_healing(state) == expected


def test_route_after_healing_with_repair_attempts():
    """Partial patch success still returns execute_tests (not escalated)."""
    attempts = [
        RepairAttempt(
            test_id="test_a", attempt_number=1,
            failure_type=FailureType.ASSERTION,
            patch_applied="Updated expected value", success=True,
        ),
        RepairAttempt(
            test_id="test_b", attempt_number=1,
            failure_type=FailureType.SCHEMA,
            patch_applied="Removed extra field access", success=False,
        ),
    ]
    state = _state(
        human_escalation_required=False,
        repair_attempts=attempts,
        retry_count=1,
    )
    assert route_after_healing(state) == "execute_tests"


def test_route_after_healing_no_attempts():
    """Zero repair attempts but no escalation still retries."""
    state = _state(
        human_escalation_required=False,
        repair_attempts=[],
        retry_count=2,
    )
    assert route_after_healing(state) == "execute_tests"


# ──────────────────────────────────────────────────────────────────────────────
# 3. route_after_coverage
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("score,aug_cycle,max_aug,expected", [
    # ── Score below 7.0, budget available → augment ───────────────────────────
    (5.0, 0, 2, "augment_tests"),
    (6.9, 1, 2, "augment_tests"),
    (0.0, 0, 3, "augment_tests"),
    # ── Score at or above 7.0 → generate_defects ─────────────────────────────
    (7.0, 0, 2, "generate_defects"),
    (8.5, 0, 2, "generate_defects"),
    (10.0, 0, 2, "generate_defects"),
    # ── Budget exhausted → generate_defects regardless of score ──────────────
    (5.0, 2, 2, "generate_defects"),   # aug_cycle == max_aug
    (3.0, 3, 2, "generate_defects"),   # aug_cycle > max_aug
    # ── No coverage report → generate_defects ────────────────────────────────
    (None, 0, 2, "generate_defects"),
])
def test_route_after_coverage(score, aug_cycle, max_aug, expected):
    cov = _coverage(score) if score is not None else None
    state = _state(
        coverage_report=cov,
        augmentation_cycle=aug_cycle,
        max_augmentation_cycles=max_aug,
    )
    assert route_after_coverage(state) == expected


# ──────────────────────────────────────────────────────────────────────────────
# 4. route_after_defects
# ──────────────────────────────────────────────────────────────────────────────

def _defect(severity: Severity) -> Defect:
    return Defect(
        title=f"{severity.value} defect",
        severity=severity,
        affected_endpoint="/api/test",
        test_id="test_xyz",
        error_detail="mock error detail",
    )


def _env_failure_result() -> TestResult:
    return TestResult(
        test_id="test_env_001",
        status="failed",
        failure_type=FailureType.ENVIRONMENT,
        error_message="ConnectionRefusedError: [Errno 111]",
    )


@pytest.mark.parametrize("defects,env_failures,expected", [
    # ── No defects, no env failures → finalize ────────────────────────────────
    ([], False, "finalize"),
    # ── CRITICAL defect → human_review ───────────────────────────────────────
    ([_defect(Severity.CRITICAL)], False, "human_review"),
    # ── HIGH defect only (not critical) → finalize ───────────────────────────
    ([_defect(Severity.HIGH)], False, "finalize"),
    # ── MEDIUM + CRITICAL → human_review (critical wins) ─────────────────────
    ([_defect(Severity.MEDIUM), _defect(Severity.CRITICAL)], False, "human_review"),
    # ── ENVIRONMENT failure in latest report → human_review ──────────────────
    ([], True, "human_review"),
    # ── No report, no defects → finalize ─────────────────────────────────────
    ([], False, "finalize"),
])
def test_route_after_defects(defects, env_failures, expected):
    if env_failures:
        report = ExecutionReport(
            total=1, passed=0, failed=1,
            per_test_results=[_env_failure_result()],
        )
    else:
        report = _report(failed=0)

    state = _state(defects=defects, execution_report=report)
    assert route_after_defects(state) == expected


def test_route_after_defects_no_report():
    """No execution report + no defects → finalize (not a crash)."""
    state = _state(defects=[], execution_report=None)
    assert route_after_defects(state) == "finalize"


# ──────────────────────────────────────────────────────────────────────────────
# 5. route_after_human_review
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("commit_status,expected", [
    # ── APPROVE → finalize ────────────────────────────────────────────────────
    ("success",  "finalize"),
    # ── REJECT → END ─────────────────────────────────────────────────────────
    ("failure",  END),
    # ── MODIFY → generate_tests ──────────────────────────────────────────────
    ("pending",  "generate_tests"),
    # ── Unknown status → finalize (defensive default) ────────────────────────
    ("unknown",  "finalize"),
    ("",         "finalize"),
])
def test_route_after_human_review(commit_status, expected):
    state = _state(commit_status=commit_status)
    assert route_after_human_review(state) == expected


def test_route_after_human_review_default_is_finalize():
    """Missing commit_status key defaults to 'failure' then returns END."""
    state = initial_state(_PR_META)
    # initial_state sets commit_status = "pending" (CommitStatus.PENDING.value)
    # Change to something unexpected to test defensive path
    state["commit_status"] = "xyzzy_invalid"
    result = route_after_human_review(state)
    assert result == "finalize"


def test_route_after_human_review_approve_produces_finalize():
    """Sanity-check the approve flow end-to-end with a real state object."""
    state = _state(commit_status="success")
    assert route_after_human_review(state) == "finalize"
    assert route_after_human_review(state) != END
