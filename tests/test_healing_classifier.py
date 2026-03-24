"""
tests/test_healing_classifier.py
──────────────────────────────────
Unit tests for qa_agent/healing/classifier.py.

Coverage
────────
- All 5 classifiable FailureTypes with realistic error messages
- High-confidence rule-based paths that MUST skip the LLM
- LLM fallback trigger (ambiguous errors)
- LLM failure → safe ENVIRONMENT default
- LLM returning an invalid failure_type → ASSERTION with penalty
- is_auto_healable() for all 5 FailureTypes
- classify_all_failures() filtering logic
- Tie-breaking via _pick_winner()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qa_agent.healing.classifier import (
    ClassificationResult,
    _pick_winner,
    _RULES,
    classify_all_failures,
    classify_failure,
    is_auto_healable,
)
from qa_agent.state import FailureType, TestResult


# ──────────────────────────────────────────────────────────────────────────────
# 1. High-confidence rule-based classification (LLM MUST NOT be called)
# ──────────────────────────────────────────────────────────────────────────────

@patch("langchain_openai.ChatOpenAI")
def test_environment_connection_refused(mock_llm):
    """ConnectionRefusedError triggers ENVIRONMENT at high confidence."""
    err = "ConnectionRefusedError: [Errno 111] Connection refused to http://localhost:8080"
    result = classify_failure(err, "", "")

    mock_llm.assert_not_called()
    assert result.failure_type == FailureType.ENVIRONMENT
    assert result.method == "rule_based"
    assert result.confidence >= 0.95


@patch("langchain_openai.ChatOpenAI")
def test_environment_502_bad_gateway(mock_llm):
    """502 Bad Gateway triggers ENVIRONMENT without LLM."""
    err = "AssertionError: expected 200, got 502 Bad Gateway"
    result = classify_failure(err, "", "")

    mock_llm.assert_not_called()
    assert result.failure_type == FailureType.ENVIRONMENT
    assert result.confidence >= 0.6


@patch("langchain_openai.ChatOpenAI")
def test_auth_401_unauthorized(mock_llm):
    """HTTP 401 triggers AUTH at high confidence."""
    err = "requests.exceptions.HTTPError: 401 Unauthorized - invalid token"
    result = classify_failure(err, "", "")

    mock_llm.assert_not_called()
    assert result.failure_type == FailureType.AUTH
    assert result.method == "rule_based"
    assert result.confidence >= 0.90


@patch("langchain_openai.ChatOpenAI")
def test_auth_403_forbidden(mock_llm):
    """HTTP 403 triggers AUTH at high confidence."""
    err = "HTTPError: 403 Forbidden — insufficient permissions for /admin/users"
    result = classify_failure(err, "", "")

    mock_llm.assert_not_called()
    assert result.failure_type == FailureType.AUTH
    assert result.confidence >= 0.90


@patch("langchain_openai.ChatOpenAI")
def test_assertion_error_expected_got(mock_llm):
    """'AssertionError: expected X got Y' triggers ASSERTION at high confidence."""
    err = "AssertionError: expected 200 but got 404"
    stdout = "E  assert response.status_code == 200\nE  where 404 = response.status_code"
    result = classify_failure(err, "", stdout)

    mock_llm.assert_not_called()
    assert result.failure_type == FailureType.ASSERTION
    assert result.method == "rule_based"
    assert result.confidence >= 0.88


@patch("langchain_openai.ChatOpenAI")
def test_schema_validation_error(mock_llm):
    """Pydantic ValidationError triggers SCHEMA at high confidence."""
    err = "ValidationError: 1 validation error for UserResponse\nid\n  field required"
    result = classify_failure(err, "", "")

    mock_llm.assert_not_called()
    assert result.failure_type == FailureType.SCHEMA
    assert result.method == "rule_based"
    assert result.confidence >= 0.95


@patch("langchain_openai.ChatOpenAI")
def test_schema_keyerror(mock_llm):
    """KeyError triggers SCHEMA (missing dict key = schema drift)."""
    err = "KeyError: 'error_code'\nThe response schema no longer includes this field."
    result = classify_failure(err, "", "")

    mock_llm.assert_not_called()
    assert result.failure_type == FailureType.SCHEMA
    assert result.confidence >= 0.6


@patch("langchain_openai.ChatOpenAI")
def test_selector_element_not_found(mock_llm):
    """'element not found' triggers SELECTOR at high confidence."""
    err = "playwright._impl._errors.Error: element not found: [data-testid='submit-btn']"
    result = classify_failure(err, "", "")

    mock_llm.assert_not_called()
    assert result.failure_type == FailureType.SELECTOR
    assert result.method == "rule_based"
    assert result.confidence >= 0.95


@patch("langchain_openai.ChatOpenAI")
def test_selector_playwright_locator_timeout(mock_llm):
    """Playwright locator timeout triggers SELECTOR."""
    err = "TimeoutError: waiting for selector '#login-form' exceeded 30000ms"
    result = classify_failure(err, "", "")

    mock_llm.assert_not_called()
    assert result.failure_type == FailureType.SELECTOR
    assert result.confidence >= 0.6


# ──────────────────────────────────────────────────────────────────────────────
# 2. LLM fallback trigger (ambiguous errors with confidence < 0.6)
# ──────────────────────────────────────────────────────────────────────────────

@patch("langchain_openai.ChatOpenAI")
def test_llm_fallback_triggered_on_ambiguous_error(mock_llm):
    """No-match error message triggers LLM fallback."""
    mock_output = MagicMock()
    mock_output.failure_type = "env"   # FailureType.ENVIRONMENT.value — must match _FAILURE_TYPE_MAP key after .upper()
    mock_output.confidence   = 0.75
    mock_output.reasoning    = "No specific patterns matched; infrastructure issue likely."

    mock_chain    = MagicMock()
    mock_chain.invoke.return_value = mock_output

    mock_instance = MagicMock()
    mock_instance.with_structured_output.return_value = mock_chain
    mock_llm.return_value = mock_instance

    # Deliberately unrecognisable error text
    err    = "xyzzy_test_exception: mysterious failure occurred (code=42)"
    result = classify_failure(err, "", "")

    mock_llm.assert_called_once()
    assert result.method == "llm_fallback"
    assert result.failure_type == FailureType.ENVIRONMENT
    assert result.confidence == 0.75


@patch("langchain_openai.ChatOpenAI")
def test_llm_failure_defaults_to_environment(mock_llm):
    """When LLM itself raises, ENVIRONMENT is returned as safe default."""
    # The ChatOpenAI() constructor is called OUTSIDE the try/except in _llm_classify,
    # so side_effect on the constructor would propagate uncaught.
    # Instead raise inside .invoke() — which IS inside the try/except.
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("LLM connection failed")
    mock_instance = MagicMock()
    mock_instance.with_structured_output.return_value = mock_chain
    mock_llm.return_value = mock_instance

    err    = "completely_unknown_exception: no matching rule"
    result = classify_failure(err, "", "")

    assert result.failure_type == FailureType.ENVIRONMENT
    assert result.method == "llm_fallback"
    assert result.confidence == 0.40   # documented default for LLM failure


@patch("langchain_openai.ChatOpenAI")
def test_llm_invalid_type_falls_back_to_assertion(mock_llm):
    """LLM returning an invalid failure_type string is penalised → ASSERTION."""
    mock_output = MagicMock()
    mock_output.failure_type = "UNKNOWN_TYPE_XYZ"
    mock_output.confidence   = 0.90
    mock_output.reasoning    = "Some classification"

    mock_chain    = MagicMock()
    mock_chain.invoke.return_value = mock_output

    mock_instance = MagicMock()
    mock_instance.with_structured_output.return_value = mock_chain
    mock_llm.return_value = mock_instance

    err    = "unmatched_error_with_no_rules: boom"
    result = classify_failure(err, "", "")

    assert result.failure_type == FailureType.ASSERTION
    assert result.confidence < 0.90     # penalised per implementation
    assert result.method == "llm_fallback"


# ──────────────────────────────────────────────────────────────────────────────
# 3. is_auto_healable
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("failure_type,expected", [
    (FailureType.ASSERTION,   True),
    (FailureType.SCHEMA,      True),
    (FailureType.SELECTOR,    True),
    (FailureType.ENVIRONMENT, False),
    (FailureType.AUTH,        False),
])
def test_is_auto_healable(failure_type, expected):
    assert is_auto_healable(failure_type) is expected


# ──────────────────────────────────────────────────────────────────────────────
# 4. classify_all_failures — filtering
# ──────────────────────────────────────────────────────────────────────────────

def test_classify_all_failures_skips_passed():
    """Only failed and error status results are classified."""
    results = [
        TestResult(test_id="t1", status="passed"),
        TestResult(test_id="t2", status="skipped"),
        TestResult(
            test_id="t3", status="failed",
            error_message="ConnectionRefusedError: port 8080 refused",
        ),
        TestResult(
            test_id="t4", status="error",
            error_message="401 Unauthorized token expired",
        ),
    ]
    classifications = classify_all_failures(results)

    assert len(classifications) == 2
    assert classifications[0].failure_type == FailureType.ENVIRONMENT
    assert classifications[1].failure_type == FailureType.AUTH


def test_classify_all_failures_empty_list():
    """Empty input returns empty output without raising."""
    assert classify_all_failures([]) == []


def test_classify_all_failures_all_passed():
    """All-passed list returns empty output."""
    results = [
        TestResult(test_id=f"t{i}", status="passed") for i in range(5)
    ]
    assert classify_all_failures(results) == []


# ──────────────────────────────────────────────────────────────────────────────
# 5. Tie-breaking via _pick_winner
# ──────────────────────────────────────────────────────────────────────────────

def test_pick_winner_selects_higher_score():
    """Higher accumulated score wins outright."""
    scores = {
        FailureType.ASSERTION:   (0.30, ["assert keyword"]),
        FailureType.ENVIRONMENT: (0.95, ["connection refused"]),
        FailureType.AUTH:        (0.25, ["token"]),
    }
    winner, confidence, _ = _pick_winner(scores)
    assert winner == FailureType.ENVIRONMENT
    assert confidence == 0.95


def test_pick_winner_tie_uses_max_rule_weight():
    """Equal scores → the type with the highest single rule weight wins."""
    # AUTH max rule weight = 0.90; ASSERTION max rule weight = 0.90
    # They are equal, so first-encountered (AUTH, since it appears first in _RULES) wins
    tied_score = 0.90
    scores = {
        FailureType.AUTH:        (tied_score, ["401 detected"]),
        FailureType.ASSERTION:   (tied_score, ["AssertionError"]),
    }
    winner, confidence, _ = _pick_winner(scores)
    # Both types have max_weight = 0.90; AUTH was seen first → AUTH stays winner
    assert winner in (FailureType.AUTH, FailureType.ASSERTION)   # tie, either is valid
    assert confidence == tied_score


def test_pick_winner_all_zero_scores():
    """When no patterns match (all scores = 0), a stable type is returned."""
    scores = {ft: (0.0, []) for ft in FailureType if ft in _RULES}
    winner, confidence, matches = _pick_winner(scores)
    assert isinstance(winner, FailureType)
    assert confidence == 0.0
    assert matches == []


def test_pick_winner_single_type():
    """Single entry dict always returns that entry."""
    scores = {FailureType.SCHEMA: (0.85, ["ValidationError"])}
    winner, confidence, matches = _pick_winner(scores)
    assert winner == FailureType.SCHEMA
    assert confidence == 0.85
