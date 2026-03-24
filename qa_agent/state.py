"""
qa_agent/state.py
─────────────────
Central state schema for the Autonomous QA Agent Framework.

All Pydantic models are defined here first; the LangGraph TypedDict
(QAAgentState) is assembled from them at the bottom.

Design principles
-----------------
- Append-only lists use operator.add as their LangGraph reducer so that
  parallel branches and retry loops accumulate results without clobbering.
- Scalar fields (coverage_report, change_manifest, etc.) use the default
  "last-write-wins" reducer — only one agent ever writes each of them.
- FailureType is kept as a plain StrEnum so it can be serialised to JSON
  without extra converters (needed by the JSONL decision log).
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class FailureType(str, Enum):
    """
    Failure taxonomy used by the SelfHealingAgent classifier.

    Healable automatically : ASSERTION, SCHEMA, SELECTOR
    Must escalate to human : ENVIRONMENT, AUTH
    Real bug found         : LOGIC
    """
    ASSERTION   = "assertion"   # wrong expected value → spec drift
    SCHEMA      = "schema"      # response structure mismatch → spec drift
    SELECTOR    = "selector"    # UI element not found → DOM change
    ENVIRONMENT = "env"         # 5xx, timeout, network → infra issue
    AUTH        = "auth"        # 401/403 → credentials/permission change
    LOGIC       = "logic"       # business logic failure → real defect


class ScenarioType(str, Enum):
    HAPPY_PATH  = "happy_path"
    EDGE_CASE   = "edge_case"
    ERROR_CASE  = "error_case"
    SECURITY    = "security"


class MergeRecommendation(str, Enum):
    APPROVE          = "APPROVE"
    REQUEST_CHANGES  = "REQUEST_CHANGES"
    BLOCK            = "BLOCK"


class CommitStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR   = "error"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


# ──────────────────────────────────────────────────────────────────────────────
# Input models (populated before the graph starts)
# ──────────────────────────────────────────────────────────────────────────────

class PRMetadata(BaseModel):
    """Immutable metadata about the pull request that triggered this run."""
    pr_number:   int
    repo:        str            # owner/repo format, e.g. "acme/api"
    head_sha:    str            # SHA of the PR branch tip
    base_sha:    str            # SHA of the target branch tip
    author:      str
    title:       str
    description: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Analysis output
# ──────────────────────────────────────────────────────────────────────────────

class AffectedEndpoint(BaseModel):
    """Single endpoint touched by the PR."""
    method:      str            # GET | POST | PUT | PATCH | DELETE
    path:        str            # e.g. /api/v1/users/{id}
    change_type: str            # added | modified | removed
    operation_id: Optional[str] = None


class ChangeManifest(BaseModel):
    """
    Output of CodeAnalystAgent.
    Answers: "What changed and how risky is it?"
    """
    affected_files:     List[str]
    affected_endpoints: List[AffectedEndpoint]
    affected_modules:   List[str]
    risk_score:         float = Field(
        ge=0.0, le=1.0,
        description="0.0 = trivial docs change; 1.0 = auth/payment logic rewrite",
    )
    change_summary:     str


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Spec parsing output
# ──────────────────────────────────────────────────────────────────────────────

class TestScenario(BaseModel):
    """
    A single test case mandate produced by SpecParserAgent.
    Answers: "What behaviors does the spec mandate?"
    """
    scenario_id:      str       # stable ID: {method}_{path}_{scenario_type}_{seq}
    endpoint:         str       # e.g. /api/v1/users/{id}
    method:           str       # HTTP method
    scenario_type:    ScenarioType
    description:      str
    preconditions:    List[str] = Field(default_factory=list)
    expected_behavior: str      # human-readable expected outcome
    expected_status:  Optional[int]   = None   # e.g. 200, 404
    expected_body:    Optional[dict]  = None   # partial body schema
    priority:         int = Field(default=2, ge=1, le=3)
    # 1 = critical, 2 = high, 3 = medium


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Test generation output
# ──────────────────────────────────────────────────────────────────────────────

class TestScript(BaseModel):
    """
    An executable test file produced by TestGeneratorAgent.
    Answers: "What code tests these behaviors?"
    """
    scenario_id:  str           # links back to TestScenario
    file_path:    str           # where the file will be written, e.g. /tmp/test_001.py
    framework:    str           # pytest | playwright
    content:      str           # full source code of the test file
    dependencies: List[str] = Field(default_factory=list)  # pip packages required
    chromadb_id:  Optional[str] = None  # set by store_tests node after storage


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Execution output
# ──────────────────────────────────────────────────────────────────────────────

class TestResult(BaseModel):
    """
    Result of a single test case from TestExecutorAgent.
    Answers: "Did this test pass?"
    """
    test_id:       str          # pytest node id or playwright test title
    scenario_id:   Optional[str] = None  # back-ref to TestScript.scenario_id
    status:        str          # passed | failed | error | skipped
    duration_ms:   float = 0.0
    error_message: Optional[str] = None
    failure_type:  Optional[FailureType] = None  # set by classifier
    stdout:        str = ""
    stderr:        str = ""


class ExecutionReport(BaseModel):
    """
    Aggregate statistics across all test runs in a single execution cycle.
    """
    total:            int = 0
    passed:           int = 0
    failed:           int = 0
    errors:           int = 0
    skipped:          int = 0
    duration_seconds: float = 0.0
    per_test_results: List[TestResult] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3b — Self-healing output
# ──────────────────────────────────────────────────────────────────────────────

class RepairAttempt(BaseModel):
    """
    Record of a single self-healing attempt by SelfHealingAgent.
    Answers: "What did I try to fix and did it work?"
    """
    test_id:        str
    attempt_number: int
    failure_type:   FailureType
    patch_applied:  str         # brief description of the patch
    success:        bool


class EscalationRequest(BaseModel):
    """
    Raised by SelfHealingAgent when a failure cannot be auto-repaired.
    Triggers human_review node via LangGraph interrupt().
    """
    reason:       str
    failure_type: FailureType
    context:      dict = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Coverage evaluation output
# ──────────────────────────────────────────────────────────────────────────────

class CoverageReport(BaseModel):
    """
    LLM-as-Judge output from CoverageEvaluatorAgent.
    Answers: "Is the coverage sufficient?"

    Scoring dimensions (each 0-10, averaged to overall_score):
    - completeness_score : all changed endpoints covered?
    - scenario_depth_score : happy + edge + error present?
    - assertion_quality_score : specific assertions vs. status-code-only?
    - regression_risk_score : high-risk changes adequately covered?
    """
    completeness_score:      float = Field(ge=0.0, le=10.0)
    scenario_depth_score:    float = Field(ge=0.0, le=10.0)
    assertion_quality_score: float = Field(ge=0.0, le=10.0)
    regression_risk_score:   float = Field(ge=0.0, le=10.0)
    overall_score:           float = Field(ge=0.0, le=10.0)
    gaps:                    List[str]       = Field(default_factory=list)
    judge_reasoning:         str             = ""
    merge_recommendation:    MergeRecommendation

    # Convenience aliases used in some routing/reporting code
    @property
    def score(self) -> float:
        return self.overall_score

    @property
    def verdict(self) -> MergeRecommendation:
        return self.merge_recommendation

    @property
    def reasoning(self) -> str:
        return self.judge_reasoning


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5 — Defect reporting output
# ──────────────────────────────────────────────────────────────────────────────

class Defect(BaseModel):
    """
    A single defect record produced by DefectReporterAgent.
    Answers: "How do I communicate this finding?"
    """
    title:              str
    severity:           Severity
    affected_endpoint:  str
    test_id:            str
    error_detail:       str
    reproduction_steps: List[str] = Field(default_factory=list)
    jira_key:           Optional[str] = None  # set after Jira ticket creation


# ──────────────────────────────────────────────────────────────────────────────
# Error tracking (any phase)
# ──────────────────────────────────────────────────────────────────────────────

class AgentError(BaseModel):
    """
    Structured error emitted by any agent node.
    Accumulated via operator.add so all errors across retries are preserved.
    """
    agent:       str
    phase:       str
    error:       str
    recoverable: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Root state — the single TypedDict that LangGraph threads through every node
# ──────────────────────────────────────────────────────────────────────────────

class QAAgentState(TypedDict):
    # ── Input (set once before graph.invoke) ─────────────────────────────────
    pr_metadata: PRMetadata

    # ── Phase 1: Analysis ────────────────────────────────────────────────────
    change_manifest:  Optional[ChangeManifest]   # CodeAnalystAgent output
    test_scenarios:   List[TestScenario]         # SpecParserAgent output

    # ── Phase 2: Generation ──────────────────────────────────────────────────
    # historical_tests : retrieved from ChromaDB by retrieve_history node
    # generated_tests  : new scripts written by TestGeneratorAgent
    historical_tests: List[TestScript]
    generated_tests:  List[TestScript]

    # ── Phase 3: Execution & Healing ─────────────────────────────────────────
    # operator.add reducer → each execution cycle appends, never overwrites
    test_results:    Annotated[List[TestResult],    operator.add]
    repair_attempts: Annotated[List[RepairAttempt], operator.add]

    # Latest aggregate report (overwritten each execution cycle)
    execution_report: Optional[ExecutionReport]

    # ── Phase 4: Coverage Evaluation ─────────────────────────────────────────
    coverage_report: Optional[CoverageReport]

    # ── Phase 5: Defect Reporting ─────────────────────────────────────────────
    defects:              Annotated[List[Defect], operator.add]
    jira_tickets_created: List[str]   # Jira issue keys, e.g. ["QA-42", "QA-43"]

    # ── Control-flow scalars (managed by routing.py) ─────────────────────────
    current_phase:               str   # human-readable label for the active node
    retry_count:                 int   # number of self-heal/re-execute cycles so far
    max_retries:                 int   # hard cap (default 3)
    augmentation_cycle:          int   # how many times augment_tests has run
    max_augmentation_cycles:     int   # hard cap (default 2)
    human_escalation_required:   bool
    escalation_reason:           Optional[str]

    # ── Error accumulation ───────────────────────────────────────────────────
    errors: Annotated[List[AgentError], operator.add]

    # ── Final outputs ────────────────────────────────────────────────────────
    pr_comment_body: Optional[str]    # Markdown body for the GitHub PR comment
    commit_status:   str              # pending | success | failure | error


# ──────────────────────────────────────────────────────────────────────────────
# Default initialiser — call this to get a clean starting state
# ──────────────────────────────────────────────────────────────────────────────

def initial_state(pr_metadata: PRMetadata) -> QAAgentState:
    """
    Returns a fully-initialised QAAgentState with safe defaults.
    Pass the result directly to graph.invoke().

    Example
    -------
    >>> from qa_agent.state import PRMetadata, initial_state
    >>> state = initial_state(PRMetadata(
    ...     pr_number=42,
    ...     repo="acme/api",
    ...     head_sha="abc123",
    ...     base_sha="def456",
    ...     author="alice",
    ...     title="Add /users endpoint",
    ...     description="Implements user CRUD",
    ... ))
    >>> compiled_graph.invoke(state)
    """
    return QAAgentState(
        pr_metadata=pr_metadata,

        # Phase 1
        change_manifest=None,
        test_scenarios=[],

        # Phase 2
        historical_tests=[],
        generated_tests=[],

        # Phase 3
        test_results=[],
        repair_attempts=[],
        execution_report=None,

        # Phase 4
        coverage_report=None,

        # Phase 5
        defects=[],
        jira_tickets_created=[],

        # Control flow
        current_phase="init",
        retry_count=0,
        max_retries=3,
        augmentation_cycle=0,
        max_augmentation_cycles=2,
        human_escalation_required=False,
        escalation_reason=None,

        # Errors
        errors=[],

        # Outputs
        pr_comment_body=None,
        commit_status=CommitStatus.PENDING.value,
    )
