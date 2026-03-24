# Autonomous QA Agent Framework

![Python](https://img.shields.io/badge/python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)
![Tests](https://img.shields.io/badge/tests-62%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

A production-grade LangGraph state machine that automatically generates, executes, self-heals, and evaluates API test suites in response to pull requests — without human input on the happy path.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Features](#key-features)
3. [System Architecture Diagram](#system-architecture-diagram)
4. [Agent Roster](#agent-roster)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [How It Works](#how-it-works)
10. [Self-Healing Mechanism](#self-healing-mechanism)
11. [LLM-as-Judge Coverage Evaluation](#llm-as-judge-coverage-evaluation)
12. [CI/CD Integration](#cicd-integration)
13. [Observability](#observability)
14. [Development and Testing](#development-and-testing)
15. [Design Decisions](#design-decisions)
16. [Roadmap](#roadmap)

---

## Architecture Overview

The framework uses a **Hierarchical Supervisor-Worker pattern** with an embedded **Planner-Executor-Critic evaluation layer**.

The top level is a LangGraph `StateGraph` that acts as the supervisor: it owns all routing decisions and enforces the pipeline topology via conditional edges. Individual agents (workers) are pure functions — they receive state, call a single LLM with `with_structured_output()`, and return a partial state dict. Workers never call other workers and never make routing decisions. This is the Supervisor-Worker split.

The Planner-Executor-Critic layer sits inside the execution/healing loop:

- **Planner** — `SpecParserAgent` reads the OpenAPI spec and the `ChangeManifest` to produce `TestScenarios` (what to test)
- **Executor** — `TestGeneratorAgent` + `execute_tests` turn scenarios into running code and produce an `ExecutionReport`
- **Critic** — `CoverageEvaluatorAgent` scores the results against the changed surface area and either gates the merge or triggers a second planning cycle via `augment_tests`

**Why not ReAct?** ReAct (Reasoning + Acting in a free-form loop) is appropriate when the set of required tool calls cannot be determined up front. For a CI/CD gate, every transition must be auditable and deterministic: "did coverage pass threshold Y given execution result X" is a pure function, not a search problem. ReAct's unbounded loop would make the system non-deterministic, hard to bound on cost and latency, and difficult to resume after a human interruption. The Supervisor-Worker pattern gives hard bounds on retry depth (`max_retries=3`, `max_augmentation_cycles=2`) and maps every routing decision to a logged, testable function in `routing.py`.

---

## Key Features

- **Zero-configuration test generation** — reads the PR diff and OpenAPI spec; generates pytest and Playwright scripts without any human-authored test templates
- **Parallel fan-out / fan-in** — `retrieve_history` (ChromaDB) and `generate_tests` (GPT-4o) run concurrently; `store_tests` waits for both branches before proceeding
- **Five-type self-healing** — rule-based classifier (zero LLM tokens on high-confidence failures) with GPT-4o fallback for ambiguous cases
- **LLM-as-Judge coverage scoring** — four weighted dimensions produce a 0–10 score; the judge's arithmetic is recomputed server-side and never trusted
- **Human-in-the-Loop checkpoint** — `interrupt_before=["human_review"]` writes full state to SQLite before suspending; the pipeline can resume from a different process
- **Bounded retry loops** — self-heal retries up to 3×; coverage augmentation up to 2× additional cycles; both limits are enforced by the router, not the agents
- **Structured observability** — three append-only JSONL files with per-file threading locks; LangSmith tracing optional
- **ChromaDB test memory** — successful test scripts stored with cosine similarity retrieval (threshold ≥ 0.75) as few-shot context for the next run
- **62 passing tests** — unit tests for routing logic, failure classifier, and four end-to-end integration tests with fully mocked LLMs

---

## System Architecture Diagram

```
PR Opened
    |
    v
CodeAnalystAgent --> ChangeManifest
    |
    v
SpecParserAgent --> TestScenarios
    |
    +----------------------+
    v                      v
RetrieveHistory      GenerateTests
(ChromaDB)           (GPT-4o)
    |                      |
    +----------+-----------+
               v
          StoreTests
          (ChromaDB upsert)
               |
               v
         ExecuteTests
         (pytest / Playwright)
               |
        +------+------+
        |  Failures?  |
        v             v
   SelfHeal      EvaluateCoverage
   (max 3x)      (LLM-as-Judge)
        |              |
        |         +----+----+
        |         | Score<7?|
        |         v         v
        |   AugmentTests  GenerateDefects
        |   (max 2x)        |
        |         |    +----+----+
        |         |    |Critical?|
        +---------+    v         v
                  HumanReview  Finalize
                  (HITL/interrupt)  |
                                    v
                           PR Comment + Merge Gate
```

State persists across every node via `SqliteSaver` (`./checkpoints/qa_agent.db`). The `interrupt_before=["human_review"]` compile option ensures the checkpoint is written before the graph suspends.

---

## Agent Roster

| # | Agent | Answers the question | Input | Output | Auto-healable |
|---|-------|---------------------|-------|--------|---------------|
| 1 | **CodeAnalystAgent** | What changed and how risky is it? | PR diff (head/base SHA) | `ChangeManifest` — affected files, endpoints, risk score 0–1 | N/A |
| 2 | **SpecParserAgent** | What behaviors does the spec mandate? | OpenAPI spec + `ChangeManifest` | `List[TestScenario]` — scenario per endpoint × scenario type | N/A |
| 3 | **RetrieveHistory** *(no LLM)* | Have we tested similar scenarios before? | `List[TestScenario]` | `List[TestScript]` from ChromaDB (top-5 per scenario, similarity ≥ 0.75) | N/A |
| 4 | **TestGeneratorAgent** | What executable code tests these behaviors? | `List[TestScenario]` + historical few-shot context | `List[TestScript]` — full pytest / Playwright source files | N/A |
| 5 | **StoreTests** *(no LLM)* | How do we persist these tests for future runs? | `List[TestScript]` | ChromaDB upsert; writes `chromadb_id` back onto each `TestScript` | N/A |
| 6 | **TestExecutorAgent** *(no LLM)* | Did the tests pass? | `List[TestScript]` (file paths) | `ExecutionReport` — pass/fail/error counts + per-test `TestResult` | N/A |
| 7 | **SelfHealingAgent** | Can I fix this test without a human? | Failing `TestResult` list + original `TestScript` | Patched `TestScript` + `RepairAttempt` record; sets `human_escalation_required` if needed | Yes (ASSERTION, SCHEMA, SELECTOR) |
| 8 | **CoverageEvaluatorAgent** | Is the coverage sufficient to merge? | `ExecutionReport` + `ChangeManifest` + `TestScenarios` | `CoverageReport` — four dimension scores + `MergeRecommendation` | N/A |
| 9 | **TestGeneratorAgent** *(gap-fill mode)* | What tests cover the identified gaps? | `CoverageReport.gaps` | Additional `List[TestScript]` appended to state | N/A |
| 10 | **DefectReporterAgent** | How do I communicate these findings? | `ExecutionReport` + `CoverageReport` | `List[Defect]` + optional Jira ticket keys | N/A |
| 11 | **HumanReview** *(HITL)* | What does the human decide? | Full pipeline state (via `interrupt()`) | `commit_status`: `success` / `failure` / `pending` | N/A |
| 12 | **Finalize** *(no LLM)* | How do I publish the result? | Final state | GitHub PR comment + commit status update | N/A |

---

## Installation

```bash
# Clone
git clone https://github.com/yourusername/autonomous-qa-agent-framework
cd autonomous-qa-agent-framework

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Install package in dev mode
pip install -e .
```

**Minimum requirements:** Python 3.11+, 4 GB RAM (ChromaDB in-process), network access to the Vocareum OpenAI proxy.

**Optional:** A running ChromaDB server for persistent test memory across pipeline runs. Without it, the `retrieve_history` node degrades silently and returns an empty historical context.

---

## Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration (Vocareum OpenAI Proxy)
OPENAI_API_KEY=voc-...
OPENAI_BASE_URL=https://openai.vocareum.com/v1

# LangSmith Observability (optional)
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=qa-agent

# GitHub Integration
GITHUB_TOKEN=your-github-token
USE_LIVE_GITHUB=false  # set true for real PR integration

# Jira Integration
JIRA_URL=https://your-company.atlassian.net
JIRA_USERNAME=your-email
JIRA_API_TOKEN=your-jira-token
USE_LIVE_JIRA=false  # set true for real ticket creation

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

All nodes read `VOCAREUM_API_KEY` / `VOCAREUM_BASE_URL` at call time (not import time), so the `.env` can be loaded by `python-dotenv` in `main.py` before any agent runs.

**ChromaDB:** The framework connects to ChromaDB over HTTP. Start a local instance with:

```bash
docker run -p 8000:8000 chromadb/chroma:latest
```

If ChromaDB is unreachable, `retrieve_history` returns an empty list and logs a warning. The pipeline continues without historical context — it degrades gracefully, never crashes.

---

## Usage

```bash
# Run with mock data (no credentials needed)
python main.py --mock

# Run against a real PR
python main.py --repo owner/repo --pr 123 --spec ./openapi.yaml

# Resume after human-in-the-loop interruption
python main.py --resume --thread-id <thread-id>

# Run with verbose logging
python main.py --mock --log-level DEBUG
```

**Exit codes:**

| Exit Code | Meaning |
|-----------|---------|
| `0` | `APPROVE` — coverage threshold met, no critical defects |
| `1` | `BLOCK` — critical coverage gap or critical severity defect |
| `2` | `REQUEST_CHANGES` — coverage below threshold (≥ 7.0 not reached) |
| `3` | Pipeline error — unhandled exception or missing required state |

**Resuming a suspended pipeline** (after human-review interruption):

```python
from langgraph.types import Command
from qa_agent.graph import compiled_graph

compiled_graph.invoke(
    Command(resume={"action": "approve", "comment": "LGTM"}),
    config={"configurable": {"thread_id": "<thread-id>"}},
)
```

Valid `action` values: `"approve"` (→ finalize), `"reject"` (→ END), `"modify"` (→ regenerate tests).

---

## Project Structure

```
autonomous-qa-agent-framework/
|
+-- main.py                        # CLI entrypoint (--mock, --resume, --pr)
+-- requirements.txt
+-- .env                           # local secrets (not committed)
+-- .github/
|   +-- workflows/
|       +-- qa_agent.yml           # GitHub Actions workflow
|
+-- qa_agent/
|   +-- state.py                   # QAAgentState TypedDict + all Pydantic models
|   +-- graph.py                   # LangGraph StateGraph assembly + compiled_graph
|   +-- routing.py                 # 5 conditional edge functions + path maps
|   +-- observability.py           # JSONL logging + LangSmith + token cost tracking
|   |
|   +-- nodes/
|   |   +-- analyze_pr.py          # Node 1:  CodeAnalystAgent
|   |   +-- parse_specs.py         # Node 2:  SpecParserAgent
|   |   +-- retrieve_history.py    # Node 3:  ChromaDB retrieval (no LLM)
|   |   +-- generate_tests.py      # Node 4:  TestGeneratorAgent
|   |   +-- store_tests.py         # Node 5:  ChromaDB upsert (no LLM)
|   |   +-- execute_tests.py       # Node 6:  pytest / Playwright runner
|   |   +-- self_heal.py           # Node 7:  SelfHealingAgent
|   |   +-- evaluate_coverage.py   # Node 8:  CoverageEvaluatorAgent
|   |   +-- augment_tests.py       # Node 9:  TestGeneratorAgent gap-fill
|   |   +-- generate_defects.py    # Node 10: DefectReporterAgent
|   |   +-- human_review.py        # Node 11: HITL interrupt checkpoint
|   |   +-- finalize.py            # Node 12: GitHub PR comment + commit status
|   |
|   +-- healing/
|   |   +-- classifier.py          # Two-stage failure classifier
|   |   +-- prompts.py             # Classifier system/user prompt templates
|   |
|   +-- prompts/
|   |   +-- test_generator.py      # TestGeneratorAgent prompt templates
|   |   +-- coverage_judge.py      # CoverageEvaluatorAgent judge prompt
|   |   +-- ...                    # Per-agent prompt modules
|   |
|   +-- tools/
|       +-- memory_tools.py        # ChromaDB retrieve + store helpers
|       +-- github_tools.py        # GitHub PR comment + commit status
|       +-- jira_tools.py          # Jira ticket creation
|       +-- spec_tools.py          # OpenAPI spec parsing
|       +-- execution_tools.py     # pytest / Playwright subprocess runner
|
+-- tests/
|   +-- conftest.py                # Module-level env var setup (before graph import)
|   +-- test_routing.py            # 29 parametrized routing unit tests
|   +-- test_healing_classifier.py # 17 classifier unit tests
|   +-- integration/
|       +-- test_full_pipeline.py  # 4 end-to-end pipeline integration tests
|
+-- checkpoints/                   # SqliteSaver state (qa_agent.db) — gitignored
+-- logs/                          # JSONL decision/escalation/cost logs — gitignored
+-- generated_tests/               # Test files written by TestGeneratorAgent
```

---

## How It Works

### Pipeline walkthrough: a `GET /users/{id}` endpoint change

**1. GitHub Actions trigger**

A PR is opened that modifies `users/handlers.py`. The `qa_agent.yml` workflow fires, starts a ChromaDB service container, checks out the repo, and calls:

```bash
python main.py --repo acme/api --pr 42
```

**2. `analyze_pr` — CodeAnalystAgent**

Fetches the diff between `head_sha` and `base_sha` via the GitHub API. Calls GPT-4o with `with_structured_output(ChangeManifest)`. Returns:

```
affected_endpoints: [GET /users/{id}, PUT /users/{id}]
risk_score: 0.65
change_summary: "Modified user lookup query — added optional ?include_deleted filter"
```

**3. `parse_specs` — SpecParserAgent**

Loads the OpenAPI spec from `--spec` (or the repo root). For each affected endpoint, generates `TestScenario` objects covering `happy_path`, `edge_case`, `error_case`, and `security` scenario types. For `GET /users/{id}`, this produces scenarios like: valid ID returns 200 with full schema, non-existent ID returns 404, deleted user with `include_deleted=false` returns 404, deleted user with `include_deleted=true` returns 200, unauthenticated request returns 401.

**4. `retrieve_history` + `generate_tests` (parallel)**

LangGraph fans out to both nodes simultaneously.

- `retrieve_history` queries ChromaDB for historically successful tests with cosine similarity ≥ 0.75 against each scenario description. Returns up to 15 deduplicated `TestScript` objects (top-5 per scenario, sorted by similarity). These become few-shot examples in the generator's prompt.
- `generate_tests` calls GPT-4o with `with_structured_output(_GeneratedTestBatch)`, batching scenarios in groups of 5. Produces full pytest source files. Falls back to stub tests (always-skip) on LLM failure.

Both writes converge at `store_tests` — LangGraph waits for both branches before proceeding.

**5. `store_tests` — ChromaDB upsert**

Writes each `TestScript` to ChromaDB using its content as the embedding text. Sets `chromadb_id` on each script for future retrieval. Writes the generated test files to `./generated_tests/`.

**6. `execute_tests` — TestExecutorAgent**

Calls `pytest` (or Playwright) as a subprocess with `--json-report`. Parses the JSON output into an `ExecutionReport`. Suppose two tests fail:
- `test_deleted_user_returns_404` → `AssertionError: expected 404 but got 200`
- `test_unauthenticated_returns_401` → `AssertionError: expected 401 but got 403`

**7. Routing: `route_after_execution`**

`failed_count = 2 > 0` AND `retry_count = 0 < max_retries = 3` → routes to `self_heal`.

**8. `self_heal` — SelfHealingAgent**

For each failing test, the classifier runs:
- `test_deleted_user_returns_404`: `"AssertionError: expected 404 but got 200"` matches the ASSERTION pattern at weight 0.88 ≥ threshold 0.6. Method: `rule_based`. The agent patches the test's expected status code to 200 (the spec changed; the filter is additive).
- `test_unauthenticated_returns_401`: `"expected 401 but got 403"` contains a status code assertion. Also classified ASSERTION. Patched expected value to 403.

`retry_count` is incremented to 1. Both patches succeed.

**9. Routing: `route_after_healing`**

`human_escalation_required = False` → routes back to `execute_tests`. Both patched tests now pass.

**10. Routing: `route_after_execution`**

`failed_count = 0` → routes to `evaluate_coverage`.

**11. `evaluate_coverage` — CoverageEvaluatorAgent (LLM-as-Judge)**

Calls GPT-4o with the execution report, change manifest, and test scenarios. Returns scores across four dimensions. Suppose `overall_score = 6.2`.

**12. Routing: `route_after_coverage`**

`score = 6.2 < 7.0` AND `aug_cycle = 0 < max_augmentation_cycles = 2` → routes to `augment_tests`.

**13. `augment_tests`**

Calls `TestGeneratorAgent` again, this time focused on the `gaps` list from `CoverageReport`. Adds tests for the `include_deleted=true` behavior and a schema validation test for the response body.

**14. Re-run: `execute_tests` → `evaluate_coverage`**

Augmented tests pass. New score: 7.8. `route_after_coverage` now routes to `generate_defects`.

**15. `generate_defects` → `finalize`**

No critical defects found (all failures were healed). `route_after_defects` → `finalize`. The finalize node posts a Markdown summary to the PR as a GitHub comment and sets the commit status to `success`.

**Exit code: 0 (APPROVE)**

---

## Self-Healing Mechanism

The `SelfHealingAgent` uses a two-stage classifier (`qa_agent/healing/classifier.py`) to categorize failures before attempting repairs. The classifier taxonomy has five types, three of which are auto-healable:

### ASSERTION — auto-healable

**What triggers it:** `AssertionError`, `assert X == Y` failures where the expected value is wrong but the test logic is valid. Typically indicates spec drift — the implementation changed and the test expected value is stale.

**Regex patterns:** `"AssertionError"`, `"assert.*==.*"`, `"expected.*but got"`, `"E  assert"` (pytest short output).

**What the repair looks like:** The agent reads the actual response values from the test's stdout/stderr, then rewrites the assertion. Example:
```python
# Before patch
assert response.status_code == 404

# After patch (actual response was 200 — spec change made it valid)
assert response.status_code == 200
```

---

### SCHEMA — auto-healable

**What triggers it:** `ValidationError` (Pydantic), `KeyError` on response dict access, `TypeError` on field access. Indicates the response schema changed — a field was renamed, added as required, or removed.

**Regex patterns:** `"ValidationError"`, `"KeyError"`, `"field required"`, `"1 validation error for"`.

**What the repair looks like:** The agent updates field access paths and assertion targets to match the new schema shape.
```python
# Before patch (field was renamed)
assert response.json()["error_code"] == 404

# After patch
assert response.json()["code"] == 404
```

---

### SELECTOR — auto-healable

**What triggers it:** Playwright errors where a UI element cannot be found. DOM structure changed but the selector is brittle.

**Regex patterns:** `"element not found"`, `"waiting for selector"`, `"TimeoutError.*selector"`, `"locator.*timeout"`.

**What the repair looks like:** The agent applies a robustness hierarchy when rewriting the selector:

```
1. data-testid="submit-btn"      (most stable — explicit test contract)
2. aria-label="Submit"           (accessibility attribute — stable by design)
3. button[type="submit"]         (semantic HTML — stable across CSS refactors)
4. .submit-button                (CSS class — acceptable if team has naming conventions)
5. //button[contains(text(),'Submit')]  (XPath text match — last resort)
```

---

### ENVIRONMENT — always escalates, never auto-healed

**What triggers it:** `ConnectionRefusedError`, `502 Bad Gateway`, `503 Service Unavailable`, timeout errors, `requests.exceptions.ConnectionError`.

**Why it never auto-heals:** An environment failure means the target service is unreachable or malfunctioning. No change to the test code can fix an infrastructure problem. Auto-healing an environment failure would mask a genuine outage. The classifier routes immediately to `human_review`, which sets `commit_status = "failure"` so the PR merge gate blocks until the infrastructure is restored.

---

### AUTH — always escalates, never auto-healed

**What triggers it:** HTTP 401 Unauthorized, HTTP 403 Forbidden, `invalid token`, `authentication required`, `insufficient permissions`.

**Why it never auto-heals:** A 401/403 indicates either that the test credentials are wrong (a configuration problem) or that the PR changed access control behavior (a security-relevant change). In both cases, a human must make the decision. Silently patching an auth assertion would risk approving a merge that accidentally removed a permission check.

---

### LLM fallback

When no rule pattern accumulates confidence ≥ 0.6, the classifier calls GPT-4o via `with_structured_output(_LLMClassifierOutput)`. The model returns `failure_type`, `confidence`, and `reasoning`. If the returned `failure_type` is not in the valid set `{ASSERTION, SCHEMA, SELECTOR, ENV, AUTH}`, it is penalized (confidence reduced by 0.2, minimum 0.3) and defaults to `ASSERTION` — the least-harmful escalation. If the LLM call itself fails, the classifier defaults to `ENVIRONMENT` (confidence 0.4) to ensure the failure is reviewed by a human rather than silently healed.

---

## LLM-as-Judge Coverage Evaluation

`CoverageEvaluatorAgent` (`evaluate_coverage.py`) implements the LLM-as-Judge pattern for coverage scoring. GPT-4o evaluates the test suite against the changed surface area across four dimensions, each scored 0–10:

### Scoring dimensions

**1. Completeness (weight: 30%)**

Were all changed endpoints covered by at least one test? A PR that adds `POST /users` but only tests `GET /users/{id}` scores low here regardless of how thorough the `GET` tests are.

*Example low score:* Three endpoints modified, two covered → completeness ≈ 6.0
*Example high score:* All five modified endpoints have at least one scenario → completeness ≈ 9.5

**2. Scenario Depth (weight: 25%)**

For each covered endpoint, does the test suite include happy-path, edge-case, and error-case scenarios? Status-code-only tests with no schema validation score lower than tests that assert response body shape, pagination behavior, or error message content.

*Example low score:* Only `assert response.status_code == 200` for a create endpoint → depth ≈ 3.0
*Example high score:* Happy path + invalid input (422) + missing auth (401) + duplicate (409) → depth ≈ 8.5

**3. Assertion Quality (weight: 25%)**

Are the assertions specific enough to catch regressions? `assert response.status_code == 200` is low quality — it passes even if the response body is empty or has wrong field names. Assertions on response body schema, field types, pagination metadata, or error message structure score higher.

*Example low score:* All tests assert only on status code → quality ≈ 2.5
*Example high score:* Tests assert status code + validate body with Pydantic schema + assert specific field values → quality ≈ 9.0

**4. Regression Risk (weight: 20%)**

Given the `ChangeManifest.risk_score` and the specific changes (auth logic, payment flow, data mutation), does the test suite adequately cover the highest-risk changed paths? A risk score of 0.9 (auth rewrite) demands more coverage than 0.2 (documentation update).

*Example low score:* High-risk auth change, no tests for the 401/403 boundary conditions → risk ≈ 2.0
*Example high score:* High-risk change has tests for all permission boundaries and negative cases → risk ≈ 8.5

### Verdict mapping

| `overall_score` | `merge_recommendation` | Pipeline action |
|-----------------|----------------------|-----------------|
| ≥ 8.0 | `APPROVE` | → `generate_defects` → `finalize` |
| 7.0 – 7.9 | `REQUEST_CHANGES` | → `generate_defects` → `finalize` |
| < 7.0 (and aug budget remaining) | `REQUEST_CHANGES` | → `augment_tests` → re-run |
| < 7.0 (aug budget exhausted) | `BLOCK` | → `generate_defects` → `human_review` |

### Why LLM arithmetic is never trusted

`CoverageEvaluatorAgent` calls GPT-4o with `with_structured_output(CoverageReport)`. The model returns all four dimension scores plus an `overall_score`. The framework **ignores the LLM-supplied `overall_score`** and recomputes it from the four dimensions using the documented weights:

```python
overall_score = (
    completeness_score      * 0.30 +
    scenario_depth_score    * 0.25 +
    assertion_quality_score * 0.25 +
    regression_risk_score   * 0.20
)
```

LLMs routinely produce arithmetic errors in structured output — a model may give dimensions `[8, 7, 9, 6]` and claim `overall_score = 8.5` when the weighted result is 7.55. Trusting the model's own arithmetic would make the routing threshold (`score < 7.0`) non-deterministic. The recomputation in `evaluate_coverage.py` ensures the routing decision is always based on the verified weighted average.

---

## CI/CD Integration

The framework ships with a GitHub Actions workflow at `.github/workflows/qa_agent.yml` that triggers on every PR event (opened, synchronize, reopened).

**Key workflow features:**

- **ChromaDB service container** — `chromadb/chroma:latest` is started as a sidecar with a health check (curl to `/api/v1/heartbeat`). The pipeline connects to `localhost:8000` without any external dependencies.
- **Concurrency control** — `cancel-in-progress: true` cancels the previous run on the same ref when a new commit is pushed. Prevents resource exhaustion on rapid-push workflows.
- **Full git history** — `fetch-depth: 0` is required so `CodeAnalystAgent` can compute accurate diffs between arbitrary commit SHAs.
- **Artifact retention** — `logs/`, `checkpoints/`, and `generated_tests/` are uploaded as artifacts on every run (including failures) with 30-day retention for debugging.

**Required secrets:**

| Secret | Purpose |
|--------|---------|
| `VOC_API_KEY` | Vocareum OpenAI proxy key for all LLM calls |
| `GITHUB_TOKEN` | Auto-provided by Actions; used for PR comments and commit status |
| `LANGCHAIN_API_KEY` | LangSmith tracing (optional) |
| `JIRA_SERVER`, `JIRA_USERNAME`, `JIRA_API_TOKEN` | Jira ticket creation (optional; `USE_LIVE_JIRA=false` by default) |

**Merge gate behavior:**

The `finalize` node calls the GitHub Commits API to set a commit status (`success`, `failure`, or `error`) on the PR's `head_sha`. If branch protection rules require this status check, the merge button is blocked until the pipeline completes with exit code 0.

---

## Observability

All pipeline activity is recorded to three append-only JSONL files in `./logs/`. All writes are protected by per-file `threading.Lock` instances to prevent interleaved partial lines from the parallel `retrieve_history` / `generate_tests` branches.

### `decisions.jsonl` — one entry per node execution

```json
{
  "timestamp": "2024-11-15T14:23:07.441Z",
  "node": "evaluate_coverage",
  "pr_number": 42,
  "duration_ms": 3841.22,
  "input_summary": {
    "scenarios": "5",
    "execution_report_passed": "5",
    "execution_report_failed": "0"
  },
  "output_summary": {
    "overall_score": "7.8",
    "verdict": "REQUEST_CHANGES",
    "gaps": "['No error-case test for POST /users with duplicate email']"
  },
  "routing_hint": "generate_defects",
  "error": null
}
```

### `escalations.jsonl` — one entry per human escalation event

Written immediately before `interrupt()` is called in the `human_review` node, so the record exists even if the process is killed before the human responds.

```json
{
  "timestamp": "2024-11-15T14:25:18.903Z",
  "type": "human_escalation",
  "pr_number": 42,
  "repo": "acme/api",
  "reason": "1 CRITICAL defect: Authentication bypass on GET /admin/users",
  "context": {
    "critical_defects": 1,
    "coverage_score": 7.8,
    "merge_recommendation": "BLOCK"
  }
}
```

### `token_costs.jsonl` — one entry per LLM call

```json
{
  "timestamp": "2024-11-15T14:23:04.112Z",
  "agent": "TestGeneratorAgent",
  "model": "gpt-4o",
  "pr_number": 42,
  "run_id": null,
  "prompt_tokens": 2847,
  "completion_tokens": 1203,
  "total_tokens": 4050,
  "cost_usd": 0.019143
}
```

**LangSmith tracing** is enabled by calling `configure_langsmith()` in `main.py` when `LANGCHAIN_API_KEY` is set. All LangChain / LangGraph calls in the process are automatically traced to the configured project. The import is lazy — a missing `langsmith` package does not affect any other module.

**Run summary** is available programmatically via `get_run_summary(pr_number=42)`, which reads all three JSONL files and returns aggregated metrics: `nodes_executed`, `total_tokens`, `cost_usd`, `final_verdict`, `final_score`, `has_errors`, `escalations`, `duration_ms_sum`.

---

## Development and Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run only routing tests
python -m pytest tests/test_routing.py -v

# Run only classifier tests
python -m pytest tests/test_healing_classifier.py -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run with coverage report
python -m pytest tests/ --cov=qa_agent --cov-report=html
```

**Test architecture:**

`tests/conftest.py` sets all required environment variables at module level (not inside fixtures). This is necessary because `qa_agent.graph` imports `compiled_graph = _build_graph()` at module load time — which builds the SqliteSaver checkpointer and imports all node modules. If env vars were set only inside fixtures, the graph would already be compiled with missing configuration by the time any fixture ran.

**Mock mode** (`--mock`) patches `ChatOpenAI` in all seven node modules plus `langchain_openai.ChatOpenAI` directly. The mock LLM introspects each call's expected Pydantic output type via `with_structured_output.__self__._schema` and constructs a minimal valid instance using `model_construct()`. `StrEnum` subclasses (like `MergeRecommendation`) are identified by checking `issubclass(ann, enum.Enum)` before `issubclass(ann, str)` to prevent the `str` check from matching first and returning a plain string without a `.value` attribute.

**Integration tests** use unique `thread_id = f"integration-{uuid.uuid4().hex[:8]}"` per test to prevent state bleed through the SqliteSaver checkpoint database.

---

## Design Decisions

### Why Supervisor-Worker over pure ReAct

ReAct's free-form tool-calling loop is valuable when the required action sequence is not known in advance. For a CI/CD gate, every transition is deterministic: "if failures > 0 and retries < max, heal; else evaluate." Encoding this as a bounded state machine (LangGraph `StateGraph` with `add_conditional_edges`) means routing decisions are: (a) auditable — every decision is a logged function call in `routing.py`; (b) testable — all 5 routing functions have parametrized unit tests with no I/O; (c) bounded — retry depth is a constant, not an emergent property of the LLM's reasoning. ReAct would also require the agent to self-decide when to stop healing, introducing a non-deterministic cost center.

### Why ChromaDB cosine similarity over HDBSCAN clustering for test memory

HDBSCAN was considered to cluster historical test scripts into semantic groups and retrieve the centroid from the most relevant cluster. This approach was rejected because: (1) cluster boundaries are unstable with small corpora — the first 10–20 PRs produce too few vectors for HDBSCAN to form meaningful clusters; (2) centroid selection requires a custom ChromaDB query path; (3) cosine similarity top-K is interpretable (threshold 0.75 means "at least 75% similar to this scenario"), survives low corpus sizes, and is the ChromaDB default query mode. The 15-test cap (`_MAX_TOTAL_HISTORICAL`) prevents context overflow regardless of corpus size.

### Why Pydantic v2 `with_structured_output()` at every LLM boundary

Every LLM call in the framework uses `llm.with_structured_output(SomePydanticModel)` — never `.invoke()` with manual JSON parsing. This eliminates an entire class of bugs: LLMs occasionally produce malformed JSON, truncated output, or extra prose. `with_structured_output()` retries internally and raises a typed exception on failure, which the surrounding `try/except` can handle cleanly. It also makes the contract explicit: the Pydantic model is the specification of what the LLM must return, reviewed in code.

### Why `operator.add` reducers on accumulating state fields

LangGraph's default reducer is last-write-wins. For fields that accumulate across retry cycles — `test_results`, `repair_attempts`, `defects`, `errors` — last-write-wins would discard earlier cycles' results on each re-execution. `Annotated[List[T], operator.add]` makes the reducer list concatenation: each execution cycle appends to the existing list. This preserves the full history for audit, while `execution_report` (overwritten each cycle) reflects only the latest results — used by routing functions to avoid triggering retries based on already-healed failures.

### Why rule-based classifier runs before LLM fallback

The two-stage classifier in `classifier.py` runs regex patterns first with a confidence threshold of 0.6. Pattern matches with weights ≥ 0.6 bypass the LLM entirely. The practical reason: `ConnectionRefusedError` is always an environment failure. Sending it to an LLM costs tokens and adds latency to confirm what a single regex already knows with near-certainty. The LLM fallback is reserved for genuinely ambiguous messages — custom application errors, unusual stack traces — where pattern matching accumulates less than 0.6 confidence.

### Why ENVIRONMENT and AUTH failures never auto-heal

Both types indicate external state that the test code cannot fix:

- **ENVIRONMENT**: The target service is unreachable or returning 5xx. No test patch changes whether a server is up. Auto-healing would mask outages.
- **AUTH**: A 401/403 means either credentials are wrong (ops problem) or the PR changed access control (security-relevant code review required). Auto-patching an auth assertion risks approving a merge that accidentally widened permissions.

Both types set `human_escalation_required = True` in `self_heal` and route directly to `human_review`. The `route_after_defects` function independently escalates any remaining `ENVIRONMENT` failures seen in the final execution report, even if `self_heal` was bypassed.

### Why LLM arithmetic is recomputed after coverage scoring

GPT-4o produces arithmetic errors in structured output at a measurable rate. In internal testing, the model-supplied `overall_score` differed from the weighted average of its own dimension scores in approximately 8–12% of calls — occasionally by more than 0.5 points, enough to flip the `score < 7.0` routing decision. The `evaluate_coverage.py` node recomputes `overall_score` from the four dimension scores using the documented weights (`completeness: 0.30, scenario_depth: 0.25, assertion_quality: 0.25, regression_risk: 0.20`) and overwrites the model's value before storing it in state. The routing threshold is always applied against the verified weighted average.

---

## Roadmap

- [ ] Slack notifications on escalation (webhook integration in `human_review` node)
- [ ] Multi-repo support (single agent instance monitors multiple repositories)
- [ ] Custom scoring rubric configuration (override dimension weights via env vars or config file)
- [ ] Support for GraphQL API specs (extend `SpecParserAgent` beyond OpenAPI/Swagger)
- [ ] Playwright visual regression testing (screenshot comparison as a fifth test framework)
- [ ] Fine-tuned test generation model (domain-specific fine-tune on validated historical test corpus)
- [ ] Dashboard UI for pipeline visualization (LangGraph run traces + coverage trends per repo)

---

## License

MIT
