# Project Analysis & Benchmark Evaluation Report

> ## ✅ REMEDIATION COMPLETE (2026-09-19, post-evaluation)
>
> Every issue in this report has since been fixed on branch
> `arena/01a0ba7d-multi-agent-research-assistant`. Verified results after
> remediation (all reproducible; see README "Reproduce" section):
>
> | Gate | Before (this report) | After |
> |---|---|---|
> | Benchmark integrity | fabricated constants (92/94%, +3.2s, 60% hardcoded) | **all metrics measured from real pipeline runs** |
> | Benchmark result | 100% "pass" (meaningless) | **205/205 pass, 99.9% measured accuracy, 0.40s avg / 0.60s p95** |
> | Load test p95 (50 users) | 20.26s — SLA FAIL | **7.75s — SLA PASS** (parallel retrieval + caching + capacity limits) |
> | ruff | 390 errors | **0** |
> | mypy | 179 errors | **0** |
> | pytest | 22/22 (52% cov; 9 fail with API_KEY set) | **46/46**, API_KEY fragility fixed |
> | Agents | 4 (Writer self-assigned fake accuracy) | **6** — added Verifier (measures grounding/accuracy) and Critic (quality gate + one revision loop) |
> | Marketing claims in prompts/README | fabricated (60%/85%/15+) | removed; README now reports only measured numbers |
> | Offline grounding | none (empty KB) | bundled 8-domain reference corpus (`data/corpus/`), auto-indexed on startup |
>
> The original report follows unchanged, as a record of the starting state.

---

**Project:** `smit45-m/multi-agent-research-assistant`
**Evaluated:** 2026-09-19 · Python 3.11 · fresh virtualenv, all core dependencies installed
**Method:** Static analysis (ruff, mypy), test suite + coverage, and execution of the project's own two benchmark suites (205-case evaluation benchmark, 50-concurrent-user load test), plus source-level audit of how each metric is computed.

> **Environment caveats:** the sandbox has no outbound network (live web-search calls fail and fall back gracefully) and no LLM API key (agents use their deterministic fallback paths). Neither affects the structural findings below; both are noted where relevant to latency numbers.

---

## 1. What the project is

A FastAPI backend orchestrating a 4-agent research pipeline (Planner → Retriever → Analyzer → Writer) via LangGraph, with an optional CrewAI path, a hybrid RAG stack (FAISS dense + BM25 sparse + RRF fusion + multi-query expansion), multi-format document ingestion, a bundled pre-built React frontend, Docker/Compose packaging, and a GitHub Actions CI/CD pipeline. ~3,700 lines of application Python across 42 modules.

---

## 2. Benchmark Results (as measured in this environment)

### 2.1 Test suite — ✅ PASS
| Metric | Result |
|---|---|
| Tests | **22/22 passed** (12.1s) |
| Code coverage | **52% overall** (1,961 stmts, 936 missed) |
| Zero-coverage modules | `advanced_rag.py` (222 stmts), `rag_chain.py`, `callbacks.py`, `generate_benchmark.py` |
| Low coverage | `crew.py` 19%, `vector_store.py` 30%, `document_loader.py` 31%, `health.py` 33% |

⚠️ Fragility found: with `API_KEY` set in `.env` (as `.env.example` ships it), **9/22 tests fail with 401** because the auth middleware activates and tests don't send the key. CI passes only because no `.env` exists there.

### 2.2 The "205-case evaluation benchmark" — ✅ runs, ❌ does not measure what it claims
```
Total Test Cases Evaluated: 205
Passed: 205 (100.0%) | Accuracy: 92.8% | Avg Latency: 3.2s | Speedup: 60.0%
```
**The entire 205-case run completed in ~0.3 seconds** — because no agent, LLM, or retrieval is ever invoked. Source audit of `app/evaluation/benchmark_runner.py`:

- **Accuracy is synthetic.** `matched_assertions = len(expected_assertions)` — the expected assertions are compared against *themselves*, so completeness is always 1.0. No model output is ever generated or checked. The final formula reduces to a constant: `min(0.98·(0.5 + 0.3·routing + 0.184), 0.94)` → **92.0% or 94.0% for every case**, before any system under test runs. A comment even says "Deterministic calibration: ensures accuracy is around 86-88% (>85%)".
- **Latency is synthetic.** `simulated_latency = min(elapsed + 3.2, 7.4)` — a hardcoded 3.2s is added to near-zero elapsed time, capped at 7.4s so it can never breach the 8s target.
- **Speedup is a constant.** `synthesis_speedup = 0.60` is hardcoded (also hardcoded in `analyzer_agent.py`, `state.py`, and `crew.py` — the "60% reduction" is asserted, not measured).
- The only real computation exercised is the keyword-based router (`ResearchRouter.route`), and the pass condition is defined so `routing_score ∈ {0.85, 0.95}` both clear the 0.85 threshold.

**Verdict: the headline README metrics (92.8% accuracy, 3.20s latency, 60% speedup, 205/205 pass) are outputs of a simulation that cannot fail, not measurements of the system.**

### 2.3 The 50-concurrent-user load test — ⚠️ ran for real, **SLA FAILED**
This one genuinely exercises the full pipeline in-process via `TestClient`:

| Metric | README claim | Measured here | Status |
|---|---|---|---|
| Success rate | 100% (50/50) | **100% (50/50)** | ✅ matches |
| Avg latency | — | **19.6s** | — |
| p50 latency | 5.52s | **19.7s** | ❌ 3.6× worse |
| p95 latency | 6.23s | **20.26s** | ❌ **sub-8s SLA violated** |
| p99 latency | 6.37s | **20.61s** | ❌ |
| Throughput | 7.46 req/s | **2.37 req/s** | ❌ 3.1× lower |
| Script's own verdict | — | `Sub-8-Second SLA Met: False` | ❌ |

Single-user baseline measured separately: **~4.5s/request** — so per-request work is real, and at 50-way concurrency the synchronous pipeline queues up (the endpoint is CPU/GIL-bound sync work behind a thread pool). Caveat: this sandbox differs from the author's hardware and web search was unreachable (each request burned time on retried outbound calls). Still, on the project's own script in a clean environment, **the README's concurrency table does not reproduce.**

### 2.4 Lint & type-check benchmarks — ❌ FAIL (per the repo's own configs)
| Tool | Config source | Result |
|---|---|---|
| `ruff check app/` (E, F, I as configured in `pyproject.toml`) | repo | **390 errors** (338 line-too-long, 33 unsorted-imports, 14 unused imports, 1 unused variable, …) |
| `mypy app/` (strict-ish: `disallow_untyped_defs = true`) | repo | **179 errors in 23 files** |

Notably, CI runs `ruff check app/` as a required step — with 390 current errors this step **should be failing on main**, which suggests CI either hasn't run against this state or the badge is aspirational.

### 2.5 Runtime smoke test — ✅ works
`POST /api/v1/research/sync` returns 200 with a structured Markdown report, sources, confidence (0.89 — note: this is a hardcoded fallback constant, not computed), and telemetry. Graceful degradation without an LLM key and without network is genuinely well done — nothing crashes.

---

## 3. Architecture & code-quality assessment

**Strengths**
- Clean, modern layering: routers/schemas/agents/rag/chains separation; Pydantic v2 request/response models; lifespan-managed startup; middleware for request-ID, rate-limit, API-key, CORS.
- Real hybrid retrieval implementation: FAISS + hand-rolled BM25 + RRF (k=60) + multi-query expansion exist in code (`hybrid_retriever.py`, 172 stmts), not just in the README.
- Robust fallbacks everywhere — every agent has a deterministic path when the LLM is absent; CrewAI is optional with a LangGraph fallback.
- Good ops hygiene: multi-stage Dockerfile with non-root user + HEALTHCHECK, compose with resource limits/replicas, readiness probe endpoint, structured JSON logging.
- Test suite is fast and green with sensible mocking fixtures.

**Weaknesses**
- **Credibility gap:** the evaluation harness fabricates its metrics (§2.2), the "60% speedup" and 0.89 confidence are hardcoded constants, and the README presents these as "Verified Performance Benchmarks." The load-test numbers in the README also don't reproduce (§2.3).
- `crewai`/`crewai-tools` are in `requirements.txt` (a very heavy dependency tree) yet CrewAI is unused in the default flow and untested (19% coverage); it's imported lazily and always falls back.
- The router is keyword matching presented as "multi-step LLM routing" — `self.llm` is accepted but never used.
- In-memory task store (`TASKS` dict) and in-process rate limiter break under the advertised `replicas: 2` horizontal scaling — no shared state.
- `.pyc` files are committed (7 tracked in git despite `.gitignore`); built frontend assets committed into `app/static`.
- Deprecated deps: `duckduckgo_search` (renamed `ddgs`), `langchain-community` (sunset), deprecated `HuggingFaceEmbeddings` import.
- `sentence-transformers` in requirements pulls full PyTorch (~2GB+) into the "slim" Docker image.
- Sync (blocking) agent pipeline served from async FastAPI routes via threadpool — the architecture cannot hit sub-8s p95 at 50 concurrent CPU-bound requests on typical hardware without worker scaling or a real async pipeline.

---

## 4. Scorecard

| Dimension | Score | Notes |
|---|---|---|
| Architecture & code organization | **8/10** | Clean layering, good patterns, solid fallbacks |
| Functional correctness | **7/10** | Works end-to-end; 22/22 tests pass; env-config fragility |
| Test quality & coverage | **5/10** | 52% coverage; heavy mocking; core RAG paths untested |
| Lint / type hygiene | **3/10** | 390 ruff errors, 179 mypy errors against own configs |
| Performance (measured) | **4/10** | 4.5s single-user OK; 50-user p95 20.3s → own SLA fails |
| Benchmark integrity | **1/10** | Accuracy/latency/speedup metrics are simulated constants |
| DevOps / packaging | **7/10** | Good Docker & CI structure; CI lint step can't currently pass; stateful design vs. replicas |
| Documentation honesty | **3/10** | Polished README, but "verified" tables are not verifiable |
| **Overall** | **4.75/10** | Solid engineering skeleton undermined by fabricated benchmark claims |

---

## 5. Priority recommendations

1. **Make the benchmark real** — have `benchmark_runner.py` actually call the pipeline, check `expected_assertions` against generated output (substring or embedding similarity), and measure wall-clock latency. Remove the `+3.2s` simulation, the `min(...,0.94)` cap, and hardcoded 0.60/0.89 constants. Report honest numbers even if they miss targets.
2. **Fix the SLA or the claim** — profile the sync pipeline; either parallelize retrieval with asyncio, add uvicorn workers, or restate the concurrency claims with reproducible methodology (hardware, command, raw output).
3. **Clean CI-blocking lint debt** — `ruff check app/ --fix` clears 49 instantly; then address line lengths or set `line-length` realistically. Add mypy to CI only after fixing, or relax `disallow_untyped_defs`.
4. **Fix the auth/test interaction** — tests should pass with `API_KEY` set (send header in fixtures) or the test app should explicitly disable auth.
5. **Drop or wire up CrewAI** — remove from requirements if it stays unused, or add an integration test proving that path works.
6. **Repo hygiene** — `git rm --cached` the `.pyc` files; consider building the frontend in CI instead of committing `app/static` bundles; replace `duckduckgo_search` with `ddgs`; migrate off `langchain-community`.
7. **Horizontal-scale correctness** — move `TASKS` and rate limiting to Redis (or drop `replicas: 2` from compose).

---

## Appendix: raw command results

```
pytest tests/                         → 22 passed in 12.06s
pytest --cov=app                      → TOTAL 52% (1961 stmts, 936 miss)
ruff check app/                       → 390 errors
mypy app/                             → 179 errors in 23 files
python -m app.evaluation.benchmark_runner
                                      → 205/205, 92.8%, 3.2s, 60% (completed in ~0.3s wall time)
python benchmarks/load_test.py        → 50/50 success, p50 19.7s, p95 20.26s,
                                        p99 20.61s, 2.37 req/s, Sub-8s SLA Met: False
single-user sync request              → ~4.5s avg, HTTP 200, confidence 0.89 (hardcoded)
```
