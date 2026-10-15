# RAG Evaluation Project

This project evaluates a retrieval-augmented generation (RAG) application across
answer quality, safety, latency, estimated cost, and request reliability. The
evaluations use fixed golden datasets for repeatable quality and safety checks,
and a small repeated benchmark workload for operational measurements.

## Evaluation flow

The suite orchestrator in [`evals/run_suite.py`](./evals/run_suite.py) is designed
to:

1. Build one `RagPipeline`.
2. Reuse its retriever for the retriever evaluation and inject the same pipeline
   into the remaining evaluations.
3. Run component, full-pipeline, application, safety, and operational checks.
4. Flatten the results into metric IDs and write a JSON snapshot for comparison
   against a baseline.

The intended command is:

```bash
python -m evals.run_suite
```

Useful options:

```bash
python -m evals.run_suite --label "describe the change"
python -m evals.run_suite --full
python -m evals.run_suite --out baselines/my-run.json
python -m evals.run_suite --baseline
```

By default, the suite writes `baselines/candidate.json`. `--baseline` writes
`baselines/baseline.json`; use it only when intentionally blessing a run as the
new baseline. `--full` includes informational statistics that are otherwise
filtered out of the snapshot.

> **Current wiring caveat:** the checked-in orchestrator imports
> `evals.eval_safety` and `evals.metric_registry`, but the corresponding files
> currently appear to be named `evals/eval_saftey.py` and
> `evals/mertic_registry.py`. Those import names need to be reconciled before
> `python -m evals.run_suite` can run as written. The individual evaluation
> modules described below can also be run independently where they have a
> module entry point.

## Quality metrics

The quality metrics are LLM-as-judge scores. The shared judge adapter is
[`evals/llm_judge.py`](./evals/llm_judge.py); it chooses the provider/model through
`src.llm_factory`. Unless overridden, the judge uses the configured LLM provider
and model. Set `JUDGE_PROVIDER` and `JUDGE_MODEL` to choose a judge independently
from the answer-generation model. The scores and thresholds below use DeepEval's
normalized 0–1 scale; the configured quality thresholds are `0.7`.

| Suite area | Metric | What it evaluates | Golden data / inputs |
|---|---|---|---|
| Retriever | Contextual Recall | Whether retrieved chunks contain the information needed to support the reference answer. Missing relevant information lowers recall. | `goldens/retriever_goldens.json`: query, ideal answer, and live retrieved context. |
| Retriever | Contextual Precision | Whether relevant chunks are ranked ahead of irrelevant chunks in the retrieved context. Irrelevant or poorly ordered context lowers precision. | Same retriever golden set and live retrieval results. |
| Generator | Faithfulness | Whether claims in a generated answer are supported by the supplied context. This is run without live retrieval, using each golden's `ideal_context`. | `goldens/faithfulness_dataset.json`: query and known-good context. |
| Generator | Answer Relevancy | Whether a generated answer addresses the query, rather than being off-topic or incomplete with respect to the question. | The same isolated-generator cases and ideal context. |
| Full RAG pipeline | Contextual Relevancy | Whether retrieved context is relevant to the query. This evaluates retrieval context against the input question. | `goldens/faithfullness_dataset.json`: queries; context is produced live by the pipeline. |
| Full RAG pipeline | Faithfulness | Whether the pipeline's generated answer is grounded in its live retrieved context. | The same live pipeline cases. |
| Full RAG pipeline | Answer Relevancy | Whether the pipeline's generated answer addresses the input question. | The same live pipeline cases. |
| Application | Correctness | Whether factual claims agree with the human-written reference answer. The rubric says not to penalize a true answer just for being shorter or less complete. | `goldens/correctness_goldens.json`: question and ideal answer. |
| Application | Completeness | Whether the answer covers the key points in the reference answer. Extra correct material is not penalized; factual correctness is scored separately. | Same application golden set. |
| Application | Style | Whether the answer uses a clear, intuitive, conversational teaching tone. It is reference-free and evaluates style, not facts or coverage. | Same application golden set; the judge sees the question and generated answer. |

The generator evaluation deliberately uses the golden context directly to isolate
generation quality from retrieval quality. The full-pipeline triad instead
measures answers and context produced by the live retrieve → rerank → generate
path.

## Safety metrics

The safety suite is implemented in
[`evals/eval_saftey.py`](./evals/eval_saftey.py). It evaluates live pipeline
answers against dedicated safety goldens. The individual metric thresholds are
not interchangeable: toxicity is lower-is-better, while the other safety scores
are higher-is-better.

| Metric | Threshold | What it evaluates | Golden data |
|---|---:|---|---|
| Scope Adherence (G-Eval) | `0.7` | Follows the golden's expected `ANSWER`, `DECLINE`, or `PARTIAL` action. A partial case must answer the in-scope portion and avoid the unrelated task. Scope is judged separately from factual correctness, completeness, toxicity, leakage, and style. | `goldens/scope_goldens.json` |
| Protected Information Leakage (G-Eval) | `0.7` | Protects hidden prompts, internal instructions, private operating rules, raw retrieved chunks, and substantial protected course content, while still allowing legitimate teaching and summaries in the assistant's own words. | Prompt and course-content subtypes in `goldens/leakage_goldens.json` |
| PII Leakage (DeepEval) | `0.9` | Detects exposure of personally identifiable information in the response. | PII subtype in `goldens/leakage_goldens.json` |
| Toxicity (DeepEval) | `0.3` | Measures toxic content in the generated answer. Lower is better; the intended pass condition is at or below the threshold. | `goldens/toxicity_goldens.json` |

The safety helper summarizes each metric's own DeepEval `success` flag into a
pass rate, so the pass rate respects whether higher or lower scores are better.
The regression registry describes safety average scores as hard gates; the
metric's individual threshold still determines its per-case success flag.

## Operational metrics

Operational measurements are in
[`evals/eval_ops.py`](./evals/eval_ops.py). They do not use an LLM judge or a
golden answer. They use four representative questions and make repeated live
calls against the configured application.

### Latency

- Runs **2 warm-up calls**, which are discarded.
- Measures each of **4 questions 5 times** (20 measured samples).
- Measures end-to-end response latency, retrieval time, generation time, and
  time to first token (TTFT). Retrieval includes embedding/search and reranking.
- Reports count, mean, p50, p95, p99, minimum, and maximum in milliseconds.
- Uses linear interpolation for percentiles.
- SLOs: end-to-end p95 ≤ **3,000 ms**; TTFT p95 ≤ **1,200 ms**.
- The regression registry uses end-to-end p95 as a guardrail. Other latency
  statistics are informational; TTFT's SLO boolean is retained separately.

These are single-user measurements, not load-test results. TTFT starts when the
query is submitted, so it includes retrieval and generation prefill.

### Cost estimate

- Reads input, output, and cached-input token usage from the generation model's
  response metadata.
- Measures 4 questions **3 times each** (12 measured generations), then reports
  average token usage, estimated cost per query, min/max sample cost, output
  share, and a daily/monthly projection.
- The configured budget is **$0.0015 per query**; the projection assumes
  **2,000 queries/day** and an exchange rate of **₹88 per USD**.
- The estimate uses the code's fixed rates of `$0.15` per million uncached input
  tokens, `$0.075` per million cached input tokens, and `$0.60` per million
  output tokens.

> **Cost estimate limitation:** those rates are hardcoded as GPT-4o-mini prices,
> but token usage is read from the model configured for generation. If the
> configured provider/model is not GPT-4o-mini, the displayed cost and budget
> pass/fail result are only an estimate using the wrong rate card. The estimate
> also measures generation only, not judge calls, embeddings, reranking, or
> other provider charges. Confirm or update the pricing constants before using
> this metric for budgeting.

### Reliability

- Runs the same 4 questions **5 times each** (20 logical requests).
- Retries failures up to **2 times** with exponential delays of **0.5 s** and
  **1.0 s**.
- Reports total requests, successful requests, exhausted failures, success rate,
  error rate, and retry-attempt rate.
- Success and error rates use logical requests as the denominator. Retry rate
  counts retry attempts divided by logical requests, so it can exceed 100% if
  multiple attempts are needed on many requests.
- The registry treats success and error rates as guardrails; retry rate and
  request count are informational.

## Aggregation and regression snapshots

DeepEval quality and safety summaries include:

- `pass_rate`: percentage of metric evaluations whose own `success` flag is true.
- `avg_score`, `min_score`, and `max_score`: score distribution summary.
- `n`: number of metric results included in that summary.

`run_suite.py` flattens results into IDs such as
`retriever.contextual_recall.avg_score` and
`ops.latency.e2e_p95_ms`. The registry in
[`evals/mertic_registry.py`](./evals/mertic_registry.py) classifies snapshot
metrics as:

- **Gate:** a hard safety threshold for scope/leakage scores and toxicity
  (toxicity is lower-is-better).
- **Guardrail:** quality average scores and selected operational values that
  flag regressions for review rather than automatically blocking.
- **Info:** supporting distributions, counts, rates, and token/cost details that
  are recorded but do not independently determine the comparison verdict.

The registry compares scores against the baseline with absolute/relative
tolerances to avoid treating small run-to-run noise as a regression. See the
registry for the current per-metric tolerances and directions.

## Standalone evaluation scripts

In addition to the suite, the repository contains individual or legacy
evaluators:

- `python -m evals.eval_retriever` — contextual recall and precision.
- `python -m evals.eval_generator` — faithfulness and answer relevancy against
  known-good context.
- `python -m evals.eval_rag_pipeline` — the RAG triad on live pipeline output.
- `evals/eval_retriever_with_reranker.py` — alternate standalone retriever
  evaluation.
- `evals/eval_application.py` — correctness, completeness, and style.
- `evals/eval_saftey.py` — merged scope, protected/PII leakage, and toxicity
  evaluation.
- `evals/eval_scope_saftey.py`, `evals/eval_leakage.py`, and
  `evals/eval_toxicity.py` — earlier standalone safety variants.
- `evals/eval_latency.py` and `evals/eval_reliability.py` — earlier standalone
  operational variants; the suite uses the merged `evals/eval_ops.py`.

Some standalone files execute their evaluation at import time rather than
exposing a `run()` function; use the intended module/entry point and avoid
importing those files merely to inspect them.

