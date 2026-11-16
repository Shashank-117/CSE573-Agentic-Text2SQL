# Two-Agent Integration Guide (for Nolan)

How to plug the two-agent pipeline (primary agent + view subagent + TRC compiler)
into the shared evaluation harness so it is scored exactly like the single-agent
and classical baselines. The harness is merged into `main` (PR #1).

Questions → Soham. Full interface definitions: [`docs/CONTRACTS.md`](CONTRACTS.md).

---

## 1. Setup (once)

```bash
git checkout main && git pull
git checkout -b nolan/two-agent           # or any branch name

pip install -r requirements.txt && pip install -e .
bash scripts/setup_evaluators.sh          # official Spider 1 evaluator (pinned) + NLTK data

bash scripts/llm/setup_llama_server.sh    # llama.cpp (CUDA) + pinned Qwen2.5-Coder-7B Q4_K_M (~4.7 GB)
bash scripts/llm/start_llama_server.sh    # serves http://127.0.0.1:8080/v1 (~7 GB VRAM)
```

Data must be under `data/Spider1` and `data/Spider2/spider2-lite` (same layout as
Shashank's pipeline). Check the harness works before plugging in your system:

```bash
python scripts/run_eval.py --system oracle --benchmark spider1 --limit 20    # must be 20/20
pytest tests/eval -q
```

If you can't run the model locally, any OpenAI-compatible server works
(`--llm-base-url http://host:port/v1 --llm-model <name>`), **but final
comparison runs must use the same model and quantization as the single agent**.

---

## 2. What you implement

Suggested location: `src/text2sql_agents/two_agent/` (the `text2sql_agents`
package is shared; the single agent lives next to it).

```python
# src/text2sql_agents/two_agent/adapter.py
from text2sql_eval.contracts import Budget, PredictionResult, QuestionInput, Status
from text2sql_llm import GenerationParams, OpenAICompatibleClient


class TwoAgentSystem:
    name = "two_agent"

    def __init__(self, llm, params, options: dict[str, str]):
        self.llm, self.params, self.options = llm, params, options

    def predict(self, question: QuestionInput, budget: Budget) -> PredictionResult:
        # 1. primary agent: question -> SQL over an idealized schema (+ TRC per relation)
        # 2. view subagent + TRC compiler: idealized relations -> views over the real DB
        # 3. return SQL + views; the harness runs and scores them
        return PredictionResult(
            question_id=question.question_id,
            db_id=question.db_id,
            system_name=self.name,
            generated_sql="SELECT department_name, COUNT(*) FROM Employee GROUP BY 1",
            status="success",
            model_calls=...,          # every LLM call, both agents, all retries
            retry_count=...,
            input_tokens=...,         # sum over all calls
            output_tokens=...,
            max_input_tokens_single_call=...,
            intermediate_artifacts={
                "views": [{"name": "Employee",
                           "sql": "SELECT s.staff_id AS employee_id, d.name AS department_name "
                                  "FROM staff s JOIN dept d ON s.dept_id = d.id"}],
                "idealized_schema": {"Employee": [["employee_id", "INTEGER"],
                                                  ["department_name", "TEXT"]]},
                "trc": {"Employee": "{ e | ... }"},
                "compile_status": "ok",
            },
        )


def build(args):
    """Factory called by scripts/run_eval.py. Never receives the evaluator or gold data."""
    llm = OpenAICompatibleClient(base_url=args.llm_base_url, model=args.llm_model)
    params = GenerationParams(temperature=args.temperature, top_p=args.top_p,
                              max_new_tokens=args.max_new_tokens,
                              seed=args.seed if args.seed is not None else 0)
    return TwoAgentSystem(llm, params, args.system_options)
```

### Inputs: `QuestionInput`
`question_id, benchmark, db_id, db_path, question, dialect, evidence`
(`evidence` = benchmark-provided docs, a tuple of `EvidenceDoc(doc_id, text)`).
**It never contains gold SQL or gold results.**

### Outputs: `PredictionResult`
| Field | What to put there |
|---|---|
| `generated_sql` | Primary agent's SQL over the idealized relations |
| `intermediate_artifacts["views"]` | `[{"name": ..., "sql": "SELECT ..."}]`, one per idealized relation |
| `status` | `"success"`, or `compile_error` (TRC → view compile failed), `unsupported` (system declines), `parse_error` (no SQL produced) |
| `model_calls`, `retry_count`, `input_tokens`, `output_tokens` | Honest totals over **both** agents |
| `error_message` | Short reason when `status` isn't success |
| `peak_gpu_memory_mb` | Leave `None` if not measured (never `0`); `text2sql_llm.GpuPeakSampler` can measure it |

Put anything else you want logged for error analysis (TRC text, idealized
schema, per-agent attempts, prompt version) in `intermediate_artifacts`. It
ends up in `runs.jsonl`.

---

## 3. What the harness does with your output

1. Checks `generated_sql` is a single read-only `SELECT`/`WITH`.
2. Creates your views as SQLite **TEMP views** on a **read-only** connection
   (in memory, discarded after the question). A view that fails → `view_error`.
3. Runs your SQL over the views (30 s timeout). Errors → `execution_error` / `timeout`.
4. Rewrites it into the equivalent standalone query
   `WITH "Employee" AS (<view sql>) <your SQL>` and scores **that** with the
   official evaluator (Spider 1 test-suite / Spider 2.0-Lite evaluation suite).
5. Checks your budget counters (8 model calls, 3 retries, 32k input tokens).
   Over budget → flagged, and counted wrong in the matched comparison.

So the failure statuses separate *where* things broke:
`compile_error` (TRC compiler) → `view_error` (view SQL invalid) →
`execution_error` (primary SQL invalid) → executed but wrong (mapping/logic).

### Rules
- **Don't decide correctness yourself.** You *may* execute SQL for repair
  feedback (the single agent does). Use the same sandbox so it's read-only and
  time-limited:
  ```python
  from text2sql_eval.sandbox import ViewDefinition, run_query
  result = run_query(question.db_path, sql, views=[ViewDefinition(name, view_sql)],
                     timeout_s=budget.sql_timeout_s, max_rows=50)
  if not result.ok: feedback = result.error
  ```
- Never open the database read-write, and never write to it. The harness
  fingerprints every DB and aborts the run if one changes.
- Never read gold files (`dev.json` queries, `gold/` CSVs, `*_gold.sql`).

---

## 4. Fairness: same resources as the single agent

| | Must match the single agent |
|---|---|
| Model | Qwen2.5-Coder-7B-Instruct, pinned Q4_K_M GGUF, via `text2sql_llm` |
| Decoding | temperature 0, top_p 1, seed 0, max 1024 new tokens (the CLI defaults) |
| Budget | 3 repair retries, 8 model calls, 32k input tokens per question |
| Schema access | `text2sql_retrieval.build_table_docs` + `select_tables` with a 6000-token budget (every local DB fits completely) |
| SQLite guidance | shared module `text2sql_agents.dialect_hints`. Put `dialect_hints(question.dialect)` in both agents' prompts (see below) |
| Evidence | Spider 2 `external_knowledge` docs via `question.evidence`, same as the single agent |

Schema access example:
```python
from text2sql_retrieval import build_table_docs, select_tables

docs = build_table_docs(question.db_path, question.db_id, sample_rows=3)
selection = select_tables(question.question, docs, token_budget=6000,
                          estimate_tokens=lambda t: int(len(t) / 3.5) + 1)
schema_text = "\n\n".join(t.prompt_text for t in selection.tables)
```

Jay's `search_docs` index will replace the BM25 placeholder for both systems.

SQLite guidance (identical text in both systems; every line is tested against the real engine):
```python
from text2sql_agents.dialect_hints import DIALECT_HINTS_VERSION, dialect_hints

system_prompt = YOUR_RULES + "\n\n" + dialect_hints(question.dialect)   # "" for non-SQLite
# log DIALECT_HINTS_VERSION in intermediate_artifacts
```

Optional, for your repair loops: `text2sql_agents.repair_feedback.explain_error(error, sql, schema)`
turns a SQLite error into schema-based guidance (for example, "column X is not in table T, it
exists in U"). The single agent uses it. If your agents have their own repair loops, using it
too keeps the comparison even.

---

## 5. Running and comparing

```bash
# quick check on 50 questions
python scripts/run_eval.py --system-module text2sql_agents.two_agent.adapter:build \
    --benchmark spider1 --sample 50 --sample-seed 0 --run-id two_agent_s50

# your own options reach build() as args.system_options
python scripts/run_eval.py --system-module text2sql_agents.two_agent.adapter:build \
    --system-option view_check=on --benchmark spider2_lite --run-id two_agent_sp2

# paired comparison with the single agent on the same questions
python scripts/analyze_runs.py compare outputs/runs/two_agent_s50 outputs/runs/<single_agent_run> --within-budget
```

Each run writes `outputs/runs/<run_id>/`:
- `runs.jsonl`: one record per question, including your artifacts.
- `results_summary.csv`: EX overall and within budget, status counts, tokens, latency.
- `results_by_difficulty.csv`: Spider 1 easy, medium, hard and extra.
- `errors.csv`: every failed question with its error.
- `config.json`: git commit, settings and environment.

`compare` prints two-agent minus single-agent EX with a 95% CI and a McNemar
p-value. It also writes `paired.csv`, showing which questions only one system
got right, which is the starting point for error analysis.

Use `--resume` to continue an interrupted run.

---

## 6. Testing your adapter without the GPU

`text2sql_llm.ScriptedLLM` returns canned responses, so you can unit-test the
pipeline logic. See `tests/agents/test_single_agent.py` for the pattern and
`tests/eval/test_runner.py::test_two_agent_views_scored_as_standalone_cte` for
how views are scored.

---

## 7. Decisions I need from you

1. **View format.** Is `intermediate_artifacts["views"] = [{"name", "sql"}]` (plus
   `idealized_schema`, `trc`) OK?
2. **Declared column/type check.** Should the harness compare your
   `idealized_schema` with each TEMP view's actual columns (`PRAGMA table_info`)
   and report mismatches separately from wrong answers?
3. **Model.** Will you use the same llama.cpp server and Q4_K_M quantization for
   the final runs?
4. **Repair loop.** If your agents have their own retry loop, do you want the same
   "you repeated the same SQL" feedback I'm adding to the single agent?

Happy to pair on the first end-to-end run.
