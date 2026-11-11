# Team Contracts (draft — please review)

Proposed interfaces between components. Owner of this doc: Soham.
Anything marked **OPEN** needs agreement from the named people.

## 1. Question IDs and manifests (Soham + Shashank)

- IDs are permanent and match `text2sql_data`:
  `spider1_dev_000123` (index into `dev.json`), Spider 2 `instance_id` (`local002`).
- `manifests/*.jsonl` lists **every** benchmark question with
  `included` and `exclusion_reason`. Build with `python scripts/build_manifests.py`.
- Current manifests:
  - `spider1_dev`: 1034 / 1034 included.
  - `spider2_lite`: 135 / 547 included (all `local*` SQLite instances);
    207 Snowflake + 205 BigQuery excluded (need cloud access).
    Results on it must be reported as "Spider 2.0-Lite local SQLite subset (135)".
- **OPEN (Shashank):** `text2sql_data.load_spider2_lite` finds only 123 local
  instances. It drops `sqlite-sakila` (7) and `Db-IMDB` (5) because the
  metadata folders are named `SQLITE_SAKILA` / `DB_IMDB`. The manifest uses
  `spider2-localdb/*.sqlite` directly and keeps all 135.

## 2. System adapter (everyone)

```python
def predict(question: QuestionInput, budget: Budget) -> PredictionResult
```

Defined in `src/text2sql_eval/contracts.py`.

- `QuestionInput`: `question_id, benchmark, db_id, db_path, question, dialect, evidence`.
  **No gold SQL or gold results, ever.** `evidence` = benchmark-provided docs
  (Spider 2 `external_knowledge`), which every system may use.
- `PredictionResult`: as in the team spec. Report `status="success"` with SQL,
  or one of `unsupported | parse_error | compile_error | timeout` with
  `error_message`. Fill token / call / retry counts honestly; leave
  `peak_gpu_memory_mb=None` if not measured (never 0).
- The harness — not the system — executes SQL, decides `execution_error`,
  `unsafe_sql`, `view_error`, `eval_error`, and scores correctness.
- Run your system without editing harness code:
  `python scripts/run_eval.py --system-module your_package.module:build --system-option key=value ...`.
  `build(args)` receives the CLI arguments (LLM endpoint, decoding settings,
  `args.system_options`) and returns an object with `.name` and `.predict()`.
  It never receives the evaluator or any gold data.

## 3. Two-agent view artifacts (Nolan)

Put compiled views in `intermediate_artifacts`:

```json
{
  "views": [{"name": "Employee", "sql": "SELECT s.staff_id AS employee_id, ... FROM staff s JOIN ..."}],
  "idealized_schema": {"Employee": [["employee_id", "INTEGER"], ["department_name", "TEXT"]]},
  "trc": {"Employee": "..."},
  "compile_status": "ok"
}
```

- `generated_sql` = primary SQL over the idealized relations.
- The harness creates the views as SQLite **TEMP views** on a read-only
  connection (memory only, discarded after every attempt), runs the primary
  SQL, then scores the equivalent standalone query
  `WITH "Employee" AS (...) <primary SQL>` with the official evaluator.
- A TRC compile failure → return `status="compile_error"` so it is
  distinguishable from a mapping (view_error / wrong result) failure.
- **OPEN (Nolan):** declared column/type check of views — proposal: the harness
  compares `idealized_schema` with the TEMP view's `PRAGMA table_info`.

## 4. Retrieval: `search_docs` (Jay)

**OPEN (Jay):** proposed signature, shared by the single-agent and two-agent systems:

```python
def search_docs(query: str, db_id: str, k: int) -> list[RetrievedDoc]   # RetrievedDoc(doc_id, text, score, source)
```

- Same index, same `k` and same total search budget for both LLM systems.
- Index content: schema docs (`outputs/retrieval/`), ontology entries, business docs.
  Must not be built from dev/test gold SQL.

Interim implementation (Soham, until Jay's index exists): `src/text2sql_retrieval/`
renders each table as CREATE TABLE text + 3 sample rows and ranks tables with
BM25 (`rank-bm25`). `select_tables(question, docs, token_budget, top_k)` fills the
budget with (1) question-matching tables, (2) their foreign-key targets,
(3) the rest by rank. With the default 6k-token budget every local DB fits
completely (largest: Baseball, ~4.8k tokens, 26 tables; most tables in one DB: 38),
so the two-agent system should get the **same** schema access for fairness.

## 5. LLM inference wrapper (Nolan + Jay + Soham)

Implemented in `src/text2sql_llm/`: `LLMClient.generate(messages, params) -> Generation`
(`text, input_tokens, output_tokens, latency_s, model, finish_reason`).
Token counts come from the server's `usage` field. `OpenAICompatibleClient` works
with llama.cpp, Ollama and vLLM, so any teammate's GPU can serve the model.

Local serving (pinned, reproducible): `scripts/llm/setup_llama_server.sh` then
`scripts/llm/start_llama_server.sh` →
`Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` @ `13fb94bf`, file `q4_k_m` (SHA-256 checked),
llama-cpp-python 0.3.16 (CUDA 12.4), all 29 layers on GPU, 12k context, chatml.
Default decoding: temperature 0, top_p 1, seed 0, max 1024 new tokens.
**OPEN (Nolan):** use this same endpoint/quantization for the two-agent runs.
Hardware note: Soham's GPU is an RTX 4060 Laptop (8 GB, ~7 GB used while serving);
QLoRA training will need a bigger GPU.

## 6. Budgets and safety (everyone)

`Budget` defaults: max 3 repair retries, 8 model calls, 32k total input tokens,
30 s SQL timeout, 100k result rows. Same budget for both LLM systems in a
matched comparison; every retry counts toward tokens and latency.

The harness checks each prediction's reported `model_calls`, `retry_count` and
`input_tokens` against the budget and records `within_budget` /
`budget_violations`. Official EX is reported unchanged; matched comparisons use
`execution_accuracy_within_budget` (over-budget answers count as wrong).
Report these counters honestly — they are the only way the harness can check them.

Execution safety (enforced by harness): single SELECT/WITH only,
read-only database connection, SQLite authorizer, per-query deadline,
database fingerprint check after each question (run aborts if any DB changes).

## 7. Scoring

- Spider 1: official `test-suite-sql-eval` `eval_exec_match` (pinned commit,
  `scripts/setup_evaluators.sh`) on the original Spider databases → standard EX.
- Spider 2.0-Lite local: official `evaluation_suite/evaluate.py` comparison
  functions (loaded unmodified; file SHA-256 recorded) against gold CSVs.
- EX = correct / all included questions. Unsupported, errors and timeouts are failures.
- Spider 1 results are also broken down by the official difficulty classifier
  (easy / medium / hard / extra, computed from gold SQL after prediction).
- Repeated runs: report mean ± sample std over repeats (`analyze_runs.py aggregate`).
  Runs are only combined if they cover the same questions with the same scoring mode.
- System comparisons: paired on identical questions (`analyze_runs.py compare`),
  with an exact McNemar test and a paired bootstrap 95% CI for the EX difference.
- **OPEN (team):** what varies across the 3 repeats. At temperature 0 repeats are
  near-identical (std ≈ 0); options are different seeds at a small temperature
  (e.g. 0.2) or temperature 0 reported as a determinism check.
