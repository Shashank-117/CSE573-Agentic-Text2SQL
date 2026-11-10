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

## 5. LLM inference wrapper (Nolan + Jay + Soham)

**OPEN:** one shared wrapper returning `text, input_tokens, output_tokens, latency_s`,
logging model id, quantization, temperature, top_p, seed, max_new_tokens.
Hardware note: Soham's GPU is an RTX 4060 Laptop (8 GB) → 4-bit 7B inference
fits; QLoRA training likely does not.

## 6. Budgets and safety (everyone)

`Budget` defaults: max 3 repair retries, 8 model calls, 32k total input tokens,
30 s SQL timeout, 100k result rows. Same budget for both LLM systems in a
matched comparison; every retry counts toward tokens and latency.

Execution safety (enforced by harness): single SELECT/WITH only,
read-only database connection, SQLite authorizer, per-query deadline,
database fingerprint check after each question (run aborts if any DB changes).

## 7. Scoring

- Spider 1: official `test-suite-sql-eval` `eval_exec_match` (pinned commit,
  `scripts/setup_evaluators.sh`) on the original Spider databases → standard EX.
- Spider 2.0-Lite local: official `evaluation_suite/evaluate.py` comparison
  functions (loaded unmodified; file SHA-256 recorded) against gold CSVs.
- EX = correct / all included questions. Unsupported, errors and timeouts are failures.
