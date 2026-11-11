# CSE 573 - Agentic Text2SQL

This repository contains the current data foundation and classical baseline for our Agentic Text2SQL project.

The goal of the overall project is to build a two-agent Text2SQL system where a primary agent generates SQL over an idealized schema and a second agent maps the required views to the actual database schema.

## Dataset Setup

The raw Spider datasets are **not included in this GitHub repository** because several database files exceed GitHub's file-size limit.

I will share the following two dataset folders separately: Copy data.zip to data/, so it contains data/Spider1 and data/Spider2

```text
Spider1/
Spider2/
```

After cloning this repository, copy both folders into the repository's `data/` directory.

## What Is Implemented

### 1. Dataset Preparation

Currently integrated:

- Spider 1.0
- Spider 2.0-Lite

Spider 1:

- 8,659 training examples
- 1,034 development examples
- 166 SQLite databases

Spider 2.0-Lite:

- 547 total benchmark examples
- 123 verified local SQLite examples
- 30 local SQLite databases
- Remaining examples are retained for future Snowflake/BigQuery support

Both datasets are converted into a common `Text2SQLExample` format so later components can use the same interface.

### 2. Schema Extraction

SQLite schema metadata is automatically extracted for all local databases.

Extracted information includes:

- Tables
- Columns
- Data types
- Primary keys
- Foreign keys
- Nullability
- Default values

Current results:

- Spider 1: 166/166 databases successfully extracted
- Spider 2: 30/30 databases successfully extracted

Generated schemas are stored in:

```text
outputs/schemas/
├── spider1/
└── spider2/
```

### 3. Retrieval Documents

Schemas are converted into table-level text documents that can be used by the BM25/retrieval component.

Generated:

- Spider 1: 873 documents
- Spider 2: 412 documents
- Total: 1,285 documents

Located in:

```text
outputs/retrieval/
```

These can be used directly as the initial document collection for the retrieval system.

### 4. Agent Training Data

Spider 1 training examples are exported with their physical database schemas.

Generated:

```text
outputs/training/primary_agent_train.jsonl
```

Current result:

- 8,659 training examples
- 8,659/8,659 have matching physical schemas

The `ontology` and `declared_schema` fields are intentionally left empty so they can be populated by the ontology/agent components.

### 5. Classical Text2SQL Baseline

A restricted non-LLM baseline is implemented using:

```text
Question
   ↓
TF-IDF
   ↓
k-NN intent classification
   ↓
Lexical schema matching
   ↓
SQL templates
   ↓
SQLite execution
```

The baseline supports simple SELECT, COUNT and SUM-style queries. More complex filtering and SQL structures are intentionally treated as unsupported.

Final Spider 1 dev results:

| Metric | Result |
|---|---:|
| Dev Questions | 1,034 |
| Supported | 695 |
| Coverage | 67.21% |
| Execution Success Rate | 66.15% |
| Execution Correct | 39 |
| Overall Execution Accuracy | 3.77% |
| Supported Execution Accuracy | 5.61% |
| Missing DBs | 0 |
| Missing Schemas | 0 |

The low execution accuracy is expected because this is intentionally a limited classical baseline that will be compared against the later LLM systems.

## Running the Project

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Prepare all datasets, schemas, retrieval documents and training examples:

```bash
python scripts/prepare_all.py
```

Run the classical baseline:

```bash
python scripts/run_baseline.py
```

Run tests:

```bash
pytest -q
```

Current test result:

```text
10 passed
```

## Important Outputs

```text
outputs/
├── normalized/
│   ├── spider1_train.json
│   ├── spider1_dev.json
│   ├── spider2_lite_all.json
│   └── spider2_lite_sqlite.json
│
├── schemas/
│   ├── spider1/
│   └── spider2/
│
├── retrieval/
│   ├── spider1_schema_docs.jsonl
│   └── spider2_schema_docs.jsonl
│
├── training/
│   └── primary_agent_train.jsonl
│
└── baseline_results/
    ├── spider1_dev_predictions.json
    └── metrics.json
```

## How the Team Can Build on This

### Ontology / Retrieval

Use:

```text
outputs/retrieval/
outputs/schemas/
```

Next steps:

- Add business ontology information
- Add database/business documentation
- Build the BM25 index
- Compare BM25 against TF-IDF/cosine retrieval

### Primary Agent

Use:

```text
outputs/training/primary_agent_train.jsonl
outputs/normalized/
```

Next steps:

- Add ontology information
- Generate declared idealized schemas
- Generate SQL over the idealized schema
- Integrate Qwen2.5-Coder-7B-Instruct

### View-Creation Subagent

Use the generated schemas and retrieval system to:

- Find real tables/columns
- Identify joins and keys
- Retrieve relevant documentation/business rules
- Generate the required view mappings/TRC

### Evaluation

Use:

```text
outputs/normalized/spider1_dev.json
outputs/normalized/spider2_lite_sqlite.json
outputs/baseline_results/
```

The classical baseline results above provide the initial comparison point.

## Remaining Work for the Full Project

The major remaining components are:

1. Business ontology
2. BM25 retrieval implementation
3. Primary LLM agent
4. View-creation subagent
5. Restricted Tuple Relational Calculus (TRC) representation
6. Deterministic TRC-to-SQL view compiler
7. Single-agent LLM baseline
8. End-to-end two-agent pipeline
9. QLoRA fine-tuning if compute permits
10. Spider 2.0-Lite enterprise evaluation
11. Ablation experiments and error analysis
12. Final comparison of:
    - Classical baseline
    - Single-agent LLM
    - Two-agent proposed system

The current repository provides the dataset, schema, retrieval-document, training-data and classical-baseline foundation for these remaining components.
## Evaluation Harness (`src/text2sql_eval`)

Shared harness for scoring every system (classical, single-agent LLM, two-agent)
on identical question IDs with the official benchmark evaluators.
Interfaces for the team are in [`docs/CONTRACTS.md`](docs/CONTRACTS.md).

```bash
bash scripts/setup_evaluators.sh            # official Spider 1 evaluator (pinned commit)
python scripts/build_manifests.py           # manifests/*.jsonl with exclusion reasons
python scripts/run_eval.py --system oracle --benchmark spider1 --limit 50   # harness self-check
python scripts/run_eval.py --system classical --benchmark spider1
pytest tests/eval -q
```

Each run writes `outputs/runs/<run_id>/` with `runs.jsonl` (one record per question,
including failures), `results_summary.csv`, `errors.csv`, `results_by_difficulty.csv`
(Spider 1, official easy/medium/hard/extra) and `config.json`
(git commit, evaluator version, packages, GPU).

Budgets: the harness enforces SQL timeout and row limits itself, and flags any
question where a system's reported model calls, retries or input tokens exceed
the `Budget` (`within_budget`, `budget_violations` per record). Summaries report
both official EX and `execution_accuracy_within_budget`; matched-budget
comparisons use the latter.

Plugging in another system (e.g. the two-agent pipeline) without editing the harness:

```bash
python scripts/run_eval.py --system-module text2sql_two_agent.adapter:build \
    --system-option views=on --benchmark spider1 --sample 50
```

`build(args)` gets the parsed CLI arguments (LLM endpoint and decoding settings,
`args.system_options`) and returns an object with `.name` and
`.predict(question, budget)`. It never receives the evaluator or gold data.

Analysis across runs:

```bash
# mean ± sample std over repeats (runs must cover the same questions and scoring mode)
python scripts/analyze_runs.py aggregate outputs/runs/sa_rep0 outputs/runs/sa_rep1 outputs/runs/sa_rep2
# paired comparison: per-question table, exact McNemar p, paired bootstrap 95% CI
python scripts/analyze_runs.py compare outputs/runs/two_agent_rep0 outputs/runs/sa_rep0 --within-budget
# add difficulty labels to a Spider 1 run made before they existed
python scripts/analyze_runs.py report outputs/runs/<run_id> --add-difficulty
```

| Module | Purpose |
|---|---|
| `contracts.py` | `QuestionInput` (no gold), `PredictionResult`, `Budget`, statuses |
| `manifest.py` | Question manifests; Spider 2.0-Lite = 135 local SQLite of 547 |
| `sql_safety.py`, `sandbox.py` | Read-only validation, read-only DB + authorizer, timeouts, TEMP views |
| `view_composition.py` | Two-agent views → standalone `WITH` query for official scoring |
| `official/` | Wrappers for test-suite-sql-eval and Spider 2.0-Lite evaluation_suite |
| `runner.py`, `metrics.py` | Per-question records, budget checks, EX / cost / difficulty summaries |
| `analysis.py` | Repeat aggregation (mean ± std), paired comparison (McNemar, bootstrap CI) |
| `systems/` | Sanity systems (gold oracle, always-unsupported) and classical-baseline adapter |

### Single-agent LLM baseline (`src/text2sql_agents`, `src/text2sql_llm`, `src/text2sql_retrieval`)

Qwen2.5-Coder-7B-Instruct generates SQL directly over the physical schema
(CREATE TABLE text + sample rows + any benchmark evidence), then the SQL is
checked, executed read-only, and repaired with the error message up to 3 times.
Prompts are versioned (`single_agent/v1`); every attempt, token count, latency
and peak GPU memory is logged per question.

```bash
bash scripts/llm/setup_llama_server.sh     # once: llama.cpp (CUDA) + pinned Q4_K_M GGUF
bash scripts/llm/start_llama_server.sh     # serves http://127.0.0.1:8080/v1
python scripts/run_eval.py --system single_agent --benchmark spider1 --sample 50 --sample-seed 0
python scripts/run_eval.py --system single_agent --benchmark spider2_lite
```

Any OpenAI-compatible server works: `--llm-base-url http://host:port/v1 --llm-model <name>`.
