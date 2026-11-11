"""Run a system through the evaluation harness.

Examples:
    # Harness self-checks (oracle must be ~100%, unsupported must be 0%)
    python scripts/run_eval.py --system oracle --benchmark spider1 --limit 50
    python scripts/run_eval.py --system always_unsupported --benchmark spider2_lite

    # Classical baseline on full Spider 1 dev
    python scripts/run_eval.py --system classical --benchmark spider1

    # Any external system (e.g. the two-agent pipeline): module:factory
    python scripts/run_eval.py --system-module text2sql_two_agent.adapter:build \
        --system-option views=on --benchmark spider1 --sample 50
    # factory(args) receives the parsed CLI args (LLM settings, args.system_options)
    # and returns an object with .name and .predict(question, budget).

Outputs go to outputs/runs/<run_id>/: runs.jsonl, results_summary.csv,
errors.csv, config.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

from text2sql_eval.contracts import Budget
from text2sql_eval.manifest import read_manifest, select
from text2sql_eval.metrics import load_records, summarize_by_difficulty, summarize_by_group, write_reports
from text2sql_eval.official import get_evaluator
from text2sql_eval.paths import MANIFEST_DIR, RUNS_DIR
from text2sql_eval.provenance import collect_environment
from text2sql_eval.runner import RunConfig, run
from text2sql_eval.systems import AlwaysUnsupportedSystem, GoldOracleSystem

MANIFEST_FOR = {"spider1": "spider1_dev.jsonl", "spider2_lite": "spider2_lite.jsonl"}
SYSTEMS = ("oracle", "always_unsupported", "classical", "single_agent")


def _build_single_agent(args: argparse.Namespace):
    from text2sql_agents import SingleAgentConfig, SingleAgentSystem
    from text2sql_llm import GenerationParams, OpenAICompatibleClient

    llm = OpenAICompatibleClient(base_url=args.llm_base_url, model=args.llm_model)
    params = GenerationParams(temperature=args.temperature, top_p=args.top_p,
                              max_new_tokens=args.max_new_tokens,
                              seed=args.seed if args.seed is not None else 0)
    config = SingleAgentConfig(schema_token_budget=args.schema_token_budget,
                               schema_top_k=args.schema_top_k, params=params)
    return SingleAgentSystem(llm, config)


def _parse_options(pairs: list[str] | None) -> dict[str, str]:
    options = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--system-option must be key=value, got {pair!r}")
        options[key] = value
    return options


def load_external_system(spec: str, args: argparse.Namespace):
    """Build a system from 'package.module:factory'. The factory never sees the evaluator (gold)."""
    module_name, sep, factory_name = spec.partition(":")
    if not sep or not module_name or not factory_name:
        raise SystemExit(f"--system-module must look like package.module:factory, got {spec!r}")
    factory = getattr(importlib.import_module(module_name), factory_name, None)
    if not callable(factory):
        raise SystemExit(f"{spec}: {factory_name!r} is not a callable in {module_name}")
    system = factory(args)
    if not isinstance(getattr(system, "name", None), str) or not callable(getattr(system, "predict", None)):
        raise SystemExit(f"{spec} must return an object with a str .name and a .predict(question, budget) method")
    return system


def _build_system(args: argparse.Namespace, evaluator):
    if args.system_module:
        return load_external_system(args.system_module, args)
    if args.system == "oracle":
        return GoldOracleSystem(evaluator)
    if args.system == "always_unsupported":
        return AlwaysUnsupportedSystem()
    if args.system == "classical":
        from text2sql_eval.systems.classical import ClassicalBaselineSystem

        return ClassicalBaselineSystem()
    if args.system == "single_agent":
        return _build_single_agent(args)
    raise ValueError(f"unknown system {args.system!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--system", choices=SYSTEMS)
    which.add_argument("--system-module", metavar="MODULE:FACTORY",
                       help="external system, e.g. text2sql_two_agent.adapter:build")
    parser.add_argument("--system-option", action="append", metavar="KEY=VALUE",
                        help="passed to the external factory as args.system_options (repeatable)")
    parser.add_argument("--benchmark", choices=sorted(MANIFEST_FOR), required=True)
    parser.add_argument("--manifest", type=Path, help="override manifest path")
    parser.add_argument("--ids", nargs="*", help="only these question IDs")
    parser.add_argument("--limit", type=int, help="only the first N included questions")
    parser.add_argument("--sample", type=int, help="seeded random sample of N questions (spread across DBs)")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--run-id", help="default: <system>_<benchmark>_<UTC timestamp>")
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sql-timeout", type=float, default=Budget.sql_timeout_s)
    parser.add_argument("--max-retries", type=int, default=Budget.max_retries)
    parser.add_argument("--keep-distinct", action="store_true",
                        help="Spider 1 only: stricter scoring that keeps DISTINCT (official default strips it)")
    parser.add_argument("--resume", action="store_true", help="skip questions already in runs.jsonl")
    parser.add_argument("--notes", default="")
    llm = parser.add_argument_group("LLM systems")
    llm.add_argument("--llm-base-url", default="http://localhost:8080/v1",
                     help="OpenAI-compatible endpoint (llama-server, Ollama, vLLM)")
    llm.add_argument("--llm-model", default="qwen2.5-coder-7b-instruct")
    llm.add_argument("--temperature", type=float, default=0.0)
    llm.add_argument("--top-p", type=float, default=1.0)
    # 512 truncated 16 Spider 2 attempts in the pilot; 1024 fits the 12k context.
    llm.add_argument("--max-new-tokens", type=int, default=1024)
    llm.add_argument("--schema-token-budget", type=int, default=6000)
    llm.add_argument("--schema-top-k", type=int, help="max tables in prompt (default: all that fit)")
    args = parser.parse_args()
    args.system_options = _parse_options(args.system_option)
    return args


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest or MANIFEST_DIR / MANIFEST_FOR[args.benchmark]
    entries = select(read_manifest(manifest_path), question_ids=args.ids, limit=args.limit,
                     sample=args.sample, sample_seed=args.sample_seed)
    if not entries:
        raise SystemExit("no included questions selected")

    options = {"keep_distinct": True} if args.keep_distinct else {}
    if options and args.benchmark != "spider1":
        raise SystemExit("--keep-distinct only applies to spider1")
    evaluator = get_evaluator(args.benchmark, **options)
    system = _build_system(args, evaluator)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{system.name}_{args.benchmark}_{stamp}"
    budget = Budget(max_retries=args.max_retries, sql_timeout_s=args.sql_timeout)
    config = RunConfig(
        run_id=run_id, system_name=system.name, benchmark=args.benchmark,
        manifest=str(manifest_path), budget=budget,
        repeat_index=args.repeat_index, seed=args.seed, notes=args.notes,
    )

    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "config": config.to_dict(),
        "selection": {"ids": args.ids, "limit": args.limit, "sample": args.sample,
                      "sample_seed": args.sample_seed, "n_questions": len(entries),
                      "question_ids": [e.question_id for e in entries]},
        "evaluator": evaluator.version,
        "system": system.describe() if hasattr(system, "describe") else {"system": system.name},
        "system_module": args.system_module,
        "system_options": args.system_options,
        "environment": collect_environment(),
    }, indent=2), encoding="utf-8")

    print(f"Run {run_id}: {len(entries)} questions, system={system.name}, evaluator={evaluator.version}")
    runs_path = run(system, entries, evaluator, config, out_dir, resume=args.resume)
    reports = write_reports(runs_path)

    for summary in summarize_by_group(load_records(runs_path)):
        print(f"\n{summary['system_name']} on {summary['benchmark']}: "
              f"EX = {summary['correct']}/{summary['evaluated']} = {summary['execution_accuracy']:.4f}")
        print(f"  within budget: EX = {summary['correct_within_budget']}/{summary['evaluated']} "
              f"({summary['n_over_budget']} over budget)")
        print(f"  executed={summary['executed']}  supported={summary['supported']}  "
              f"timeouts={summary['n_timeout']}  exec_errors={summary['n_execution_error']}  "
              f"unsafe={summary['n_unsafe_sql']}  eval_errors={summary['n_eval_error']}")
    for row in summarize_by_difficulty(load_records(runs_path)):
        print(f"  {row['difficulty']:>8}: {row['correct']}/{row['evaluated']} = {row['execution_accuracy']:.4f}")
    print(f"\nOutputs: {out_dir}")
    for name, path in reports.items():
        print(f"  {name}: {path.name}")


if __name__ == "__main__":
    main()
