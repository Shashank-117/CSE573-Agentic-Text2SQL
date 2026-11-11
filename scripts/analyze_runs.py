"""Analyze finished runs in outputs/runs/.

Examples:
    # Mean ± std over 3 repeats (one row per system x benchmark)
    python scripts/analyze_runs.py aggregate outputs/runs/sa_rep0 outputs/runs/sa_rep1 outputs/runs/sa_rep2

    # Paired comparison of two systems on the same questions
    python scripts/analyze_runs.py compare outputs/runs/two_agent_rep0 outputs/runs/sa_rep0 --within-budget

    # Rewrite a run's reports (adds difficulty labels to older Spider 1 runs)
    python scripts/analyze_runs.py report outputs/runs/pilot_single_agent_spider1_s50 --add-difficulty
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from text2sql_eval.analysis import PAIRED_FIELDS, RunMismatchError, aggregate_repeats, compare_runs
from text2sql_eval.manifest import read_manifest
from text2sql_eval.metrics import load_records, summarize_by_difficulty, write_reports
from text2sql_eval.official import get_evaluator
from text2sql_eval.paths import MANIFEST_DIR, PROJECT_ROOT


def _runs_path(run_dir: Path) -> Path:
    path = Path(run_dir) / "runs.jsonl"
    if not path.is_file():
        raise SystemExit(f"no runs.jsonl in {run_dir}")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(args: argparse.Namespace) -> None:
    runs = {}
    for run_dir in args.run_dirs:
        records = load_records(_runs_path(run_dir))
        runs[records[0]["run_id"] if records else str(run_dir)] = records
    rows = aggregate_repeats(runs)
    for row in rows:
        print(f"{row['system_name']} on {row['benchmark']} [{row['evaluator']}] ({row['n_repeats']} repeats, "
              f"{row['n_questions']} questions): EX = {row['execution_accuracy_mean']:.4f} "
              f"± {row['execution_accuracy_std']:.4f} (per repeat: {row['correct_per_repeat']})")
    if args.out:
        _write_csv(args.out, rows, list(rows[0]) if rows else [])
        print(f"wrote {args.out}")


def _compare(args: argparse.Namespace) -> None:
    a = load_records(_runs_path(args.run_a))
    b = load_records(_runs_path(args.run_b))
    result = compare_runs(a, b, within_budget_only=args.within_budget,
                          allow_different_evaluators=args.allow_different_evaluators,
                          bootstrap_iterations=args.iterations, seed=args.seed)
    s = result.summary
    print(f"A = {s['system_a']} ({s['run_a']}): EX {s['ex_a']:.4f}")
    print(f"B = {s['system_b']} ({s['run_b']}): EX {s['ex_b']:.4f}")
    print(f"A - B = {s['ex_diff_a_minus_b']:+.4f}  95% CI [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]")
    print(f"both {s['n_both']}, only A {s['n_only_a']}, only B {s['n_only_b']}, neither {s['n_neither']}; "
          f"McNemar exact p = {s['mcnemar_exact_p']:.4g}")
    out_dir = args.out or Path(args.run_a) / f"compare_vs_{Path(args.run_b).name}"
    _write_csv(out_dir / "paired.csv", result.paired_rows, PAIRED_FIELDS)
    (out_dir / "comparison.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")


def _report(args: argparse.Namespace) -> None:
    runs_path = _runs_path(args.run_dir)
    if args.add_difficulty:
        records = load_records(runs_path)
        if {r["benchmark"] for r in records} != {"spider1"}:
            raise SystemExit("--add-difficulty only applies to Spider 1 runs")
        evaluator = get_evaluator("spider1")
        db_paths = {e.question_id: str(PROJECT_ROOT / e.db_path)
                    for e in read_manifest(MANIFEST_DIR / "spider1_dev.jsonl") if e.db_path}
        labelled = [{**r, "difficulty": evaluator.difficulty(r["question_id"], db_paths[r["question_id"]])}
                    for r in records]
        runs_path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in labelled),
                             encoding="utf-8")
    reports = write_reports(runs_path)
    for row in summarize_by_difficulty(load_records(runs_path)):
        print(f"  {row['difficulty']:>8}: {row['correct']}/{row['evaluated']} = {row['execution_accuracy']:.4f}")
    for name, path in reports.items():
        print(f"{name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    agg = sub.add_parser("aggregate", help="mean/std of repeated runs")
    agg.add_argument("run_dirs", nargs="+", type=Path)
    agg.add_argument("--out", type=Path, help="CSV path for the aggregate table")
    agg.set_defaults(func=_aggregate)

    cmp_ = sub.add_parser("compare", help="paired comparison of two runs")
    cmp_.add_argument("run_a", type=Path)
    cmp_.add_argument("run_b", type=Path)
    cmp_.add_argument("--within-budget", action="store_true",
                      help="count correct-but-over-budget answers as wrong (matched-budget comparison)")
    cmp_.add_argument("--allow-different-evaluators", action="store_true",
                      help="compare runs scored differently (e.g. default vs keep_distinct)")
    cmp_.add_argument("--iterations", type=int, default=10_000)
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.add_argument("--out", type=Path, help="output directory (default: inside run_a)")
    cmp_.set_defaults(func=_compare)

    rep = sub.add_parser("report", help="rewrite a run's summary CSVs")
    rep.add_argument("run_dir", type=Path)
    rep.add_argument("--add-difficulty", action="store_true", help="label Spider 1 records with official difficulty")
    rep.set_defaults(func=_report)

    args = parser.parse_args()
    try:
        args.func(args)
    except RunMismatchError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
