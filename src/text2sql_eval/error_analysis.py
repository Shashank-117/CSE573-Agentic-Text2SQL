"""Post-hoc error analysis for Spider 1 runs (uses gold SQL; never run inside a system).

Each wrong prediction is compared with its gold query using the official Spider
component matcher (``evaluation.Evaluator.eval_partial_match`` from
test-suite-sql-eval). Components that differ give the error categories; the
first one in ``CATEGORY_ORDER`` is the primary category.

The component matcher ignores literal values, so "components match" means the
query has the right shape but a different value, DISTINCT, join condition or
an arguable gold answer. Those need manual review.
"""

from __future__ import annotations

import copy
import importlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .official.spider1 import Spider1Evaluator, _load_hardness

EXECUTION_FAILURE = "execution failure"
UNPARSEABLE = "outside Spider grammar (EXISTS, correlated subquery, CASE, ||): review manually"
EXTRA_TABLES = "extra table joined"
MISSING_TABLES = "missing table / join"
WRONG_TABLES = "different tables (schema linking)"
LEFT_JOIN = "LEFT JOIN keeps unmatched rows (gold uses inner join)"
SET_OPS = "set operation / nesting"
AGGREGATION = "aggregation in SELECT"
EXTRA_COLUMNS = "extra columns in SELECT"
SELECT_COLUMNS = "wrong or missing SELECT columns"
WHERE = "WHERE condition"
GROUP = "GROUP BY / HAVING"
ORDER = "ORDER BY / LIMIT"
SAME_SHAPE = "same shape (values, DISTINCT, join keys or gold issue)"

CATEGORY_ORDER = (
    EXECUTION_FAILURE, UNPARSEABLE, WRONG_TABLES, MISSING_TABLES, EXTRA_TABLES, LEFT_JOIN, SET_OPS, AGGREGATION,
    SELECT_COLUMNS, EXTRA_COLUMNS, WHERE, GROUP, ORDER, SAME_SHAPE,
)
# Coarser groups for the report.
CATEGORY_GROUP = {
    EXECUTION_FAILURE: "SQL failed to run",
    UNPARSEABLE: "needs manual review",
    WRONG_TABLES: "schema linking / join path", MISSING_TABLES: "schema linking / join path",
    EXTRA_TABLES: "schema linking / join path", LEFT_JOIN: "join semantics",
    AGGREGATION: "output columns", SELECT_COLUMNS: "output columns", EXTRA_COLUMNS: "output columns",
    SET_OPS: "conditions / grouping / ordering", WHERE: "conditions / grouping / ordering",
    GROUP: "conditions / grouping / ordering", ORDER: "conditions / grouping / ordering",
    SAME_SHAPE: "values or gold answer",
}
_FAILED_STATUSES = frozenset({"execution_error", "timeout", "view_error", "unsafe_sql", "parse_error",
                              "unsupported", "compile_error", "system_error"})


_CLAUSE_WORDS = ("AS", "ON", "WHERE", "JOIN", "LEFT", "INNER", "CROSS", "GROUP", "ORDER", "LIMIT", "HAVING",
                 "UNION", "EXCEPT", "INTERSECT", "USING", "NATURAL", "OUTER")
_JOIN_KIND = re.compile(r"\b(?:LEFT\s+OUTER|LEFT|INNER|CROSS)\s+JOIN\b", re.IGNORECASE)
_TABLE_ALIAS = re.compile(
    r"\b(FROM|JOIN)\s+(\w+)\s+(?!(?:" + "|".join(_CLAUSE_WORDS) + r")\b)([A-Za-z_]\w*)\b", re.IGNORECASE)
_ALIASES = re.compile(r"(\b(?:FROM|JOIN)\s+\w+\s+AS\s+\w+)|(\s+AS\s+(?:\w+|\"[^\"]+\"|'[^']+'))", re.IGNORECASE)
_QUOTED_WORD = re.compile(r'(=|<>|!=|<|>|\bLIKE|\bIN\s*\(|,)?(\s*)"(\w+)"', re.IGNORECASE)


def _unquote_identifiers(text: str) -> str:
    """Drop double quotes around identifiers but keep them on values ("Bob" after =, LIKE, IN)."""
    def repl(match: re.Match) -> str:
        operator, space, word = match.groups()
        if operator and operator != ",":
            return match.group(0)
        return f"{operator or ''}{space}{word}"
    return _QUOTED_WORD.sub(repl, text)
_FUNC_ALIAS = re.compile(r"(\b\w+\s*\([^()]*\))\s+AS\s+(\w+)", re.IGNORECASE)
_COUNT_COLUMN = re.compile(r"\bCOUNT\s*\(\s*(?!DISTINCT\b)[\w.]+\s*\)", re.IGNORECASE)
_LEFT_JOIN = re.compile(r"\bLEFT\s+(?:OUTER\s+)?JOIN\b", re.IGNORECASE)


def _inline_function_aliases(text: str) -> str:
    """Replace later uses of `COUNT(*) AS n` style aliases (ORDER BY n, HAVING n) with the expression."""
    for expr, alias in _FUNC_ALIAS.findall(text):
        head, sep, tail = text.partition(f"{expr} AS {alias}") if f"{expr} AS {alias}" in text else (text, "", "")
        if sep:
            text = head + sep + re.sub(rf"\b{re.escape(alias)}\b", expr, tail)
    return text


def normalize_for_spider_parser(sql: str) -> str:
    """Rewrite modern SQL into the subset the 2018 Spider parser accepts (analysis only, never scoring).

    Strips a trailing semicolon, double quotes around identifiers and SELECT
    column aliases; adds AS to bare table aliases; turns LEFT/INNER/CROSS JOIN
    into JOIN (LEFT JOIN is reported separately); treats COUNT(col) as COUNT(*)
    so the matcher does not report a column error for the count argument.
    """
    text = sql.strip().rstrip(";").strip()
    text = _COUNT_COLUMN.sub("COUNT(*)", text)
    text = _unquote_identifiers(text)
    text = _JOIN_KIND.sub("JOIN", text)
    text = _inline_function_aliases(text)
    text = _TABLE_ALIAS.sub(r"\1 \2 AS \3", text)
    return _ALIASES.sub(lambda m: m.group(1) or "", text)


@dataclass(frozen=True)
class ErrorRecord:
    question_id: str
    db_id: str
    difficulty: str
    status: str
    primary: str
    categories: tuple[str, ...]
    predicted_sql: str
    gold_sql: str
    detail: str = ""
    extra: dict = field(default_factory=dict)


class _SpiderParser:
    """Official parser + value/column rebuild, as in evaluation.evaluate()."""

    def __init__(self, evaluator: Spider1Evaluator):
        self._classifier, self._process_sql = _load_hardness(evaluator.evaluator_repo)
        self._evaluation = importlib.import_module("evaluation")
        tables = evaluator.spider_root / "tables.json"
        self._kmaps = self._evaluation.build_foreign_key_map_from_json(str(tables))
        self._schemas: dict[str, object] = {}

    def parse(self, db_id: str, db_path: str, sql: str) -> dict:
        ev = self._evaluation
        if db_path not in self._schemas:
            self._schemas[db_path] = self._process_sql.Schema(self._process_sql.get_schema(db_path))
        schema = self._schemas[db_path]
        parsed = self._process_sql.get_sql(schema, sql)
        valid = ev.build_valid_col_units(parsed["from"]["table_units"], schema)
        parsed = ev.rebuild_sql_val(parsed)
        return ev.rebuild_sql_col(valid, parsed, self._kmaps[db_id])

    def components(self, pred: dict, gold: dict) -> dict[str, dict]:
        """Official partial-match scores per component, plus raw SELECT match counts.

        get_scores() is all-or-nothing, so recall is recomputed from eval_sel()
        (on copies: it mutates its input) to tell "extra columns" apart.
        """
        scores = self._classifier.eval_partial_match(copy.deepcopy(pred), copy.deepcopy(gold))
        label_total, pred_total, matched, _ = self._evaluation.eval_sel(copy.deepcopy(pred), copy.deepcopy(gold))
        scores["select_counts"] = {"label_total": label_total, "pred_total": pred_total, "matched": matched}
        return scores


def _tables(parsed: dict) -> list:
    return sorted(str(unit) for unit in parsed["from"]["table_units"])


def _f1(components: dict[str, dict], name: str) -> float:
    return components.get(name, {}).get("f1", 1)


def _table_category(pred_tables: list, gold_tables: list) -> str | None:
    pred, gold = set(pred_tables), set(gold_tables)
    if pred == gold:
        return None
    if gold < pred:
        return EXTRA_TABLES
    if pred < gold:
        return MISSING_TABLES
    return WRONG_TABLES


def _select_category(components: dict[str, dict]) -> str | None:
    select = components.get("select", {})
    if select.get("f1", 1) == 1:
        return None
    if _f1(components, "select(no AGG)") == 1:
        return AGGREGATION
    counts = components.get("select_counts", {})
    # every gold column is predicted but the prediction has more
    if counts and counts["matched"] == counts["label_total"] < counts["pred_total"]:
        return EXTRA_COLUMNS
    return SELECT_COLUMNS


def categorize(components: dict[str, dict], pred_tables: list, gold_tables: list) -> list[str]:
    """Error categories (ordered by CATEGORY_ORDER) from official component scores and table sets."""
    found = [c for c in (_table_category(pred_tables, gold_tables),) if c]
    if _f1(components, "IUEN") != 1:
        found.append(SET_OPS)
    if select := _select_category(components):
        found.append(select)
    if _f1(components, "where") != 1:
        found.append(WHERE)
    if _f1(components, "group") != 1 or _f1(components, "group(no Having)") != 1:
        found.append(GROUP)
    if _f1(components, "order") != 1:
        found.append(ORDER)
    return sorted(found, key=CATEGORY_ORDER.index) or [SAME_SHAPE]


def analyze_record(record: dict, gold_sql: str, db_path: str, parser: _SpiderParser) -> ErrorRecord:
    base = dict(question_id=record["question_id"], db_id=record["db_id"],
                difficulty=record.get("difficulty") or "unknown", status=record["status"],
                predicted_sql=(record.get("generated_sql") or "").strip(), gold_sql=gold_sql.strip())
    if record["status"] in _FAILED_STATUSES:
        reason = record.get("execution_error") or record.get("error_message") or record["status"]
        return ErrorRecord(**base, primary=EXECUTION_FAILURE, categories=(EXECUTION_FAILURE,), detail=str(reason))
    try:
        gold = parser.parse(record["db_id"], db_path, gold_sql)
    except Exception as exc:  # the official parser rejects a few gold queries
        return ErrorRecord(**base, primary=UNPARSEABLE, categories=(UNPARSEABLE,), detail=f"gold: {exc!r}")
    try:
        pred = parser.parse(record["db_id"], db_path, normalize_for_spider_parser(base["predicted_sql"]))
    except Exception as exc:
        return ErrorRecord(**base, primary=UNPARSEABLE, categories=(UNPARSEABLE,), detail=f"pred: {exc!r}")
    found = categorize(parser.components(pred, gold), _tables(pred), _tables(gold))
    if _LEFT_JOIN.search(base["predicted_sql"]) and not _LEFT_JOIN.search(gold_sql):
        found = sorted({*found, LEFT_JOIN} - {SAME_SHAPE}, key=CATEGORY_ORDER.index)
    return ErrorRecord(**base, primary=found[0], categories=tuple(found))


def analyze_run(records: list[dict], db_paths: dict[str, str], evaluator: Spider1Evaluator) -> list[ErrorRecord]:
    """Categorize every wrong Spider 1 prediction in a run."""
    parser = _SpiderParser(evaluator)
    errors = []
    for record in records:
        if record.get("correct"):
            continue
        gold = evaluator.gold_sql(record["question_id"])
        if gold is None:
            continue
        errors.append(analyze_record(record, gold, str(Path(db_paths[record["question_id"]])), parser))
    return errors
