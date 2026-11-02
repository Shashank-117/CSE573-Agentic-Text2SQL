import re


def sql_to_intent(sql: str) -> str:
    """Restricted baseline labels from the proposal: projection/filter/count/sum."""
    upper = sql.upper()
    if not upper.lstrip().startswith("SELECT"):
        return "UNSUPPORTED"
    if re.search(r"\bCOUNT\s*\(", upper):
        return "COUNT_FILTER" if " WHERE " in f" {upper} " else "COUNT"
    if re.search(r"\bSUM\s*\(", upper):
        return "SUM_FILTER" if " WHERE " in f" {upper} " else "SUM"
    if " WHERE " in f" {upper} ":
        return "FILTER"
    return "SELECT"
