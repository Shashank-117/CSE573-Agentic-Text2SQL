"""Placeholder BM25 schema retrieval until Jay's search_docs lands.

Ranks a database's table documents against the question and selects tables
under a token budget. The same selection function must be used by every
LLM system in a matched comparison (see docs/CONTRACTS.md, section 4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from rank_bm25 import BM25Okapi

from .schema_docs import TableDoc

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return _TOKEN.findall(spaced.lower())


def rank_tables(question: str, docs: Sequence[TableDoc]) -> list[tuple[TableDoc, float]]:
    """BM25 scores, highest first; ties keep schema order."""
    if not docs:
        return []
    corpus = [tokenize(d.index_text) or ["_empty_"] for d in docs]
    scores = BM25Okapi(corpus).get_scores(tokenize(question))
    order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
    return [(docs[i], float(scores[i])) for i in order]


@dataclass(frozen=True)
class SchemaSelection:
    tables: tuple[TableDoc, ...]  # in original schema order
    ranked_ids: tuple[str, ...]   # full BM25 ranking (for retrieval analysis)
    complete: bool                # True if every table fit
    estimated_tokens: int


def select_tables(
    question: str,
    docs: Sequence[TableDoc],
    token_budget: int,
    estimate_tokens: Callable[[str], int],
    top_k: int | None = None,
) -> SchemaSelection:
    """Pick tables under the token budget; ``top_k`` is a hard cap on count.

    Priority: (1) tables with a positive BM25 score, best first;
    (2) foreign-key targets of those, so join partners are visible;
    (3) remaining tables in BM25 order as filler.
    """
    ranked = rank_tables(question, docs)
    limit = len(docs) if top_k is None else max(0, top_k)
    chosen: dict[str, TableDoc] = {}
    used = 0

    def try_add(doc: TableDoc) -> None:
        nonlocal used
        cost = estimate_tokens(doc.prompt_text)
        if doc.doc_id not in chosen and len(chosen) < limit and used + cost <= token_budget:
            chosen[doc.doc_id] = doc
            used += cost

    by_name = {d.table: d for d in docs}
    for doc, score in ranked:
        if score > 0:
            try_add(doc)
    for doc in list(chosen.values()):
        for ref in doc.references:
            if ref in by_name:
                try_add(by_name[ref])
    for doc, _ in ranked:
        try_add(doc)

    ordered = tuple(d for d in docs if d.doc_id in chosen)
    return SchemaSelection(
        tables=ordered,
        ranked_ids=tuple(d.doc_id for d, _ in ranked),
        complete=len(ordered) == len(docs),
        estimated_tokens=used,
    )
