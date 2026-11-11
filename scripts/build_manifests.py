"""Build question manifests (with explicit exclusion reasons).

Usage:
    python scripts/build_manifests.py
"""

from collections import Counter

from text2sql_eval.manifest import build_spider1_manifest, build_spider2_lite_manifest, write_manifest
from text2sql_eval.paths import MANIFEST_DIR


def _report(name: str, entries: list) -> None:
    included = sum(e.included for e in entries)
    print(f"{name}: {len(entries)} questions, {included} included, {len(entries) - included} excluded")
    for reason, count in Counter(e.exclusion_reason for e in entries if not e.included).most_common():
        print(f"    excluded {count:4d}  {reason}")


def main() -> None:
    manifests = {
        "spider1_dev": build_spider1_manifest("dev"),
        "spider2_lite": build_spider2_lite_manifest(),
    }
    for name, entries in manifests.items():
        path = write_manifest(entries, MANIFEST_DIR / f"{name}.jsonl")
        _report(name, entries)
        print(f"    -> {path}")


if __name__ == "__main__":
    main()
