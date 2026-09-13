"""M0 onboarding: skim the corpus and print structural observations.

Run: uv run python scripts/explore_corpus.py
Writes nothing; prints a summary you distil into docs/OBSERVATIONS.md.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

MANIFEST = Path("corpus/manifest.json")


def main() -> None:
    if not MANIFEST.exists():
        print(f"[!] {MANIFEST} not found. Place the sealed corpus, then re-run.")
        return

    docs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if isinstance(docs, dict):  # tolerate {"documents": [...]} shape
        docs = docs.get("documents", [])

    print(f"Total documents: {len(docs)}\n")

    by_type = Counter(d.get("content_type", "?") for d in docs)
    by_year = Counter(d.get("year", "?") for d in docs)
    tag_counts: Counter[str] = Counter()
    for d in docs:
        tag_counts.update(d.get("tags", []))

    print("By content_type:")
    for k, v in by_type.most_common():
        print(f"  {k:20s} {v}")
    print("\nBy year:")
    for k, v in sorted(by_year.items(), key=lambda kv: str(kv[0])):
        print(f"  {k}: {v}")
    print("\nTop tags:")
    for k, v in tag_counts.most_common(15):
        print(f"  {k:20s} {v}")

    print("\nFile size sample (first existing local_path per type):")
    seen: set[str] = set()
    for d in docs:
        ct = d.get("content_type", "?")
        if ct in seen:
            continue
        p = Path(str(d.get("local_path", "")))
        if p.exists():
            print(f"  {ct:20s} {p.name:30s} {p.stat().st_size / 1024:.1f} KB")
            seen.add(ct)


if __name__ == "__main__":
    main()
