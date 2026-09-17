#!/usr/bin/env python3
"""Run the M4 eval harness fully offline, on in-repo fakes.

Needs no Qdrant, no OPENAI_API_KEY, and none of the S8 config/CLI patches applied — it
depends only on the ``research_navigator.eval`` package and the packaged golden set. It
exists so the harness (and the shape of its JSON + Markdown reports) can be exercised and
reviewed immediately.

    PYTHONPATH=src python scripts/eval_dry_run.py --out eval/

The live equivalent, once the patches are applied, is ``make eval`` / ``rn eval``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research_navigator.eval import EvalSettings, load_golden_set, run_eval
from research_navigator.eval.factory import build_dry_run_builder
from research_navigator.eval.golden import default_golden_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=None, help="Golden-set JSON path.")
    parser.add_argument("--out", type=Path, default=Path("eval"), help="Report output dir.")
    parser.add_argument("--now-year", type=int, default=2026, help="'Current' year.")
    args = parser.parse_args()

    golden = load_golden_set(args.golden or default_golden_path())
    settings = EvalSettings(now_year=args.now_year, report_dir=args.out, judge_enabled=False)
    report = run_eval(builder=build_dry_run_builder(golden), golden=golden, settings=settings)
    for path in report.write(settings):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
