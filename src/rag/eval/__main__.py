from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def run_eval(dataset_path: str, report_path: str, baseline_path: str | None) -> None:
    from rag.eval.dataset import GoldenDataset
    from rag.eval.runner import EvalRunner

    dataset = GoldenDataset.from_jsonl(dataset_path)
    runner = EvalRunner()
    report = await runner.run(dataset)
    report.save(report_path)

    print(f"Evaluation complete: {report_path}")
    for k, v in report.aggregate.items():
        print(f"  {k}: {v:.4f}")

    if baseline_path:
        import pathlib
        baseline_text = pathlib.Path(baseline_path).read_text()
        baseline = json.loads(baseline_text)
        baseline_agg = baseline.get("aggregate", {})
        print("\nComparison vs baseline:")
        for k, v in report.aggregate.items():
            diff = v - baseline_agg.get(k, 0.0)
            marker = "+" if diff >= 0 else ""
            print(f"  {k}: {v:.4f} ({marker}{diff:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag.eval")
    sub = parser.add_subparsers(dest="command")

    run_cmd = sub.add_parser("run", help="Run evaluation")
    run_cmd.add_argument("--dataset", required=True)
    run_cmd.add_argument("--report", required=True)
    run_cmd.add_argument("--baseline", default=None)

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run_eval(args.dataset, args.report, args.baseline))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
