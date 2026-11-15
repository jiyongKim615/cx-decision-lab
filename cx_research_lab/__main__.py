"""python3 -m cx_research_lab --project R1 --alpha .2"""
import argparse
import json

from .common import ContractError, persist
from .demo import run


def main():
    parser = argparse.ArgumentParser(description="Synthetic CX research lab; no credentials or external actions")
    parser.add_argument("--project", choices=("all", "R1", "R2", "R3"), default="all")
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--review-capacity", type=int, default=3)
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--drop-outcome", action="store_true")
    parser.add_argument("--output-dir", help="Optional immutable, content-addressed JSON directory")
    args = parser.parse_args()
    try:
        all_results = run(alpha=args.alpha, manual_only=args.manual_only, review_capacity=args.review_capacity, drop_outcome=args.drop_outcome)
        result = all_results if args.project == "all" else all_results[args.project]
        if args.output_dir:
            artifact = persist(args.output_dir, result)
            print(json.dumps({"artifact": artifact, "result": result}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except ContractError as error:
        parser.exit(2, f"CONTRACT_ERROR: {error}\n")


if __name__ == "__main__":
    main()
