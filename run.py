"""Unified CLI for the main reproducible stages."""
from __future__ import annotations
import argparse
import subprocess
import sys

STAGES = {
    "prepare": "experiments.prepare_data",
    "original": "experiments.train_original",
    "retrained": "experiments.train_retrained",
    "unlearn": "experiments.run_combined_unlearning",
    "benchmark": "evaluation.benchmark",
    "mia": "experiments.run_mia",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[*STAGES, "all"])
    args = parser.parse_args()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n=== {stage} ===")
        subprocess.run([sys.executable, "-m", STAGES[stage]], check=True)


if __name__ == "__main__":
    main()
