"""Precomputed 768-dimensional image features plus aligned text/metrology/wafer facts; lot/time-disjoint CPU training50 epochs seed11 and train-only action likelihoods."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def absolute_path(value):
    return str(Path(value).resolve())


def build_steps(args):
    output = (ROOT / args.output).resolve()
    return [
        ["-m", "fabroot", "prepare", "--source", args.source, "--output", str(output / "data")],
        ["-m", "fabroot", "train", "--data", str(output / "data/dataset.json"), "--variant", "full", "--epochs", "50", "--seed", "11", "--output", str(output / "full")],
        ["-m", "fabroot", "train", "--data", str(output / "data/dataset.json"), "--variant", "early_concat", "--epochs", "50", "--seed", "11", "--output", str(output / "early_concat")],
        ["-m", "fabroot", "evaluate", "--data", str(output / "data/dataset.json"), "--model", str(output / "full/model.pt"), "--actions", "actions.json", "--output", str(output / "full_eval")],
        ["-m", "fabroot", "evaluate", "--data", str(output / "data/dataset.json"), "--model", str(output / "early_concat/model.pt"), "--actions", "actions.json", "--output", str(output / "early_concat_eval")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--source', required=True, type=absolute_path)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
