"""Finite wafer images and measurements; fixed pixel tokens, chronological lot-disjoint root-cause evaluation."""

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
    source = args.source or str(output / 'generated/events.json')
    steps = []
    if not args.source:
        steps.append(['finite_data.py', '--output', str(output / 'generated'), '--seed', '11'])
    return steps + [
        ["-m", "fabroot", "prepare", "--image-encoder", "finite_texture", "--source", source, "--output", str(output / "data")],
        ["-m", "fabroot", "train", "--data", str(output / "data/dataset.json"), "--variant", "full", "--epochs", "50", "--seed", "11", "--output", str(output / "full")],
        ["-m", "fabroot", "train", "--data", str(output / "data/dataset.json"), "--variant", "early_concat", "--epochs", "50", "--seed", "11", "--output", str(output / "early_concat")],
        ["-m", "fabroot", "evaluate", "--data", str(output / "data/dataset.json"), "--model", str(output / "full/model.pt"), "--actions", "actions.json", "--output", str(output / "full_eval")],
        ["-m", "fabroot", "evaluate", "--data", str(output / "data/dataset.json"), "--model", str(output / "early_concat/model.pt"), "--actions", "actions.json", "--output", str(output / "early_concat_eval")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/finite-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--source', type=absolute_path, help='Optional prepared input; otherwise generate the finite setting')
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
