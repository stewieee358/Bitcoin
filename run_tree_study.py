"""Run the MH6807 tree/feature-selection extension from the Bitcoin repository."""
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from src.tree_study import run_tree_study

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/config_tree_study_2h_2025.json')
    parser.add_argument('--output-dir', type=Path, help='default: separate automatic tree-study directory')
    parser.add_argument('--evaluation-end', help='exclusive UTC end; shorten the experiment for a smoke run')
    parser.add_argument('--force', action='store_true', help='recompute this study despite verified caches')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.evaluation_end:
        config['evaluation_end'] = args.evaluation_end
    with threadpool_limits(limits=1):
        out = run_tree_study(config, ROOT, args.output_dir, args.force)
    print(f'Completed tree study: {out}', flush=True)


if __name__ == '__main__':
    main()
