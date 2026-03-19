#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import random
from collections import defaultdict
from pathlib import Path

from src.benchmark.utils import ensure_dir, read_jsonl, write_jsonl


def sample_per_class(rows, label_key: str, k: int, seed: int):
    groups = defaultdict(list)
    for r in rows:
        groups[r[label_key]].append(r)
    rng = random.Random(seed)
    out = []
    for label, items in groups.items():
        rng.shuffle(items)
        out.extend(items[:k])
    rng.shuffle(out)
    return out


def sample_total(rows, total_k: int, seed: int):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:total_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='data/fewshot')
    parser.add_argument('--seeds', nargs='+', type=int, default=[13, 21, 42, 87, 100])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)

    # classification
    for task in ['sentiment', 'topic']:
        train_path = data_dir / task / 'train.jsonl'
        if not train_path.exists():
            continue
        rows = read_jsonl(train_path)
        for seed in args.seeds:
            task_seed_dir = ensure_dir(out_dir / task / f'seed_{seed}')
            for k in [16, 32, 64]:
                sampled = sample_per_class(rows, 'label', k, seed)
                write_jsonl(sampled, task_seed_dir / f'train_{k}.jsonl')

    # qa + generation tasks
    for task, ks in [('qa', [128, 256, 512]), ('summarization', [256, 512, 1024]), ('translation', [256, 512, 1024])]:
        train_path = data_dir / task / 'train.jsonl'
        if not train_path.exists():
            continue
        rows = read_jsonl(train_path)
        for seed in args.seeds:
            task_seed_dir = ensure_dir(out_dir / task / f'seed_{seed}')
            for k in ks:
                sampled = sample_total(rows, k, seed)
                write_jsonl(sampled, task_seed_dir / f'train_{k}.jsonl')


if __name__ == '__main__':
    main()
