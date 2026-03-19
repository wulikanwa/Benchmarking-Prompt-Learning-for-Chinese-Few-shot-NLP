#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    args = parser.parse_args()

    rows = []
    for path in Path(args.root).rglob('metrics.json'):
        with open(path, 'r', encoding='utf-8') as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit('No metrics.json files found.')
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print(df.groupby(['task', 'method']).mean(numeric_only=True))


if __name__ == '__main__':
    main()
