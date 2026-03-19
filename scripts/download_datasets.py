#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path

from datasets import load_dataset

from src.benchmark.utils import ensure_dir, write_jsonl


def dump_split(hf_ds, fields, out_path):
    rows = []
    for ex in hf_ds:
        row = {k: ex.get(v, ex.get(k)) for k, v in fields.items()}
        rows.append(row)
    write_jsonl(rows, out_path)


def try_chnsenticorp(data_dir: Path):
    out = ensure_dir(data_dir / 'sentiment')
    candidates = ['seamew/ChnSentiCorp', 'lansinuote/ChnSentiCorp']
    last_err = None
    for name in candidates:
        try:
            ds = load_dataset(name)
            train = ds['train']
            test = ds['test'] if 'test' in ds else ds['validation']
            valid = ds['validation'] if 'validation' in ds else test.select(range(min(500, len(test))))
            fields = {'text': 'text', 'label': 'label'}
            dump_split(train, fields, out / 'train.jsonl')
            dump_split(valid, fields, out / 'dev.jsonl')
            dump_split(test, fields, out / 'test.jsonl')
            print(f'[OK] ChnSentiCorp from {name}')
            return
        except Exception as e:
            last_err = e
    print(f'[WARN] Could not auto-download ChnSentiCorp: {last_err}')
    print('Manual fallback: place train/dev/test JSONL under data/sentiment/. Fields: text, label')


def try_cmrc2018(data_dir: Path):
    out = ensure_dir(data_dir / 'qa')
    ds = load_dataset('hfl/cmrc2018')

    def flatten(split):
        rows = []
        for ex in split:
            answers = ex['answers']['text'] if isinstance(ex.get('answers'), dict) else ex.get('answers', [])
            answer = answers[0] if answers else ''
            rows.append({'context': ex['context'], 'question': ex['question'], 'answer': answer})
        return rows

    write_jsonl(flatten(ds['train']), out / 'train.jsonl')
    write_jsonl(flatten(ds['validation']), out / 'dev.jsonl')
    write_jsonl(flatten(ds['validation']), out / 'test.jsonl')
    print('[OK] CMRC2018 from hfl/cmrc2018')


def try_lcsts(data_dir: Path):
    out = ensure_dir(data_dir / 'summarization')
    try:
        ds = load_dataset('hugcyp/LCSTS')
        train_name = 'train' if 'train' in ds else list(ds.keys())[0]
        val_name = 'validation' if 'validation' in ds else train_name
        test_name = 'test' if 'test' in ds else val_name

        def normalize(split):
            rows = []
            for ex in split:
                source = ex.get('text') or ex.get('content') or ex.get('source')
                summary = ex.get('summary') or ex.get('target')
                if source is None or summary is None:
                    continue
                rows.append({'source': source, 'summary': summary})
            return rows

        write_jsonl(normalize(ds[train_name]), out / 'train.jsonl')
        write_jsonl(normalize(ds[val_name]), out / 'dev.jsonl')
        write_jsonl(normalize(ds[test_name]), out / 'test.jsonl')
        print('[OK] LCSTS from hugcyp/LCSTS')
    except Exception as e:
        print(f'[WARN] Could not auto-download LCSTS: {e}')
        print('Manual fallback: place train/dev/test JSONL under data/summarization/. Fields: source, summary')


def try_translation(data_dir: Path):
    out = ensure_dir(data_dir / 'translation')
    ds = load_dataset('wmt/wmt19', 'zh-en')

    def normalize(split):
        rows = []
        for ex in split:
            tr = ex['translation']
            rows.append({'source': tr['zh'], 'target': tr['en']})
        return rows

    write_jsonl(normalize(ds['train']), out / 'train.jsonl')
    val = ds['validation'] if 'validation' in ds else ds['train'].select(range(2000))
    test = ds['test'] if 'test' in ds else val
    write_jsonl(normalize(val), out / 'dev.jsonl')
    write_jsonl(normalize(test), out / 'test.jsonl')
    print('[OK] zh-en translation from wmt/wmt19')


def manual_only_thucnews(data_dir: Path):
    out = ensure_dir(data_dir / 'topic')
    print('[INFO] THUCNews usually requires manual download/preparation.')
    print('Place train/dev/test JSONL under data/topic/ with fields: text, label')
    print('Expected 10 labels: 财经, 房产, 教育, 科技, 军事, 汽车, 体育, 游戏, 娱乐, 时尚')
    print(f'[PATH] {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()
    data_dir = ensure_dir(args.data_dir)
    try_chnsenticorp(data_dir)
    manual_only_thucnews(data_dir)
    try_cmrc2018(data_dir)
    try_lcsts(data_dir)
    try_translation(data_dir)


if __name__ == '__main__':
    main()
