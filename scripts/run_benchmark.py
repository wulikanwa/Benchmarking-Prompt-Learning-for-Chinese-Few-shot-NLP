#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path

from src.benchmark.data import build_tokenized_classification, build_tokenized_generation, build_tokenized_qa
from src.benchmark.models import build_classification_model, build_qa_model, build_seq2seq_model
from src.benchmark.trainer import train_classification, train_generation, train_qa
from src.benchmark.utils import read_jsonl, save_metrics, set_seed


def infer_num_labels(task: str) -> int:
    return 2 if task == 'sentiment' else 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['sentiment', 'topic', 'qa', 'summarization', 'translation'])
    parser.add_argument('--method', required=True, choices=['full_ft', 'manual_prompt', 'ptuning_v2', 'lora'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--val_file', required=True)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.task in ['sentiment', 'topic']:
        bundle = build_tokenized_classification(args.task, args.model, args.train_file, args.val_file, args.test_file)
        model = build_classification_model(args.model, infer_num_labels(args.task), args.method)
        out = train_classification(model, args.model, bundle, args.output_dir)

    elif args.task == 'qa':
        bundle = build_tokenized_qa(args.model, args.train_file, args.val_file, args.test_file)
        labels = [x['answer'] for x in read_jsonl(args.test_file)]
        model = build_qa_model(args.model, args.method)
        out = train_qa(model, args.model, bundle, labels, args.output_dir)

    else:
        bundle = build_tokenized_generation(args.task, args.model, args.train_file, args.val_file, args.test_file)
        key = 'summary' if args.task == 'summarization' else 'target'
        labels = [x[key] for x in read_jsonl(args.test_file)]
        model = build_seq2seq_model(args.model, args.method)
        out = train_generation(args.task, model, args.model, bundle, labels, args.output_dir)

    metrics = {
        'task': args.task,
        'method': args.method,
        'model': args.model,
        'seed': args.seed,
        **out.metrics,
    }
    save_metrics(metrics, args.output_dir)
    print(metrics)


if __name__ == '__main__':
    main()
