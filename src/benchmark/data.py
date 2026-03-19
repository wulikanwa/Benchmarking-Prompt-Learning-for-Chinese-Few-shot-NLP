from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import Dataset
from transformers import AutoTokenizer

from .prompts import build_classification_prompt, build_generation_prompt, build_qa_prompt
from .utils import read_jsonl


@dataclass
class TokenizedBundle:
    train: Dataset
    val: Dataset
    test: Dataset


def load_jsonl_dataset(train_file: str, val_file: str, test_file: str) -> Dict[str, Dataset]:
    return {
        'train': Dataset.from_list(read_jsonl(train_file)),
        'val': Dataset.from_list(read_jsonl(val_file)),
        'test': Dataset.from_list(read_jsonl(test_file)),
    }


def build_tokenized_classification(task: str, model_name: str, train_file: str, val_file: str, test_file: str,
                                   max_length: int = 256) -> TokenizedBundle:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    ds = load_jsonl_dataset(train_file, val_file, test_file)

    def preprocess(batch):
        texts = [build_classification_prompt(task, x) for x in batch['text']]
        enc = tok(texts, truncation=True, padding='max_length', max_length=max_length)
        enc['labels'] = batch['label']
        return enc

    return TokenizedBundle(
        train=ds['train'].map(preprocess, batched=True),
        val=ds['val'].map(preprocess, batched=True),
        test=ds['test'].map(preprocess, batched=True),
    )


def build_tokenized_qa(model_name: str, train_file: str, val_file: str, test_file: str,
                       max_length: int = 384) -> TokenizedBundle:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    ds = load_jsonl_dataset(train_file, val_file, test_file)

    def preprocess(batch):
        prompts = [build_qa_prompt(c, q) for c, q in zip(batch['context'], batch['question'])]
        enc = tok(prompts, truncation=True, padding='max_length', max_length=max_length)
        # simplified placeholder: use text labels for evaluation only
        enc['labels_text'] = batch['answer']
        return enc

    return TokenizedBundle(
        train=ds['train'].map(preprocess, batched=True),
        val=ds['val'].map(preprocess, batched=True),
        test=ds['test'].map(preprocess, batched=True),
    )


def build_tokenized_generation(task: str, model_name: str, train_file: str, val_file: str, test_file: str,
                               max_source_length: int = 256, max_target_length: int = 128) -> TokenizedBundle:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    ds = load_jsonl_dataset(train_file, val_file, test_file)

    target_field = 'summary' if task == 'summarization' else 'target'

    def preprocess(batch):
        src = [build_generation_prompt(task, x) for x in batch['source']]
        model_inputs = tok(src, truncation=True, padding='max_length', max_length=max_source_length)
        labels = tok(text_target=batch[target_field], truncation=True, padding='max_length', max_length=max_target_length)
        model_inputs['labels'] = labels['input_ids']
        model_inputs['labels_text'] = batch[target_field]
        return model_inputs

    return TokenizedBundle(
        train=ds['train'].map(preprocess, batched=True),
        val=ds['val'].map(preprocess, batched=True),
        test=ds['test'].map(preprocess, batched=True),
    )
