from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .metrics import classification_metrics, qa_exact_match, summarization_metrics, translation_metrics


@dataclass
class TrainOutput:
    metrics: Dict


def train_classification(model, model_name: str, tokenized, output_dir: str, epochs: int = 3, batch_size: int = 8):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        evaluation_strategy='epoch',
        save_strategy='no',
        logging_steps=10,
        report_to=[],
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return classification_metrics(preds.tolist(), labels.tolist())

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized.train,
        eval_dataset=tokenized.val,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    test_metrics = trainer.evaluate(eval_dataset=tokenized.test)
    test_metrics['train_time_sec'] = train_time
    test_metrics['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return TrainOutput(metrics=test_metrics)


def train_qa(model, model_name: str, tokenized, raw_test_labels: List[str], output_dir: str, epochs: int = 2, batch_size: int = 8):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        evaluation_strategy='no',
        save_strategy='no',
        logging_steps=10,
        report_to=[],
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized.train.remove_columns(['labels_text']),
        eval_dataset=tokenized.val.remove_columns(['labels_text']),
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
    )
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    # simplified QA inference placeholder
    preds = ["" for _ in raw_test_labels]
    metrics = qa_exact_match(preds, raw_test_labels)
    metrics['train_time_sec'] = train_time
    metrics['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return TrainOutput(metrics=metrics)


def train_generation(task: str, model, model_name: str, tokenized, raw_test_labels: List[str], output_dir: str,
                     epochs: int = 3, batch_size: int = 4, gen_max_len: int = 64):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        num_train_epochs=epochs,
        evaluation_strategy='no',
        save_strategy='no',
        logging_steps=10,
        report_to=[],
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized.train.remove_columns(['labels_text']),
        eval_dataset=tokenized.val.remove_columns(['labels_text']),
        tokenizer=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    model.eval()
    preds = []
    dl = torch.utils.data.DataLoader(tokenized.test.remove_columns(['labels_text']), batch_size=batch_size)
    device = model.device
    for batch in dl:
        batch = {k: torch.tensor(v).to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=gen_max_len)
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))

    if task == 'summarization':
        metrics = summarization_metrics(preds, raw_test_labels)
    else:
        metrics = translation_metrics(preds, raw_test_labels)
    metrics['train_time_sec'] = train_time
    metrics['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return TrainOutput(metrics=metrics)
