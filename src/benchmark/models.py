from __future__ import annotations
from typing import Tuple

from peft import LoraConfig, PromptEncoderConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
)


def build_classification_model(model_name: str, num_labels: int, method: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if method == 'full_ft' or method == 'manual_prompt':
        return model
    if method == 'lora':
        config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1)
        return get_peft_model(model, config)
    if method == 'ptuning_v2':
        config = PromptEncoderConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=32, encoder_hidden_size=128)
        return get_peft_model(model, config)
    raise ValueError(f'Unsupported method: {method}')


def build_qa_model(model_name: str, method: str):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    if method == 'full_ft' or method == 'manual_prompt':
        return model
    if method == 'lora':
        config = LoraConfig(task_type=TaskType.QUESTION_ANS, r=8, lora_alpha=16, lora_dropout=0.1)
        return get_peft_model(model, config)
    if method == 'ptuning_v2':
        config = PromptEncoderConfig(task_type=TaskType.QUESTION_ANS, num_virtual_tokens=32, encoder_hidden_size=128)
        return get_peft_model(model, config)
    raise ValueError(f'Unsupported method: {method}')


def build_seq2seq_model(model_name: str, method: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if method == 'full_ft' or method == 'manual_prompt':
        return model
    if method == 'lora':
        config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16, lora_dropout=0.1)
        return get_peft_model(model, config)
    if method == 'ptuning_v2':
        config = PromptEncoderConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=32, encoder_hidden_size=128)
        return get_peft_model(model, config)
    raise ValueError(f'Unsupported method: {method}')
