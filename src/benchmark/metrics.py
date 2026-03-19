from __future__ import annotations
from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import evaluate

rouge = evaluate.load('rouge')
bleu = evaluate.load('sacrebleu')


def classification_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'macro_f1': float(f1_score(labels, preds, average='macro')),
    }


def qa_exact_match(preds: List[str], labels: List[str]) -> Dict[str, float]:
    norm = lambda s: ''.join(str(s).split())
    em = np.mean([1.0 if norm(p) == norm(g) else 0.0 for p, g in zip(preds, labels)]) if labels else 0.0
    return {'exact_match': float(em)}


def summarization_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    result = rouge.compute(predictions=preds, references=refs, use_stemmer=False)
    return {'rouge1': float(result['rouge1'])}


def translation_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    result = bleu.compute(predictions=preds, references=[[r] for r in refs])
    return {'bleu': float(result['score'])}
