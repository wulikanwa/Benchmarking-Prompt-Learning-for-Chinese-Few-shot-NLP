from typing import Dict, Tuple

SENTIMENT_LABELS = {0: "消极", 1: "积极"}
TOPIC_LABELS = {
    0: "财经", 1: "房产", 2: "教育", 3: "科技", 4: "军事",
    5: "汽车", 6: "体育", 7: "游戏", 8: "娱乐", 9: "时尚"
}


def build_classification_prompt(task: str, text: str) -> str:
    if task == "sentiment":
        return f"下面是一条中文评论：{text}\n这条评论的情感是：[MASK]。"
    if task == "topic":
        return f"下面是一条中文新闻标题：{text}\n这条新闻属于：[MASK]。"
    raise ValueError(f"Unsupported classification prompt task: {task}")


def get_label_text(task: str, label_id: int) -> str:
    if task == "sentiment":
        return SENTIMENT_LABELS[label_id]
    if task == "topic":
        return TOPIC_LABELS[label_id]
    raise ValueError(f"Unsupported label task: {task}")


def build_generation_prompt(task: str, source_text: str, extra: Dict | None = None) -> str:
    if task == "summarization":
        return f"请为下面的中文文本生成简洁摘要：\n{source_text}"
    if task == "translation":
        return f"请将下面的中文句子翻译成英文：\n{source_text}"
    raise ValueError(f"Unsupported generation task: {task}")


def build_qa_prompt(context: str, question: str) -> str:
    return f"阅读下面的文章并回答问题。\n文章：{context}\n问题：{question}\n答案："
