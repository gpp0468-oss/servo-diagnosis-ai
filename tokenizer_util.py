# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# 模块说明
#   职责：单字 + 二元组分词，替代 jieba（避免新版 Python 装不上）
#   依赖：re
#   输入输出：tokenize() -> list[str]
#   调试提示：检索与建索引必须用同一分词器
# ---------------------------------------------------------------------------
"""轻量中文分词：单字 + 二元组（不依赖 jieba，避免新 Python 版本装不上）。"""
import re

_SPLIT = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


def tokenize(text: str):
    text = (text or "").lower()
    tokens = []
    for chunk in _SPLIT.split(text):
        if not chunk:
            continue
        chars = list(chunk)
        tokens.extend(chars)
        tokens.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return tokens


def tokenize_query(text: str):
    return tokenize(text)
