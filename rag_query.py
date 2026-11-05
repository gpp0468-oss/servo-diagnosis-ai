# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# 模块说明
#   职责：BM25 + 向量混合检索，支持大类加权与查询扩展
#   依赖：rank_bm25, onnxruntime, tokenizers, numpy, tokenizer_util, kb_category
#   输入输出：Rag.search() -> list[chunk]
#   调试提示：向量与 chunks 行序必须一致；hybrid=False 表示缺模型降级为 BM25；
#             行数不一致会抛 RagIndexMismatch（硬失败，不降级）
# ---------------------------------------------------------------------------
"""Hybrid retrieval (BM25 + bge-small-zh embeddings) with category boost.

用法：
    python rag_query.py "定位不准怎么办"
    python rag_query.py --cat 通讯与总线 "EE08.1 状态切换异常"
"""
import json
import os
import pickle
import sys

try:  # 向量检索依赖缺失时自动降级为 BM25
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    HAS_VECTOR = True
except Exception:  # noqa: BLE001
    HAS_VECTOR = False

from kb_category import CATEGORIES, expand_query
from tokenizer_util import tokenize

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag")


class RagIndexMismatch(RuntimeError):
    """chunks.jsonl 与 vectors.npy 行数不一致。

    检索按行下标把向量对到切片上（hits=[(int(i), score)] -> chunks[i]），
    行数不同意味着「新切片 + 旧向量」，结果会静默出错。
    所以这里必须硬失败，不能像缺模型那样降级成纯 BM25。
    """


# 三个入口脚本共用的缺索引提示（措辞统一，避免读者以为是不同问题）
RAG_MISSING_HINT = (
    "检索索引（rag/）尚未就绪，二选一：\n"
    "  1) 复现 README 指标：下载 Release 的 rag-v1.0.zip，解压到仓库根目录得到 rag/\n"
    "  2) 本地重建（结果不等于 rag-v1.0，不承诺 README 指标）：\n"
    "     python rag_build.py && python embed_build.py   # 需 rag/model/model.onnx"
)


class Rag:
    def __init__(self, path: str = DIR, k: int = 5):
        self.k = k
        self.chunks = [json.loads(l) for l in open(os.path.join(path, "chunks.jsonl"), encoding="utf-8")]
        with open(os.path.join(path, "bm25.pkl"), "rb") as fh:
            self.bm25 = pickle.load(fh)["bm25"]
        self.cats = [c.get("category", "") for c in self.chunks]

        self.vectors = None
        vec_path = os.path.join(path, "vectors.npy")
        model_path = os.path.join(path, "model", "model.onnx")
        tok_path = os.path.join(path, "model", "tokenizer.json")
        if HAS_VECTOR and os.path.exists(vec_path) and os.path.exists(model_path):
            try:
                self.vectors = np.load(vec_path)
                self._tok = Tokenizer.from_file(tok_path)
                self._tok.enable_truncation(max_length=512)
                self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self._names = {i.name for i in self._session.get_inputs()}
            except Exception:
                self.vectors = None

        # P0-8 硬校验：行数不一致时禁止继续，否则 eval_run 会静默输出错误指标
        if self.vectors is not None and len(self.vectors) != len(self.chunks):
            raise RagIndexMismatch(
                "rag 索引不一致：chunks.jsonl 有 %d 条，vectors.npy 有 %d 条。\n"
                "请重跑 python embed_build.py，或删除 rag/ 后重新解压 Release 的 rag-v1.0.zip。\n"
                "重建产物不能用于复现 README 指标。"
                % (len(self.chunks), len(self.vectors)))

    @property
    def hybrid(self) -> bool:
        return self.vectors is not None

    def _embed(self, texts):
        enc = self._tok.encode_batch(texts)
        maxlen = max(len(e.ids) for e in enc)
        ids = np.zeros((len(enc), maxlen), dtype=np.int64)
        mask = np.zeros((len(enc), maxlen), dtype=np.int64)
        for r, e in enumerate(enc):
            ids[r, :len(e.ids)] = e.ids
            mask[r, :len(e.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self._session.run(None, feed)[0]
        v = out[:, 0, :].astype(np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    def search(self, query: str, k: int | None = None, categories=None):
        k = k or self.k
        q = expand_query(query)
        toks = tokenize(q)
        bm = np.asarray(self.bm25.get_scores(toks), dtype=np.float32)
        if bm.max() > 0:
            bm = bm / bm.max()

        if self.vectors is not None:
            qv = self._embed([q])[0]
            cos = self.vectors @ qv
            score = 0.5 * bm + 0.5 * ((cos + 1.0) / 2.0)
        else:
            score = bm

        if categories:
            mask = np.array([c in categories for c in self.cats], dtype=np.float32)
            score = score + 0.05 * mask

        order = np.argsort(-score)[:k]
        hits = [(int(i), float(score[i])) for i in order if score[i] > 0]
        return self._pack(hits, bool(categories))

    def _pack(self, items, boosted):
        out = []
        for i, s in items:
            c = dict(self.chunks[i])
            c["score"] = round(s, 4)
            c["filtered"] = boosted
            out.append(c)
        return out


def describe(c: dict) -> str:
    cat = c.get("category", "")
    if c["source"] == "manual":
        base = "手册 P.%s（章节 %s）" % (c.get("page", "-"), c.get("chapter", "-"))
    elif c["source"] == "note":
        base = "学习笔记 · %s" % c.get("section", "-")
    elif c["source"] == "rules":
        base = "规则库 · %s（P.%s）" % (c.get("code", "-"), c.get("page", "-"))
    elif c["source"] == "cases":
        base = "案例库 · %s" % c.get("phenomenon", "-")
    else:
        base = c["source"]
    return "%s · %s" % (cat, base) if cat else base


if __name__ == "__main__":
    args = sys.argv[1:]
    cats = None
    if args and args[0] == "--cat":
        cats = [args[1]]
        args = args[2:]
    q = " ".join(args) or "定位不准"
    rag = Rag()
    print("query:", q, "| filter:", cats or "全部", "| hybrid:", rag.hybrid)
    for i, c in enumerate(rag.search(q, 5, cats), 1):
        print("\n[%d] %s  score=%.3f" % (i, describe(c), c["score"]))
        print("    " + c["text"][:150].replace("\n", " "))






