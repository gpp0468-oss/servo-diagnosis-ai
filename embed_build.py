# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# 模块说明
#   职责：用 bge-small-zh ONNX 模型为全部切片算向量
#   依赖：onnxruntime, tokenizers, numpy
#   输入输出：写 rag/vectors.npy（N×512，L2 归一化）
#   调试提示：耗时约 45s；行数必须等于 chunks 行数
# ---------------------------------------------------------------------------
"""Compute sentence embeddings for all RAG chunks (bge-small-zh-v1.5, ONNX)."""
import json
import os
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DIR = "rag"
BATCH = 16

chunks = [json.loads(l) for l in open(os.path.join(DIR, "chunks.jsonl"), encoding="utf-8")]
tokenizer = Tokenizer.from_file(os.path.join(DIR, "model", "tokenizer.json"))
tokenizer.enable_truncation(max_length=512)
session = ort.InferenceSession(os.path.join(DIR, "model", "model.onnx"),
                               providers=["CPUExecutionProvider"])
input_names = {i.name for i in session.get_inputs()}
print("inputs:", input_names, "chunks:", len(chunks))

t0 = time.time()
vecs = []
for i in range(0, len(chunks), BATCH):
    batch = [c["text"] for c in chunks[i:i + BATCH]]
    enc = tokenizer.encode_batch(batch)
    maxlen = max(len(e.ids) for e in enc)
    ids = np.zeros((len(enc), maxlen), dtype=np.int64)
    mask = np.zeros((len(enc), maxlen), dtype=np.int64)
    for r, e in enumerate(enc):
        ids[r, :len(e.ids)] = e.ids
        mask[r, :len(e.ids)] = 1
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(ids)
    out = session.run(None, feed)[0]
    v = out[:, 0, :].astype(np.float32)          # CLS pooling
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    vecs.append(v)
    if (i // BATCH) % 20 == 0:
        print("  %d/%d  %.0fs" % (i, len(chunks), time.time() - t0))

vectors = np.vstack(vecs)
np.save(os.path.join(DIR, "vectors.npy"), vectors)
print("saved", vectors.shape, "%.0fs" % (time.time() - t0))

