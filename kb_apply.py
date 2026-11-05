# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# 模块说明
#   职责：把 feedback.csv 中「已通过」的条目并入 cases.csv，并增量重建索引
#   依赖：csv, numpy, onnxruntime, tokenizers, rank_bm25, fta_build
#   输入输出：更新 cases/chunks/bm25/vectors 与 FTA md
#   调试提示：只新增、不删除；重复条目按 phenomenon+cause 去重
# ---------------------------------------------------------------------------
"""Merge approved feedback into cases.csv and rebuild retrieval artefacts.

Steps:
  1. feedback.csv -> cases.csv (only rows checked 通过)
  2. chunks.jsonl: keep non-cases chunks, regenerate cases chunks
  3. rebuild BM25; embed only the new case chunks; splice vectors
  4. regenerate FTA markdown
"""
import csv
from csvio import read_rows
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from tokenizer_util import tokenize
from fix_type import classify_fix

ROOT = os.path.dirname(os.path.abspath(__file__))
FB = os.path.join(ROOT, "data", "feedback.csv")
CASES = os.path.join(ROOT, "data", "cases.csv")
RAG = os.path.join(ROOT, "rag")


def read_csv(path):
    if not os.path.exists(path):
        return [], []
    rows = read_rows(path)
    return rows, (list(rows[0].keys()) if rows else [])


def main():
    fb, _ = read_csv(FB)
    approved = [r for r in fb
                if str(r.get("通过", "")).strip().lower() in ("true", "1", "是")
                and r.get("status") != "已应用"]
    print("approved new entries:", len(approved))

    cases, fields = read_csv(CASES)
    existing = {(c["phenomenon"], c["cause"]) for c in cases}
    added = 0
    for r in approved:
        key = (r["phenomenon"], r["cause"])
        if key in existing:
            continue
        cases.append({"phenomenon": r["phenomenon"], "category": r["category"],
                      "cause": r["cause"], "confirm": r["confirm"],
                      "action": r["action"], "verify": r["verify"],
                      "note": r.get("note", ""),
                      "fix_type": classify_fix(r["action"])})
        existing.add(key)
        added += 1

    with open(CASES, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["phenomenon", "category", "cause", "confirm",
                                           "action", "verify", "fix_type", "note"])
        w.writeheader()
        w.writerows(cases)
    print("cases.csv: +%d -> %d" % (added, len(cases)))

    # mark applied
    for r in fb:
        if str(r.get("通过", "")).strip().lower() in ("true", "1", "是"):
            r["status"] = "已应用"
    with open(FB, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fb[0].keys()) if fb else [])
        w.writeheader()
        w.writerows(fb)

    # ---- chunks -----------------------------------------------------------
    chunks = [json.loads(l) for l in open(os.path.join(RAG, "chunks.jsonl"), encoding="utf-8")]
    others = [c for c in chunks if c["source"] != "cases"]
    new_cases = []
    for c in cases:
        note = c.get("note", "")
        text = "现象：%s。原因：%s。确认：%s。处理：%s%s。可验证性：%s。处置类型：%s" % (
            c["phenomenon"], c["cause"], c["confirm"], c["action"],
            ("。备注：%s" % note) if note else "", c["verify"], c.get("fix_type", ""))
        new_cases.append({"text": text, "source": "cases", "category": c.get("category", ""),
                          "phenomenon": c["phenomenon"]})
    for c in others + new_cases:
        c["tokens"] = tokenize(c["text"])

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([c["tokens"] for c in others + new_cases])
    with open(os.path.join(RAG, "chunks.jsonl"), "w", encoding="utf-8") as fh:
        for c in others + new_cases:
            json.dump({k: v for k, v in c.items() if k != "tokens"}, fh, ensure_ascii=False)
            fh.write("\n")
    with open(os.path.join(RAG, "bm25.pkl"), "wb") as fh:
        pickle.dump({"bm25": bm25, "tokens": [c["tokens"] for c in others + new_cases]}, fh)
    print("chunks: %d -> %d" % (len(chunks), len(others) + len(new_cases)))

    # ---- vectors ----------------------------------------------------------
    vec_path = os.path.join(RAG, "vectors.npy")
    old_vecs = np.load(vec_path)
    keep = min(len(others), len(old_vecs))
    if keep < len(others):
        print("WARN: vector/chunk mismatch, keep=%d others=%d" % (keep, len(others)))

    tokenizer = Tokenizer.from_file(os.path.join(RAG, "model", "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    session = ort.InferenceSession(os.path.join(RAG, "model", "model.onnx"), providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}

    texts = [c["text"] for c in new_cases]
    new_vecs = []
    for i in range(0, len(texts), 16):
        enc = tokenizer.encode_batch(texts[i:i + 16])
        maxlen = max(len(e.ids) for e in enc)
        ids = np.zeros((len(enc), maxlen), dtype=np.int64)
        mask = np.zeros((len(enc), maxlen), dtype=np.int64)
        for r, e in enumerate(enc):
            ids[r, :len(e.ids)] = e.ids
            mask[r, :len(e.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = session.run(None, feed)[0][:, 0, :].astype(np.float32)
        out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
        new_vecs.append(out)

    vectors = old_vecs[:keep] if keep else old_vecs[:0]
    if new_vecs:
        vectors = np.vstack([vectors] + new_vecs)
    np.save(vec_path, vectors)
    print("vectors:", vectors.shape)

    # ---- FTA --------------------------------------------------------------
    out = subprocess.run([sys.executable, os.path.join(ROOT, "fta_build.py")],
                         capture_output=True, text=True, cwd=ROOT)
    print(out.stdout.strip()[-300:] or "", out.stderr.strip()[-300:] if out.stderr else "")
    print("done")


if __name__ == "__main__":
    main()
