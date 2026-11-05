# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# 模块说明
#   职责：从手册 PDF / 笔记 / rules.csv / cases.csv 生成 chunks.jsonl 与 bm25.pkl
#   依赖：pypdf, rank_bm25, tokenizer_util, kb_category
#   输入输出：写 rag/chunks.jsonl, rag/bm25.pkl（不写 vectors.npy）
#   调试提示：重建只覆盖 chunks/bm25，必须重跑 embed_build，否则向量与切片错位。
#             无 PDF 时跳过 manual 部分，产物与 Release 的 rag-v1.0 不是同一份索引。
# ---------------------------------------------------------------------------
"""Build the SV660N RAG knowledge base (section-aware chunking).

Sources:
  1. SV660N manual (PDF)         -> split by numbered headings, ~700 chars per chunk
  2. docs/notes/00-伺服原理笔记.md -> split by ### subsections
  3. data/rules.csv              -> one chunk per fault code
  4. data/cases.csv              -> one chunk per phenomenon cause

用法：
    python rag_build.py                 # 无 PDF：仅用 笔记+rules+cases 构建（本地调试用）
    python rag_build.py --pdf <PDF 路径>  # 含 manual 的完整重建
    SV660N_PDF=<PDF 路径> python rag_build.py

重要：本脚本产出的是「重建索引」，与 Release 的 rag-v1.0 **不是同一份产物**。
      README 里的评测指标只在 Release rag-v1.0 上保证可复现。
"""
import argparse
import json
import os
import pickle
import re
import time

from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from csvio import read_rows
from kb_category import guess_category
from tokenizer_util import tokenize

NOTE = "docs/notes/00-伺服原理笔记.md"
RULES = "data/rules.csv"
CASES = "data/cases.csv"
OUT_DIR = "rag"
MAX_LEN = 700

HEAD_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\s+[\u4e00-\u9fa5A-Za-z]")
NOISE = re.compile(r"^([‑-]\d+[‑-]|故障处理|伺服驱动器|目录)$")

# P0-7 / P0-8 警告文案（逐字保留互认版 §2.1，勿改）
WARN_NO_PDF = (
    "[WARN] 未提供手册 PDF：已跳过 manual 部分。\n"
    "       本次产物与 Release 的 rag-v1.0 不是同一份索引，\n"
    "       → 可用于本地调试，不可用于复现 README 指标。"
)
WARN_REBUILD = (
    "[WARN] 本次为“重建索引”，与 Release 的 rag-v1.0 不是同一份产物。\n"
    "       README 指标只在 Release rag-v1.0 上保证可复现。"
)
WARN_STALE_VEC = (
    "[WARN] 已覆盖 rag/chunks.jsonl 与 rag/bm25.pkl，但 rag/vectors.npy 仍是旧索引。\n"
    "       向量与切片现在不对应，eval_run 会静默给出错误结果。\n"
    "       → 立即重跑：python embed_build.py\n"
    "       → 或删除 rag/ 后重新解压 Release 的 rag-v1.0.zip"
)

ap = argparse.ArgumentParser(
    description="构建 RAG 索引（manual 部分需要手册 PDF，可用 --pdf 或 SV660N_PDF 指定）")
ap.add_argument("--pdf", default=os.environ.get("SV660N_PDF", ""),
                help="手册 PDF 路径；也可用环境变量 SV660N_PDF")
PDF = (ap.parse_args().pdf or "").strip()

os.makedirs(OUT_DIR, exist_ok=True)
chunks = []


def add(text, source, category=None, **meta):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 40:
        return
    chunks.append({"text": text, "source": source,
                   "category": category or guess_category(text), **meta})


def _warn_stale_vectors(n_chunks):
    """P0-8：本脚本只写 chunks/bm25，不写 vectors.npy；若旧向量仍在，必须显式提醒。"""
    vec_path = os.path.join(OUT_DIR, "vectors.npy")
    chunk_path = os.path.join(OUT_DIR, "chunks.jsonl")
    if not os.path.exists(vec_path) or not os.path.exists(chunk_path):
        return
    stale = os.path.getmtime(vec_path) < os.path.getmtime(chunk_path)
    n_vec = None
    try:
        import numpy as np
        n_vec = int(np.load(vec_path, mmap_mode="r").shape[0])
    except Exception:  # noqa: BLE001
        pass
    if stale or (n_vec is not None and n_vec != n_chunks):
        print(WARN_STALE_VEC)


# ---------------------------------------------------------------- 1. manual
t0 = time.time()
has_pdf = bool(PDF) and os.path.exists(PDF)

if has_pdf:
    reader = PdfReader(PDF)
    chapter = ""

    for p in range(1, len(reader.pages) + 1):
        try:
            txt = reader.pages[p - 1].extract_text() or ""
        except Exception:
            continue
        if len(txt.strip()) < 40:
            continue

        buf, heading = [], ""

        for line in txt.split("\n"):
            s = line.strip()
            if not s or NOISE.match(s):
                continue
            m = HEAD_RE.match(s)
            if m:
                if buf:
                    add(" ".join(buf), "manual", page=p - 1, chapter=chapter, heading=heading[:40])
                    buf = []
                heading = s
                chapter = m.group(1).split(".")[0]
                buf.append(s)
                continue
            buf.append(s)
            if sum(len(x) for x in buf) > MAX_LEN:
                add(" ".join(buf), "manual", page=p - 1, chapter=chapter, heading=heading[:40])
                buf = []
        if buf:
            add(" ".join(buf), "manual", page=p - 1, chapter=chapter, heading=heading[:40])

    print("manual chunks:", len(chunks), "%.0fs" % (time.time() - t0))
else:
    if PDF:
        print("[WARN] 指定的 PDF 不存在：%s" % PDF)
    print(WARN_NO_PDF)

# ---------------------------------------------------------------- 2. note
note = open(NOTE, encoding="utf-8").read()
for part in re.split(r"(?m)^##\s+", note)[1:]:
    title = part.split("\n", 1)[0].strip()
    body = part.split("\n", 1)[1] if "\n" in part else ""
    subs = re.split(r"(?m)^###\s+", body)
    if len(subs) > 1:
        for sub in subs[1:]:
            sub_title = sub.split("\n", 1)[0].strip()
            sub_body = re.sub(r"[#>*`]", "", sub.split("\n", 1)[1] if "\n" in sub else "")
            for i in range(0, len(sub_body), 900):
                add(sub_title + "。" + sub_body[i:i + 900], "note",
                    category=guess_category(sub_title),
                    section="%s / %s" % (title, sub_title))
    else:
        clean = re.sub(r"[#>*`]", "", body)
        for i in range(0, len(clean), 900):
            add(title + "。" + clean[i:i + 900], "note", category=guess_category(title), section=title)

print("with note:", len(chunks))

# ---------------------------------------------------------------- 3. rules
rows = read_rows(RULES)
by_code = {}
for r in rows:
    by_code.setdefault(r["code"], []).append(r)
for code, rs in by_code.items():
    r0 = rs[0]
    lines = ["故障码 %s（%s）" % (code, r0["name"]),
             "故障源：%s；现象类别：%s；故障域：%s；级别：%s；可复位：%s；处置类型：%s"
             % (r0["category"], r0.get("phenomenon_tags", ""), r0["domain"],
                r0["level"], r0["resettable"], r0.get("fix_type", ""))]
    for idx, r in enumerate(rs, 1):
        lines.append("%d. 原因：%s；确认：%s；处理：%s；可验证性：%s"
                     % (idx, r["cause"], r["confirm"], r["action"], r["verify"]))
    add("\n".join(lines), "rules", category=r0["category"], code=code, page=r0.get("page", ""))

print("with rules:", len(chunks))

# ---------------------------------------------------------------- 4. cases
if os.path.exists(CASES):
    for c in read_rows(CASES):
        c_note = c.get("note", "")
        extra = ("。备注：%s" % c_note) if c_note else ""
        add("现象：%s。原因：%s。确认：%s。处理：%s%s。可验证性：%s。处置类型：%s"
            % (c["phenomenon"], c["cause"], c["confirm"], c["action"], extra,
               c["verify"], c.get("fix_type", "")),
            "cases", category=c.get("category", ""), phenomenon=c["phenomenon"])
    print("with cases:", len(chunks))

# ---------------------------------------------------------------- index
for c in chunks:
    c["tokens"] = tokenize(c["text"])
corpus = [c["tokens"] for c in chunks]
bm25 = BM25Okapi(corpus)

with open(os.path.join(OUT_DIR, "chunks.jsonl"), "w", encoding="utf-8") as fh:
    for c in chunks:
        json.dump({k: v for k, v in c.items() if k != "tokens"}, fh, ensure_ascii=False)
        fh.write("\n")
with open(os.path.join(OUT_DIR, "bm25.pkl"), "wb") as fh:
    pickle.dump({"bm25": bm25, "tokens": corpus}, fh)

print("total chunks:", len(chunks), "total %.0fs" % (time.time() - t0))

# ---------------------------------------------------------------- 边界声明
if has_pdf:
    print(WARN_REBUILD)

# 本脚本只覆盖 chunks/bm25，不覆盖 vectors.npy —— 检查是否留下错位
_warn_stale_vectors(len(chunks))
