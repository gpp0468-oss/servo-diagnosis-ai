# -*- coding: utf-8 -*-
"""评测 UI 的 LLM 层：同一批题上，对比「纯检索」与「检索 + DeepSeek」。

为什么要单独测：`ui/app.py` 的 `render_single` 在规则命中时直接 return，
**不调用模型**。所以 LLM 实际只覆盖 data/eval.csv 里 18 条无码题。
本脚本对这 18 条跑两条链路，用同一把尺子量。

用法（必须先设代理和 key，脚本不读任何密钥文件）：

    set HTTPS_PROXY=http://127.0.0.1:7890
    set DEEPSEEK_API_KEY=sk-...
    python eval_llm.py                  # 默认 deepseek-chat、temperature=0
    python eval_llm.py --limit 5        # 先小样本
    python eval_llm.py --model deepseek-reasoner

产出：
    eval_llm_report.md   汇总表 + 逐条明细
    eval_llm_raw.jsonl   原始输出，供复核（不上传）

判据说明：
    - 原因关键词来自 eval.csv 的 expected_cause_keywords
    - 中文按「字符二元组召回率」匹配，阈值 0.5 记命中；这是替代分词的可复现近似，
      不追求语义等价，只回答「这个说法有没有被提到」。
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csvio import read_rows              # noqa: E402
from kb_category import guess_category   # noqa: E402
from rag_query import Rag, describe      # noqa: E402

API = "https://api.deepseek.com/chat/completions"
PROMPT_SRC = os.path.join("ui", "app.py")


def load_system_prompt() -> str:
    """直接从 ui/app.py 里抠 SYSTEM_PROMPT，保证测的是线上提示词。"""
    text = open(PROMPT_SRC, encoding="utf-8").read()
    m = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', text, re.S)
    if not m:
        raise SystemExit("没能从 ui/app.py 里解析出 SYSTEM_PROMPT")
    return m.group(1).strip()


def bigrams(s: str):
    s = re.sub(r"\s+", "", s)
    return [s[i:i + 2] for i in range(len(s) - 1)] if len(s) > 1 else [s]


def recall(phrase: str, text: str) -> float:
    bg = bigrams(phrase)
    if not bg:
        return 0.0
    hit = 0
    for b in bg:
        if b in text:
            hit += 1
    return hit / len(bg)


def cause_tokens(raw: str):
    """'啸叫' / '1.控制电源电压瞬时下降' / 'A；B' → ['啸叫', '控制电源电压瞬时下降', ...]

    注意：本数据集的期望关键词多为 2 字领域术语（啸叫/过流/回零/抱闸…，10/18 条），
    因此下限放到 2 字——2 字词的「二元组召回」就等于「是否出现」，这是对领域术语
    最直接也最可复现的判法。
    """
    parts = []
    for piece in re.split(r"[;\n；]", raw):
        piece = piece.strip()
        if not piece:
            continue
        piece = re.sub(r"^\s*\d+\s*[.、,，)]\s*", "", piece)
        if len(piece) >= 2:
            parts.append(piece)
    return parts


def cover(tokens, text):
    """返回 (平均覆盖率, 命中 token 数, token 总数)；命中 = 召回 ≥ 0.5"""
    if not tokens:
        return 0.0, 0, 0
    rs = [recall(t, text) for t in tokens]
    hits = sum(1 for r in rs if r >= 0.5)
    return sum(rs) / len(rs), hits, len(tokens)


def ask(key, model, system, user, retries=3):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
    }).encode("utf-8")
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                API, data=body,
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode("utf-8"))
            return d["choices"][0]["message"]["content"], time.time() - t0, d["usage"]
        except Exception as exc:                      # noqa: BLE001
            last = exc
            time.sleep(2 * (i + 1))
    raise last


def hypotheses_of(answer: str):
    """尽量从输出里挖出假设条目：先试 JSON，再退化为编号行。"""
    m = re.search(r"\{.*\}", answer, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            hy = obj.get("假设") or obj.get("hypotheses") or []
            out = []
            for h in hy:
                if isinstance(h, str):
                    out.append(h)
                elif isinstance(h, dict):
                    out.append(" ".join(str(v) for v in h.values()))
            if out:
                return out, True
        except Exception:                             # noqa: BLE001
            pass
    lines = [l.strip() for l in answer.splitlines()
             if re.match(r"^\s*(H\d|#?\d+[.、)]|\-|\*)", l.strip())]
    return lines, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0=全部")
    ap.add_argument("--sleep", type=float, default=0.5, help="每次调用间隔秒")
    ap.add_argument("--dataset", default=os.path.join("data", "eval.csv"))
    ap.add_argument("--out", default="eval_llm_report.md")
    ap.add_argument("--raw", default="eval_llm_raw.jsonl")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY（脚本不会读取任何密钥文件）")
    system = load_system_prompt()

    rows = read_rows(args.dataset)
    nocode = [r for r in rows if not r["expected_code"].strip()]
    if args.limit:
        nocode = nocode[:args.limit]

    rag = Rag()
    print("模型 %s ｜ 无码用例 %d 条 ｜ 检索 hybrid=%s" %
          (args.model, len(nocode), rag.hybrid))

    detail, raw_fh = [], open(args.raw, "w", encoding="utf-8")
    tot = {"r_cover": 0.0, "l_cover": 0.0, "r_hit": 0, "l_hit": 0,
           "cat": 0, "fmt": 0, "lat": 0.0, "tok": 0}
    for i, r in enumerate(nocode, 1):
        sym = r["symptom"]
        cats = [guess_category(sym)]
        hits = rag.search(sym, 4, cats)
        context = "".join("[%d]（%s）%s\n\n" % (j, describe(h), h["text"][:400])
                          for j, h in enumerate(hits, 1))
        user = "故障域：%s\n现场描述：%s\n\n【参考资料】\n%s" % (
            guess_category(sym), sym, context)

        answer, lat, usage = ask(key, args.model, system, user)
        raw_fh.write(json.dumps({"id": r["id"], "symptom": sym, "answer": answer,
                                 "latency": round(lat, 2), "usage": usage},
                                ensure_ascii=False) + "\n")

        tokens = cause_tokens(r["expected_cause_keywords"])
        retr_text = context
        r_cover, _, n_tok = cover(tokens, retr_text)
        l_cover, l_hit_n, _ = cover(tokens, answer)

        hy, fmt_ok = hypotheses_of(answer)
        cat_hit = any(guess_category(h) == r["expected_category"] for h in hy)
        # 术语命中：期望术语至少被说到一次（对 2 字术语即「是否出现」）
        term_hit = l_hit_n > 0
        term_hit_r = (cover(tokens, retr_text)[1] > 0)

        tot["r_cover"] += r_cover
        tot["l_cover"] += l_cover
        tot["r_hit"] += int(term_hit_r)
        tot["l_hit"] += int(term_hit)
        tot["cat"] += int(cat_hit)
        tot["fmt"] += int(fmt_ok)
        tot["lat"] += lat
        tot["tok"] += usage.get("total_tokens", 0)

        detail.append({
            "id": r["id"], "cat": r["expected_category"], "tokens": n_tok,
            "r_cover": round(r_cover, 3), "l_cover": round(l_cover, 3),
            "r_hit": term_hit_r, "l_hit": term_hit,
            "cat_hit": cat_hit, "fmt": fmt_ok,
            "hy_n": len(hy), "lat": round(lat, 1),
        })
        print("  [%2d/%2d] %s 检索覆盖 %.2f → LLM 覆盖 %.2f  %s" % (
            i, len(nocode), r["id"], r_cover, l_cover,
            "类✓" if cat_hit else "类✗"))
        time.sleep(args.sleep)
    raw_fh.close()

    n = len(nocode)
    lines = []
    lines.append("# LLM 层评测（%s）\n" % args.model)
    lines.append("对象：`data/eval.csv` 中 **%d 条无码用例**。"
                 "带报警码的 15 条在 `render_single` 里直接走规则直出，不调模型，故不在此表。\n" % n)
    lines.append("| 指标 | 纯检索（Top-4） | 检索 + %s |" % args.model)
    lines.append("|---|---|---|")
    lines.append("| 原因术语覆盖率（均值） | %.1f%% | **%.1f%%** |"
                 % (tot["r_cover"] / n * 100, tot["l_cover"] / n * 100))
    lines.append("| 原因术语命中率（该条至少说对一个术语） | %d/%d = %.0f%% | **%d/%d = %.0f%%** |"
                 % (tot["r_hit"], n, tot["r_hit"] / n * 100,
                    tot["l_hit"], n, tot["l_hit"] / n * 100))
    lines.append("| 故障大类命中率 | —（不输出类别） | **%d/%d = %.0f%%** |"
                 % (tot["cat"], n, tot["cat"] / n * 100))
    lines.append("| 输出可解析为结构化假设 | — | %d/%d = %.0f%% |"
                 % (tot["fmt"], n, tot["fmt"] / n * 100))
    lines.append("| 平均延迟 | — | %.1f s |" % (tot["lat"] / n))
    lines.append("| 总 token 消耗 | 0 | %d |" % tot["tok"])
    lines.append("")
    lines.append("> 「原因术语覆盖率」= 对 `expected_cause_keywords` 的字符二元组召回率取均值，"
                 "阈值 0.5 记命中。该数据集 10/18 条的关键词是 2 字领域术语（啸叫/过流/回零/抱闸…），"
                 "对 2 字词而言这个指标就等于「模型有没有说出这个术语」。"
                 "**它不判语义等价**，同义改写（把「啸叫」写成「尖叫」）会记为未命中。")
    lines.append("")
    lines.append("| id | 期望大类 | 检索覆盖 | LLM 覆盖 | 术语命中 | 大类命中 | 延迟 |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in detail:
        lines.append("| %s | %s | %.2f | %.2f | %s | %s | %.1fs |" % (
            d["id"], d["cat"], d["r_cover"], d["l_cover"],
            "✓" if d["l_hit"] else "✗", "✓" if d["cat_hit"] else "✗", d["lat"]))
    lines.append("")
    lines.append("**限制**：样本仅 %d 条；`temperature=0` 只降低而非消除波动；"
                 "关键词匹配对改写句不敏感，长句会被低估。" % n)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print("\n汇总：")
    print("  原因术语覆盖  纯检索 %.1f%% → LLM %.1f%%" %
          (tot["r_cover"] / n * 100, tot["l_cover"] / n * 100))
    print("  术语命中率    检索 %d/%d → LLM %d/%d" % (tot["r_hit"], n, tot["l_hit"], n))
    print("  大类命中      %d/%d = %.0f%%" % (tot["cat"], n, tot["cat"] / n * 100))
    print("  可解析        %d/%d    总 token %d    平均 %.1fs" %
          (tot["fmt"], n, tot["tok"], tot["lat"] / n))
    print("  报告：%s   原文：%s" % (args.out, args.raw))


if __name__ == "__main__":
    main()
