# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 模块说明
#   职责：成本敏感的信息增益排序——决定分步诊断"下一步查什么"
#   依赖：math, random, re
#   输入输出：rank_checks(hypotheses) -> [(score, ig, cost, hyp), ...]
#   调试提示：似然分两部分——对角线=这条判据对得上它自己的假设（sens × specificity），
#             非对角线=对不上却也可能"像"（基线误报 + 语义重合抬高）。
#             python diag_rank.py 会打印「收敛阈值 → 步数/收敛率/降幅」敏感性表：
#             · 收敛率接近 0% ⇒ 似然太平，steps 被假设条数封顶，降幅是假的
#               （2026-11-04 修过一次：steps 恒等于假设条数，降幅恒为 0%）；
#             · 各阈值降幅都 <5% ⇒ 判据区分度不足，收益来自「先做便宜的」而非「先做信息量大的」。
# ---------------------------------------------------------------------------
import math
import random
import re

# 检查成本：换件远贵于实测/观察
COST = {"实测": 1.0, "观察": 1.5, "替换": 5.0}

# 灵敏度：这个检查能多大程度确认"确实是这条原因"
SENS = {"实测": 0.90, "观察": 0.70, "替换": 0.95}

FALSE_POS = 0.10    # 不是这条原因，判据却"像"的基线误报率
OVERLAP_W = 0.55    # 判据与别的假设语义重合时，把误报率抬高的权重
SENS_FLOOR = 0.65   # 判据特异性对灵敏度的折扣下限（spec=0.5 时只打到 82.5%）
CONV_TARGET = 0.80  # 后验到此即视为收敛（0.90 在 n 步内基本达不到，见 __main__ 扫描）
SEED = 42

PARAM_RE = re.compile(r"H0[0-9A-Ea-e]\.\d{1,2}")
NUM_RE = re.compile(r"[0-9]+\s*(%|V|ms|rpm|Ω|MΩ|Hz|A)")


def specificity(confirm: str) -> float:
    """判据的特异性：有参数号/阈值 → 高；泛泛检查 → 低。"""
    c = confirm or ""
    score = 0.5
    if PARAM_RE.search(c):
        score += 0.25
    if NUM_RE.search(c):
        score += 0.15
    if any(k in c for k in ("查看", "测量", "读", "观察")):
        score += 0.10
    if any(k in c for k in ("检查", "确认")) and len(c) < 12:
        score -= 0.10
    return max(0.5, min(0.95, score))


def check_type(h) -> str:
    """成本/灵敏度一律取 check_type，缺失才回退 verify（交接文档决策 3）。"""
    return h.get("check_type") or h.get("verify") or "观察"


def sens_of(verify, spec):
    """灵敏度 = 检查类型基线 × 判据特异性的温和折扣。"""
    base = SENS.get(verify, 0.70)
    return max(0.05, min(0.98, base * (SENS_FLOOR + (1 - SENS_FLOOR) * spec)))


def sens_fp(verify: str, confirm: str):
    """返回 (对角线灵敏度, 非对角线基线误报)，都是 0~1 的概率。"""
    return sens_of(verify, specificity(confirm)), FALSE_POS


def entropy(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0)


# ---------------- 故障-判据依赖矩阵（用文本重合度近似） ----------------

def _bigrams(t):
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", t or "")
    return {t[i:i+2] for i in range(len(t)-1)} or set()


def dep_matrix(hypotheses):
    """D[i][j] = 判据 i 与假设 j 的语义重合度（0~1）。"""
    causes = [_bigrams(h.get("假设", "")) for h in hypotheses]
    checks = [_bigrams(h.get("判据", "")) for h in hypotheses]
    D = []
    for ci in checks:
        row = []
        for cj in causes:
            if not ci or not cj:
                row.append(0.0)
            else:
                row.append(len(ci & cj) / len(ci | cj))
        D.append(row)
    return D


def likelihood(D, i, j, verify, spec=1.0):
    """P(判据 i 确认 | 假设 j 成立)。

    对角线：这条判据就是为这条假设写的，确认概率 = 灵敏度 × 判据特异性。
    非对角线：不是这条原因，但判据措辞与它重合时，更容易被误读成"像"。
    """
    sens = sens_of(verify, spec)
    if i == j:
        return sens
    ov = D[i][j] if D is not None else 0.0
    return min(0.95, FALSE_POS + OVERLAP_W * ov * sens)


def _posterior(priors, D, i, verify, outcome, spec=1.0):
    """给定判据 i 的结果（True=确认 / False=未确认），更新后验。"""
    lik = []
    for j in range(len(priors)):
        p = likelihood(D, i, j, verify, spec)
        lik.append(p if outcome else 1 - p)
    total = sum(p * l for p, l in zip(priors, lik))
    if total <= 1e-12:
        return list(priors)
    return [(p * l) / total for p, l in zip(priors, lik)]


def ig_check(priors, D, i, verify, spec=1.0):
    """判据 i 的期望信息增益（bit）。"""
    lik = [likelihood(D, i, j, verify, spec) for j in range(len(priors))]
    p_true = sum(p * l for p, l in zip(priors, lik))
    if p_true <= 1e-9 or p_true >= 1 - 1e-9:
        return 0.0
    h0 = entropy(priors)
    h_true = entropy(_posterior(priors, D, i, verify, True, spec))
    h_false = entropy(_posterior(priors, D, i, verify, False, spec))
    return max(0.0, h0 - (p_true * h_true + (1 - p_true) * h_false))


def rank_checks(hypotheses, use_relevance=True):
    """返回按「单位成本信息增益」降序的检查列表。

    use_relevance=False 时忽略语义重合（非对角线一律基线误报），
    作为"只看判据自身可信度、不看区分能力"的对照组。
    """
    n = len(hypotheses)
    if n == 0:
        return []
    priors = [1.0 / n] * n
    D = dep_matrix(hypotheses) if use_relevance else None
    out = []
    for i, h in enumerate(hypotheses):
        ct = check_type(h)
        spec = specificity(h.get("判据", ""))
        cost = COST.get(ct, 1.5)
        ig = ig_check(priors, D, i, ct, spec)
        out.append((ig / cost, ig, cost, h))
    out.sort(key=lambda x: (-x[0], x[2]))
    return out


# ---------------------------------------------------------------- 模拟对比

def _simulate(hypotheses, policy, D=None, trials=200, seed=SEED):
    """返回 (平均步数, 收敛率)。收敛率 = 后验到达 CONV_TARGET 的仿真占比。"""
    n = len(hypotheses)
    if n == 0:
        return 0.0, 1.0
    D = D if D is not None else dep_matrix(hypotheses)
    rng = random.Random(seed)
    total, converged = 0, 0
    for truth in range(n):
        for _ in range(trials):
            priors = [1.0 / n] * n
            asked, steps, ok = set(), 0, False
            while True:
                if max(priors) >= CONV_TARGET:
                    ok = True
                    break
                if steps >= n:
                    break
                cand = [k for k in range(n) if k not in asked]
                if not cand:
                    break
                k = policy(priors, cand, hypotheses, D)
                asked.add(k)
                steps += 1
                ct = check_type(hypotheses[k])
                spec = specificity(hypotheses[k].get("判据", ""))
                p = likelihood(D, k, truth, ct, spec)
                priors = _posterior(priors, D, k, ct, rng.random() < p, spec)
            total += steps
            converged += 1 if ok else 0
    runs = n * trials
    return total / runs, converged / runs


def _policy_cost(priors, cand, hypotheses, D):
    """旧策略：只按检查成本排（实测 > 观察 > 替换）。"""
    order = {"实测": 0, "观察": 1, "替换": 2}
    return sorted(cand, key=lambda k: order.get(check_type(hypotheses[k]), 3))[0]


def _policy_ig(priors, cand, hypotheses, D):
    """新策略：单位成本信息增益最大者优先。"""
    best, best_score = cand[0], -1.0
    for k in cand:
        ct = check_type(hypotheses[k])
        spec = specificity(hypotheses[k].get("判据", ""))
        score = ig_check(priors, D, k, ct, spec) / COST.get(ct, 1.5)
        if score > best_score:
            best, best_score = k, score
    return best


def _group(rows):
    by = {}
    for r in rows:
        by.setdefault(r["code"], []).append({
            "假设": r["cause"], "判据": r["confirm"],
            "verify": r["verify"], "check_type": r["check_type"] or r["verify"]})
    return by


def show_ranking(code, hyps):
    """打印单个故障码的排序表。"""
    n = len(hyps)
    ranked = rank_checks(hyps)
    print()
    print("%s   假设 %d 条 | 均匀先验 %.3f | 先验熵 H0 = %.2f bit"
          % (code, n, 1.0 / n, entropy([1.0 / n] * n)))
    print("  %-3s %-18s %-4s %5s %5s %6s %8s"
          % ("#", "假设", "类型", "成本", "sens", "IG", "IG/成本"))
    for i, (score, ig, cost, h) in enumerate(ranked, 1):
        sens, _ = sens_fp(check_type(h), h.get("判据", ""))
        print("  %-3d %-18s %-4s %5.1f %5.2f %6.3f %8.3f"
              % (i, h["假设"][:16], check_type(h), cost, sens, ig, score))


def compare(codes, trials=200, targets=(0.70, 0.75, 0.80, 0.85, 0.90)):
    """新旧策略对比，并给出收敛阈值敏感性——单点结论容易被"刚好调到 0.8"骗过去。"""
    global CONV_TARGET
    codes = [c for c in codes if len(c[1]) >= 2]
    if not codes:
        return
    print("对比 %d 个故障码（>=2 条原因），每真值 %d 次" % (len(codes), trials))
    print("  阈值 | 旧策略 步/收敛 | 新策略 步/收敛 |  降幅")
    keep = CONV_TARGET
    deltas = []
    try:
        for target in targets:
            CONV_TARGET = target
            old_s = old_c = new_s = new_c = 0.0
            for _, hyps in codes:
                D = dep_matrix(hyps)
                a, ac = _simulate(hyps, _policy_cost, D, trials)
                b, bc = _simulate(hyps, _policy_ig, D, trials)
                old_s += a
                new_s += b
                old_c += ac
                new_c += bc
            k = float(len(codes))
            delta = 100 * (1 - new_s / old_s) if old_s > 0 else 0.0
            deltas.append(delta)
            print("  %.2f | %5.2f %4.0f%% | %5.2f %4.0f%% | %5.1f%%"
                  % (target, old_s / k, 100 * old_c / k, new_s / k, 100 * new_c / k, delta))
    finally:
        CONV_TARGET = keep
    print("  降幅区间 %.1f%%~%.1f%%" % (min(deltas), max(deltas)))
    if max(deltas) < 5:
        print("  ⚠ 各阈值下降幅均不足 5%：说明当前判据区分度低，"
              "收益主要来自「先做便宜检查」，而非「先做信息量大的检查」。")


if __name__ == "__main__":
    import sys

    from csvio import read_rows

    by = _group(read_rows("data/rules.csv"))
    want = sys.argv[1] if len(sys.argv) > 1 else None
    if want:
        for c, h in by.items():
            if c.upper() == want.upper():
                show_ranking(c, h)
    compare(list(by.items()))
