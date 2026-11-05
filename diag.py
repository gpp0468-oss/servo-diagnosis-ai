# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 模块说明
#   职责：从「现象」生成分步诊断的假设表（无故障码时也能用）
#   依赖：kb_dedupe.jaccard, kb_category.guess_category
#   输入输出：generate_hypotheses(symptom, rules, llm=None) -> (code, name, rows)
#   调试提示：先看返回的 source 字段（rules / keyword / generic），判断命中来源
# ---------------------------------------------------------------------------
import json
import re

from kb_category import guess_category
from kb_dedupe import jaccard

# 关键词 → 专用假设（覆盖高频现象，优先于通用模板）
# 注意：抱闸是「失电制动（fail-safe）」——常态弹簧抱死，通 24V 才松开。
# 所以「抱不住 / 下坠」类现象，要查的是「该闭合时没闭合或闭合太慢」，不是「没通电」。
KEYWORD_HYPOTHESES = {
    "抱闸": [
        ("抱闸闭合延时过长：STO 触发后抱闸还没夹紧，负载先下坠",
         "测 STO 触发到抱闸实际闭合的时间，量下坠距离", "实测",
         "缩短抱闸闭合延时（参数/回路），或增加第二重制动"),
        ("抱闸保持力矩不足（闸片磨损、油污、间隙过大）",
         "断电后手动加载测保持力矩，检查闸片磨损与气隙", "实测",
         "更换闸片/调整气隙，或更换更大保持力矩的抱闸"),
        ("抱闸回路未断电（继电器粘连、DO 常输出），抱闸一直处于松开状态",
         "STO 触发瞬间测抱闸端子电压是否降到 0V", "实测",
         "更换继电器，修正 DO 逻辑，确保 STO 时切断抱闸供电"),
        ("抱闸时序参数设置不当（提前松开/延后夹紧）",
         "核对抱闸开合时序参数与 STO 动作顺序", "观察",
         "按手册重新设定抱闸时序"),
        ("垂直轴负载超过抱闸额定保持力矩（选型或减速比不足）",
         "计算重力矩，与抱闸额定保持力矩对比", "观察",
         "加大抱闸规格、调整减速比或加配重"),
    ],
    "啸叫": [
        ("速度环增益过高", "降增益后是否消失", "观察", "降低速度环增益并重做整定"),
        ("机械共振", "特定转速才响，做 FFT", "实测", "加陷波滤波器"),
        ("载波频率偏低", "提高载频看噪音变化", "观察", "适当提高载频"),
    ],
    "抖动": [
        ("增益/惯量比不匹配", "做惯量辨识，看电流波形", "观察", "重新整定"),
        ("总线周期抖动或丢帧", "看工作计数器与周期抖动", "实测", "对齐周期、改善屏蔽"),
        ("机械连接松动", "手动检查联轴器/皮带", "观察", "紧固或更换"),
    ],
    "定位": [
        ("电子齿轮比/脉冲当量错误", "走固定距离实测", "实测", "重新计算并写入"),
        ("机械反向间隙", "打表测正反向差值", "实测", "调整机械或补偿"),
        ("位置环增益偏低", "提高增益看跟随误差", "观察", "提高位置环增益"),
    ],
}

GENERIC = [
    ("机械卡滞或负载异常", "脱开负载手动盘车，看是否卡滞", "观察", "解除卡滞或减载"),
    ("接线/线缆不良", "检查动力线与编码器线插头、屏蔽", "观察", "重新插接或换线"),
    ("参数与指令不匹配", "核对控制模式、限位、转矩限制", "观察", "按手册修正参数"),
    ("驱动或电机本体异常", "交叉测试：换驱动器/电机", "替换", "更换故障件"),
]

# 口语别名 → 复用同一组假设
_ALIAS = {"嗡嗡": "啸叫", "尖叫": "啸叫", "异响": "啸叫", "噪音": "啸叫",
          "跑偏": "定位", "走不到位": "定位", "过冲": "定位",
          "震动": "抖动", "振荡": "抖动"}
for _a, _k in _ALIAS.items():
    if _k in KEYWORD_HYPOTHESES:
        KEYWORD_HYPOTHESES.setdefault(_a, KEYWORD_HYPOTHESES[_k])

LLM_PROMPT = """你是伺服排故专家。针对下面的现场现象，给出 3-4 个互斥的根因假设。
只输出 JSON 数组，不要解释，格式：
[{"假设":"...","判据":"怎么验证...","处置":"..."}]
现象："""


def _from_rules(symptom, rules, limit=5):
    scored = []
    for r in rules:
        s = jaccard(symptom, r["name"] + r["cause"])
        if s >= 0.18:
            scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, r in scored:
        if r["cause"] in seen:
            continue
        seen.add(r["cause"])
        out.append({"假设": r["cause"], "判据": r["confirm"], "状态": "待验证",
                    "备注": "", "verify": r["verify"], "action": r["action"],
                    "source": "rules"})
        if len(out) >= limit:
            break
    return out


def _from_keyword(symptom):
    for key, items in KEYWORD_HYPOTHESES.items():
        if key in symptom:
            return [{"假设": a, "判据": b, "状态": "待验证", "备注": "",
                     "verify": v, "action": act, "source": "keyword"} for a, b, v, act in items]
    return []


def _from_generic(symptom):
    return [{"假设": a, "判据": b, "状态": "待验证", "备注": "",
             "verify": v, "action": act, "source": "generic"} for a, b, v, act in GENERIC]


def _from_llm(symptom, llm, limit=4):
    try:
        text = llm(LLM_PROMPT + symptom)
        m = re.search(r"\[.*\]", text, re.S)
        data = json.loads(m.group(0)) if m else []
        out = []
        for d in data[:limit]:
            verdict = "实测" if any(k in d.get("判据", "") for k in ["测", "看波形", "读"]) else "观察"
            out.append({"假设": d["假设"], "判据": d.get("判据", ""), "状态": "待验证",
                        "备注": "", "verify": verdict, "action": d.get("处置", ""),
                        "source": "llm"})
        return out
    except Exception:  # noqa: BLE001
        return []


def generate_hypotheses(symptom: str, rules, llm=None):
    """返回 (code, name, rows)。code 为空表示没有命中故障码。"""
    upper = symptom.upper()
    for r in rules:
        if r["code"].upper() in upper:
            rows = [x for x in rules if x["code"] == r["code"]]
            return r["code"], r["name"], [
                {"假设": x["cause"], "判据": x["confirm"], "状态": "待验证", "备注": "",
                 "verify": x["verify"], "action": x["action"], "source": "rules"} for x in rows]

    rows = _from_rules(symptom, rules)
    if len(rows) < 3:
        kw = _from_keyword(symptom)
        if kw:
            rows = kw + [r for r in rows if r["假设"] not in {k["假设"] for k in kw}]

    if len(rows) < 3 and llm is not None:
        rows = _from_llm(symptom, llm) or rows

    if not rows:
        rows = _from_generic(symptom)

    return "", "现象诊断：%s（%s）" % (symptom[:18], guess_category(symptom)), rows[:5]
