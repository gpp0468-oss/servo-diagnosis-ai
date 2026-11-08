# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# 模块说明
#   职责：Streamlit 四种模式：单轮问答 / 分步诊断 / 归档改善 / 评测
#   依赖：rag_query, kb_category, kb_dedupe, eval_run, openai
#   输入输出：浏览器 UI；不直接读写 indices
#   调试提示：改 UI 后无需重建索引；改数据需重建
# ---------------------------------------------------------------------------
"""工业设备排故 Agent（原型）—— 三种模式。

  单轮问答：输入现象 → 路由 → 检索 → 大模型推理
  分步诊断：按假设表逐步排查、剪枝、给下一步
  归档改善：把现场经验归档，合并进知识库并重建索引
"""
import csv
import os
import re
import subprocess
import sys
from datetime import datetime

import streamlit as st
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from rag_query import Rag, describe  # noqa: E402
from kb_category import CATEGORIES, guess_category  # noqa: E402
from csvio import read_rows  # noqa: E402

st.set_page_config(page_title="工业排故 Agent", page_icon="🛠️", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(ROOT, "data", "rules.csv")
CASES_PATH = os.path.join(ROOT, "data", "cases.csv")
EVAL_PATH = os.path.join(ROOT, "data", "eval.csv")
FEEDBACK_PATH = os.path.join(ROOT, "data", "feedback.csv")


@st.cache_data
def load_rules(path: str = RULES_PATH):
    if not os.path.exists(path):
        return []
    return read_rows(path)


@st.cache_resource
def get_rag():
    try:
        return Rag()
    except Exception:
        return None


RULES = load_rules()

SYSTEM_PROMPT = """你是资深伺服 / PLC 排故专家。请严格按以下步骤输出：

1. 列出 3 个互斥的根因假设；
2. 对每个假设写出「支持证据」与「反驳证据」；
3. 剪掉被证据否定的假设，并说明剪枝理由；
4. 如果证据不足，明确输出「需人工介入」，并列出待核实清单；
5. 最后输出 JSON：{"假设": [], "结论": "", "置信度": 0.0, "下一步": []}

禁止编造现场没有提供的参数。"""


def rows_of(code: str):
    return [r for r in RULES if r["code"] == code]


def rule_route(text: str):
    upper = text.upper()
    for r in RULES:
        if r["code"].upper() in upper:
            return r["code"], rows_of(r["code"])
    for r in RULES:
        base = r["code"].upper().split(".")[0]
        if re.search(base + r"(?!\d)", upper):
            return r["code"], rows_of(r["code"])
    return None, []


def domain_route(text: str) -> str:
    if any(k in text for k in ["过流", "电压", "短路", "绝缘", "烧"]):
        return "电气"
    if any(k in text for k in ["插补", "轨迹", "限位", "回零", "伺服"]):
        return "运动控制"
    if any(k in text for k in ["通讯", "通信", "超时", "丢包", "报文", "EtherCAT"]):
        return "通讯"
    return "其他"


def ask_llm(client, model, symptom, domain, context=""):
    refs = f"\n\n【参考资料】\n{context}" if context else ""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"故障域：{domain}\n现场描述：{symptom}{refs}"}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def ask_plain(client, model, prompt: str) -> str:
    """无系统提示的一句话调用（用于生成假设）。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------- 单轮问答

def render_single(symptom: str):
    code, rows = rule_route(symptom)
    domain = domain_route(symptom)

    st.subheader("① 路由结果")
    c1, c2 = st.columns(2)
    c1.metric("L0 规则命中", code or "未命中")
    c2.metric("L1 故障域", domain)

    if code:
        st.success(f"命中规则库 {code}：{rows[0]['name']}（{len(rows)} 种可能原因），直接返回手册处置，未调用大模型")
        st.dataframe([{"故障原因": r["cause"], "确认方法": r["confirm"],
                       "处理措施": r["action"], "可验证性": r["verify"],
                       "处置类型": r.get("fix_type", "")} for r in rows],
                     use_container_width=True, hide_index=True)
        st.caption(f"来源：SV660N 手册第 16 章 P.{rows[0]['page']}")
        return

    cats = [guess_category(symptom)]
    rag = get_rag()
    hits = rag.search(symptom, 4, cats) if rag else []

    st.subheader("② 检索（RAG）")
    engine = "BM25 + 向量（混合）" if (rag and rag.hybrid) else "BM25"
    st.caption(f"检索引擎：{engine} ｜ 范围：{cats[0]}（同类加权）")

    context = ""
    for i, h in enumerate(hits, 1):
        with st.expander(f"[{i}] {describe(h)} · 得分 {h['score']}"):
            st.write(h["text"][:600])
        context += "[%d]（%s）%s\n\n" % (i, describe(h), h["text"][:400])

    client = client_from_key()
    if client is None:
        st.info("未填写 DeepSeek API Key，已跳过 ③ 模型推理。")
        return

    with st.spinner("推理中…"):
        try:
            answer = ask_llm(client, model_name(), symptom, domain, context)
        except Exception as exc:  # noqa: BLE001
            st.error(f"调用失败：{exc}")
            return

    st.subheader("③ 模型推理")
    st.markdown(answer)


# ---------------------------------------------------------------- 分步诊断

def _new_diag(symptom: str):
    """按当前现象重建假设表（重跑生成逻辑，不复用旧状态）。"""
    from diag import generate_hypotheses  # noqa: PLC0415

    client = client_from_key()
    llm = (lambda p: ask_plain(client, model_name(), p)) if client else None
    code, name, hyps = generate_hypotheses(symptom, RULES, llm)
    gen = st.session_state.get("diag_gen", 0) + 1
    st.session_state.diag_gen = gen
    return {"code": code, "name": name, "rows": hyps, "log": [],
            "symptom": symptom.strip(), "gen": gen}


def render_diag(symptom: str):
    st.caption("按假设表逐步排查：每轮标记一个假设的状态，系统自动剪枝并给出下一步。")

    if "diag" not in st.session_state:
        st.session_state.diag = None

    if st.session_state.diag is None:
        if st.button("开始诊断", type="primary"):
            if not symptom.strip():
                st.warning("先把现象填进来。")
                return
            st.session_state.diag = _new_diag(symptom)
            st.rerun()
        else:
            return

    d = st.session_state.diag
    title = f"{d['code']}（{d['name']}）" if d["code"] else d["name"]
    st.markdown(f"**{title}** —— 共 {len(d['rows'])} 个假设")

    if d.get("symptom") and d["symptom"] != symptom.strip():
        st.warning("输入框里的现象已经改了，点「重新诊断」会用新现象重新生成假设。")

    c1, c2 = st.columns([3, 1])
    if c1.button("用当前现象重新诊断", use_container_width=True):
        if not symptom.strip():
            st.warning("现象是空的。")
        else:
            st.session_state.diag = _new_diag(symptom)
            st.rerun()
    if c2.button("清空", use_container_width=True):
        st.session_state.diag = None
        st.rerun()

    edited = st.data_editor(
        [{"假设": r["假设"], "判据": r["判据"], "状态": r["状态"], "备注": r["备注"]} for r in d["rows"]],
        use_container_width=True, hide_index=True, num_rows="fixed",
        column_config={
            "状态": st.column_config.SelectboxColumn(options=["待验证", "疑似", "排除"], width="small"),
            "判据": st.column_config.TextColumn(width="large", disabled=True),
            "假设": st.column_config.TextColumn(width="large", disabled=True),
        },
        key=f"diag_table_{d.get('gen', 0)}",
    )

    for old, new in zip(d["rows"], edited):
        if old["状态"] != new["状态"]:
            d["log"].append((datetime.now().strftime("%H:%M"), new["假设"], old["状态"], new["状态"]))
    for i, new in enumerate(edited):
        d["rows"][i]["状态"] = new["状态"]
        d["rows"][i]["备注"] = new.get("备注", "")

    pending = [r for r in d["rows"] if r["状态"] == "待验证"]
    suspected = [r for r in d["rows"] if r["状态"] == "疑似"]
    excluded = [r for r in d["rows"] if r["状态"] == "排除"]

    st.subheader("当前推理")
    m1, m2, m3 = st.columns(3)
    m1.metric("待验证", len(pending))
    m2.metric("疑似成立", len(suspected))
    m3.metric("已排除", len(excluded))

    if not pending and len(suspected) == 1:
        st.success(f"收敛：**{suspected[0]['假设']}**")
        st.markdown(f"**处置**：{suspected[0]['action']}")
    elif not pending and not suspected:
        st.error("全部排除：按手册需人工介入或返厂，请记录现场信息。")
    elif pending:
        from diag_rank import rank_checks  # noqa: PLC0415
        ranked = rank_checks([{**r, "check_type": r.get("check_type") or r.get("verify")} for r in pending])
        nxt = ranked[0][3]
        st.info(f"下一步建议：**{nxt['假设']}**\n\n判据：{nxt['判据'] or '（见手册）'}")
        st.caption(f"排序依据：单位成本信息增益（分数 {ranked[0][0]:.3f}，成本 {ranked[0][2]:.1f}）；"
                   f"备选：" + " / ".join("%s(%.3f)" % (r[3]["假设"][:8], r[0]) for r in ranked[1:3]))

    if d["log"]:
        with st.expander("排查记录"):
            for t, h, a, b in d["log"]:
                st.write(f"{t}　{h}：{a} → {b}")

    col_a, col_b = st.columns(2)
    with col_a:
        csv_text = "\n".join(["时间,假设,原状态,新状态"] + [",".join(map(str, x)) for x in d["log"]])
        st.download_button("导出排查记录 CSV", csv_text, file_name="diag_log.csv",
                           mime="text/csv", use_container_width=True)
    with col_b:
        if st.button("归档到经验库", use_container_width=True):
            best = suspected[0] if suspected else None
            ev = "；".join("%s %s→%s" % (t, h[:10], b) for t, h, a, b in d["log"])
            st.session_state.feedback_prefill = {
                "phenomenon": (f"{d['code']} {d['name']}" if d["code"] else d["name"]),
                "cause": best["假设"] if best else "",
                "confirm": best["判据"] if best else "",
                "action": best["action"] if best else "",
                "note": (best.get("备注", "") if best else ""),
                "evidence": ev,
            }
            st.success("已带入「归档改善」（含备注与排查记录），切换模式补充后提交。")


# ---------------------------------------------------------------- 评测

def render_eval():
    st.caption("用 eval.csv 给当前链路打分：规则命中、检索命中、分类准确率。")
    from eval_run import report_md, run_eval  # noqa: PLC0415

    path = EVAL_PATH
    if not os.path.exists(path):
        st.warning("缺少 data/eval.csv。")
        return

    if st.button("运行评测", type="primary"):
        res = run_eval(path, rag=get_rag())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("L0 命中", "%d/%d" % res["l0"])
        c2.metric("检索 Top-1", "%d/%d" % res["top1"])
        c3.metric("检索 Top-3", "%d/%d" % res["top3"])
        c4.metric("分类准确", "%d/%d" % res["cat"])
        st.caption("平均延迟 %.0f ms / 例" % res["lat"])
        st.dataframe(res["details"], use_container_width=True, hide_index=True)
        st.download_button("下载报告 (Markdown)", report_md(res),
                           file_name="eval_report.md", mime="text/markdown")


# ---------------------------------------------------------------- 归档改善

def render_feedback():
    st.caption("把现场验证过的经验归档；审核通过后可合并进知识库并重建索引。")

    prefill = st.session_state.get("feedback_prefill", {})
    with st.form("fb"):
        c1, c2 = st.columns(2)
        phenomenon = c1.text_input("现象（问题描述）", value=prefill.get("phenomenon", ""))
        category = c2.selectbox("故障大类", CATEGORIES,
                                index=CATEGORIES.index(guess_category(phenomenon)) if phenomenon else 0)
        cause = st.text_input("根因", value=prefill.get("cause", ""))
        confirm = st.text_input("确认方法（手册判据）", value=prefill.get("confirm", ""))
        action = st.text_input("处理措施", value=prefill.get("action", ""))
        note = st.text_area("备注（现场证据，最有价值）", value=prefill.get("note", ""), height=80)
        evidence = st.text_area("排查记录（分步诊断自动带入）",
                                value=prefill.get("evidence", ""), height=70)
        use_note = st.checkbox("用备注优化确认方法（把实际验证手段写进判据）", value=True)
        c3, c4 = st.columns(2)
        verify = c3.selectbox("可验证性", ["实测", "观察", "替换"])
        source = c4.text_input("来源", value="现场经验")
        if st.form_submit_button("提交归档", type="primary"):
            if not (phenomenon and cause and action):
                st.warning("现象、根因、处理措施至少三项要填。")
            else:
                final_confirm = confirm
                if use_note and note.strip():
                    final_confirm = note.strip() + ("；" + confirm if confirm else "")
                entry = {"phenomenon": phenomenon, "cause": cause,
                         "action": action, "confirm": final_confirm}
                from kb_dedupe import check_entry  # noqa: PLC0415
                issues = check_entry(entry, _read_cases(), _read_feedback())
                payload = (phenomenon, category, cause, final_confirm, action, verify,
                           source, note, evidence)
                if issues:
                    st.session_state.pending_fb = {"entry": payload, "issues": issues}
                else:
                    _append_feedback(*payload)
                    st.success("已归档，状态为「待审核」（未发现重复）。")

    pending = st.session_state.get("pending_fb")
    if pending:
        st.warning("查重发现相似或冲突的条目，请确认后再入库：")
        st.dataframe(
            [{"级别": lvl, "来源": src, "现象": old.get("phenomenon", ""), "根因": old.get("cause", ""),
              "说明": msg} for lvl, src, old, msg in pending["issues"]],
            use_container_width=True, hide_index=True,
        )
        c_ok, c_no = st.columns(2)
        with c_ok:
            if st.button("确认是新经验，仍然归档", type="primary", use_container_width=True):
                _append_feedback(*pending["entry"])
                st.session_state.pending_fb = None
                st.success("已强制归档（标记为新经验）。")
                st.rerun()
        with c_no:
            if st.button("放弃这次归档", use_container_width=True):
                st.session_state.pending_fb = None
                st.rerun()

    rows = _read_feedback()
    if not rows:
        st.info("还没有归档记录。")
        return

    pending_n = sum(1 for r in rows if r["status"] == "待审核")
    st.subheader(f"待审核（{pending_n}）")
    edited = st.data_editor(
        rows, use_container_width=True, hide_index=True,
        column_config={
            "通过": st.column_config.CheckboxColumn(width="small"),
            "status": st.column_config.SelectboxColumn(options=["待审核", "已通过", "已应用"], width="small"),
        },
        disabled=["created", "phenomenon", "category", "cause", "confirm", "action", "verify", "source"],
        key="fb_table",
    )
    if st.button("保存审核结果"):
        _save_feedback(edited)
        st.success("已保存。")

    st.divider()
    st.subheader("应用并重建")
    st.caption("把「已通过」的条目合并进 cases.csv，并重建检索索引（约 30 秒）。")
    if st.button("应用并重建索引", type="primary"):
        with st.spinner("合并并重建中…"):
            try:
                out = subprocess.run([sys.executable, os.path.join(ROOT, "kb_apply.py")],
                                     capture_output=True, text=True, timeout=900, cwd=ROOT)
                st.code((out.stdout or "")[-1500:] + ("\n" + out.stderr[-800:] if out.stderr else ""))
                if out.returncode == 0:
                    st.success("完成：cases.csv 已更新，索引已重建。")
                    get_rag.clear()
                else:
                    st.error("重建失败，见上方输出。")
            except Exception as exc:  # noqa: BLE001
                st.error(f"执行失败：{exc}")


def _append_feedback(phenomenon, category, cause, confirm, action, verify, source,
                     note="", evidence=""):
    new = not os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["created", "phenomenon", "category", "cause", "confirm", "action",
                        "verify", "note", "evidence", "source", "status", "通过"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), phenomenon, category,
                    cause, confirm, action, verify, note, evidence, source, "待审核", ""])


def _read_cases():
    p = CASES_PATH
    if not os.path.exists(p):
        return []
    return read_rows(p)


def _read_feedback():
    if not os.path.exists(FEEDBACK_PATH):
        return []
    return read_rows(FEEDBACK_PATH)


def _save_feedback(rows):
    with open(FEEDBACK_PATH, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["created", "phenomenon", "category", "cause",
                                           "confirm", "action", "verify", "note", "evidence",
                                           "source", "status", "通过"])
        w.writeheader()
        w.writerows(rows)


def client_from_key():
    key = st.session_state.get("api_key", "").strip()
    return OpenAI(api_key=key, base_url="https://api.deepseek.com") if key else None


def model_name():
    return st.session_state.get("model", "deepseek-chat")


# ---------------------------------------------------------------- 主界面

st.title("工业设备排故 Agent（原型）")

with st.sidebar:
    st.subheader("配置")
    st.text_input("DeepSeek API Key", type="password", key="api_key",
                  value=os.getenv("DEEPSEEK_API_KEY", ""))
    st.selectbox("模型", ["deepseek-chat", "deepseek-reasoner"], key="model")
    if st.button("测试连接", use_container_width=True):
        c = client_from_key()
        if c is None:
            st.warning("先填入 key")
        else:
            try:
                c.models.list()
                st.success("连接正常")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc)[:160])
    st.divider()
    fb_count = len(_read_feedback()) if os.path.exists(FEEDBACK_PATH) else 0
    st.markdown(f"- 规则：{len(set(r['code'] for r in RULES))} 个码 / {len(RULES)} 条原因\n"
                f"- 检索：BM25 + 向量\n- 经验库：{fb_count} 条")

mode = st.radio("模式", ["单轮问答", "分步诊断", "归档改善", "评测"], horizontal=True)

symptom_box = ""
if mode in ("单轮问答", "分步诊断"):
    symptom_box = st.text_area("故障现象 / 报警 / 参数", height=120,
                               placeholder="例：伺服报 E201.0 过流，空载仍报，三相电阻平衡")

if mode == "单轮问答":
    if st.button("开始分析", type="primary"):
        if symptom_box.strip():
            render_single(symptom_box)
        else:
            st.warning("先把现象填进来。")
elif mode == "分步诊断":
    render_diag(symptom_box)
elif mode == "评测":
    render_eval()
else:
    render_feedback()

st.divider()
st.caption("链路：L0 规则 → L1 域分类 → 混合检索 → DeepSeek 结构化生成（三假设 + 证据 + 剪枝） → 人工确认")


