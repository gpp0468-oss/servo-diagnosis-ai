# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 模块说明
#   职责：B 任务——按「故障源」原则对 rules.csv 逐行重分类（依据来自学习笔记）
#         1) 先查参数号 → 按 §9 参数组映射（H0E=通讯，其余 H 组=参数与配置）
#         2) 再查机理关键词 → 按 §15 接线 / §16 面板 / §12 指令链
#         3) 兜底 → 其他
#   输入输出：读 rules.csv → 写 reclassify_review.csv（含「命中依据」列）
#   调试提示：只输出对照表，不回写 rules.csv；确认后再改
# ---------------------------------------------------------------------------
import csv
import re
from collections import Counter

from csvio import read_rows

# §9 参数组 → 故障源（H0E 是通讯，H30/H31 是通讯读写）
PARAM_MAP = {
    "H00": "参数与配置", "H01": "参数与配置", "H02": "参数与配置",
    "H03": "参数与配置", "H04": "参数与配置", "H05": "参数与配置",
    "H06": "参数与配置", "H07": "参数与配置", "H08": "参数与配置",
    "H09": "参数与配置", "H0A": "参数与配置", "H0B": "参数与配置",
    "H0C": "参数与配置", "H0D": "参数与配置", "H0E": "通讯与总线",
    "H16": "参数与配置", "H17": "参数与配置", "H18": "参数与配置",
    "H19": "参数与配置", "H30": "通讯与总线", "H31": "通讯与总线",
}
PARAM_RE = re.compile(r"(H0[0-9A-Ea-e]|H1[6-9]|H3[01])\.\d{1,2}", re.I)

# 关键词兜底（顺序即优先级）：依据 §15 接线、§16 面板、§12 指令链、§14 选型
KEYWORDS = [
    ("驱动器本体", ["MCU", "FPGA", "加密芯片", "固件", "控制板", "PHY",
                    "芯片", "协处理器", "HOST", "版本号", "烧录"]),
    ("通讯与总线", ["上位机", "主站", "报文", "总线", "EtherCAT", "同步", "丢帧",
                    "看门狗", "站号", "网络", "网线", "从站", "PDO", "SDO", "数据帧",
                    "邮箱", "分布式时钟", "XML", "EEPROM中EtherCAT"]),
    ("编码器与反馈", ["编码器", "光栅", "旋变", "分辨率"]),
    ("安全与抱闸", ["抱闸", "STO", "急停", "安全", "制动器", "刹车"]),
    ("电源与功率", ["IGBT", "电容", "母线", "软启", "整流", "制动电阻", "MOS",
                    "保险丝", "继电器", "变压器", "电源", "短路", "绝缘", "漏电",
                    "缺相", "过压", "欠压", "泄放", "相电流", "过流", "过载"]),
    ("电机与机械", ["接线", "线缆", "屏蔽", "接地", "端子", "插头", "联轴器",
                    "减速箱", "负载", "轴承", "皮带", "丝杠", "导轨", "齿轮", "限位开关",
                    "磁环", "布线", "紧固", "机械", "卡", "摩擦", "堵转", "油封",
                    "风扇", "安装方向", "间隔", "环境温度"]),
    ("参数与配置", ["参数", "设置", "设定", "增益", "齿轮比", "惯量", "刚性", "滤波",
                    "阈值", "整定", "限幅", "模式", "使能", "前馈", "密码", "初始化",
                    "功能码", "电子齿轮", "功能选择", "端子功能", "停机方式", "无效",
                    "定标", "功能编号", "回零方式", "原点回归", "查找原点"]),
    ("运动控制", ["插补", "跟随", "定位", "轨迹", "振荡", "响应", "飞车", "超程",
                  "偏差", "指令", "速度", "转矩", "位置"]),
]


# 特定词优先：避免「齿轮」吃掉「齿轮比」、「缺相」吃掉主题
OVERRIDES = [
    ("齿轮比", "参数与配置"),
    ("电子齿轮", "参数与配置"),
    ("啸叫", "电机与机械"),
    ("异响", "电机与机械"),
    ("嗡嗡", "电机与机械"),
    ("面板", "电源与功率"),
    ("不显示", "电源与功率"),
    ("上位机", "通讯与总线"),
]


def classify(cause: str, name: str):
    """返回 (类别, 命中依据)。"""
    text = name + " " + cause
    m = PARAM_RE.search(text)
    if m:
        group = m.group(1).upper()
        if group in PARAM_MAP:
            return PARAM_MAP[group], "参数号 " + group
    for k, cat in OVERRIDES:
        if k in text:
            return cat, "特定词 " + k
    for cat, kws in KEYWORDS:
        for k in kws:
            if k in text:
                return cat, "关键词 " + k
    return "其他", "未命中"


def main():
    rows = read_rows("data/rules.csv")
    changed = 0
    for r in rows:
        cat, basis = classify(r["cause"], r["name"])
        r["_new"] = cat
        r["_basis"] = basis
        if cat != r["category"]:
            changed += 1

    print("总行数:", len(rows), "| 分类变化:", changed)
    print("\n新分类分布:")
    for k, v in Counter(r["_new"] for r in rows).most_common():
        print("  %-8s %3d" % (k, v))

    print("\n命中方式:", dict(Counter(r["_basis"].split()[0] for r in rows)))

    with open("reclassify_review.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "name", "cause", "旧分类", "新分类", "命中依据", "变更"])
        for r in rows:
            w.writerow([r["code"], r["name"], r["cause"], r["category"],
                        r["_new"], r["_basis"],
                        "改" if r["_new"] != r["category"] else ""])
    print("\n对照表: reclassify_review.csv")


if __name__ == "__main__":
    main()
