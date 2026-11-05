# -*- coding: utf-8 -*-
"""Generate an FTA (fault tree analysis) document from rules.csv."""
import csv
from csvio import read_rows
import os
from collections import Counter, defaultdict

SRC = "data/rules_updated.csv" if os.path.exists("data/rules_updated.csv") else "data/rules.csv"
CASES = "data/cases.csv"
OUT = "SV660N故障树分析.md"

rows = read_rows(SRC)
by_code = defaultdict(list)
for r in rows:
    by_code[r["code"]].append(r)

by_cat = defaultdict(list)
for code, rs in by_code.items():
    by_cat[rs[0]["category"]].append((code, rs))

order = ["电源与功率", "运动控制", "编码器与反馈", "通讯与总线",
         "参数与配置", "电机与机械", "安全与抱闸", "其他"]
order = [c for c in order if c in by_cat]

lines = []
lines.append("# SV660N 故障树分析（FTA）")
lines.append("")
lines.append("> 顶事件：SV660N 伺服驱动系统「功能失效」——表现为停机、失控、精度不达标或无法启动。")
lines.append("> 数据来源：`rules.csv`（手册第 16 章，247 条原因）＋ 手册 16.6/16.8 ＋《伺服驱动器原理学习笔记》。")
lines.append("> 生成方式：按故障大类分组，每组取原因最多的故障码展开；叶节点标注可验证性。")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 1. 顶事件与一级分解")
lines.append("")
lines.append("本树用 **OR 门**（任一分支发生即导致顶事件）。一级分支按故障大类划分：")
lines.append("")
lines.append("```")
lines.append("T  SV660N 功能失效")
for i, cat in enumerate(order):
    n_codes = len(by_cat[cat])
    n_rows = sum(len(rs) for _, rs in by_cat[cat])
    last = i == len(order) - 1
    lines.append("%s %s（%d 个故障码 / %d 条原因）" % ("└─" if last else "├─", cat, n_codes, n_rows))
lines.append("```")
lines.append("")
lines.append("| 分支 | 故障码数 | 原因条数 | 典型代表 |")
lines.append("|---|---|---|---|")
for cat in order:
    codes = by_cat[cat]
    top = sorted(codes, key=lambda x: -len(x[1]))[:2]
    rep = "、".join("%s %s" % (c, rs[0]["name"]) for c, rs in top)
    lines.append("| %s | %d | %d | %s |" % (cat, len(codes), sum(len(rs) for _, rs in codes), rep))
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 2. 各分支展开")
lines.append("")

for cat in order:
    codes = sorted(by_cat[cat], key=lambda x: (-len(x[1]), x[0]))
    total = sum(len(rs) for _, rs in codes)
    lines.append("### %s（%d 条原因）" % (cat, total))
    lines.append("")
    show = codes[:8]
    if len(codes) > 8:
        lines.append("> 共 %d 个故障码，下表列出原因最多的 8 个。「可验证性」为「替换」的条目意味着**没有判断依据、只能换件**，排故时应放在最后。" % len(codes))
        lines.append("")
    lines.append("| 故障码 | 名称 | 典型原因（节选） | 可验证性 | 手册页 |")
    lines.append("|---|---|---|---|---|")
    for code, rs in show:
        causes = [r["cause"] for r in rs if r["cause"] and not r["cause"].startswith("（手册未拆分")]
        cause_txt = "；".join(causes[:3]) if causes else "（见手册）"
        if len(cause_txt) > 60:
            cause_txt = cause_txt[:60] + "…"
        ver = Counter(r["verify"] for r in rs).most_common(1)[0][0]
        lines.append("| %s | %s | %s | %s | P.%s |"
                     % (code, rs[0]["name"], cause_txt, ver, rs[0].get("page", "-")))
    lines.append("")

lines.append("---")
lines.append("")
lines.append("## 3. 定性最小割集：单点就能导致停机的事件")
lines.append("")
lines.append("以下基本事件单独发生即可让设备停机或失控，属于**单点故障**，应优先设计预防措施：")
lines.append("")
lines.append("| 基本事件 | 所属分支 | 为什么是单点 |")
lines.append("|---|---|---|")
singles = [
    ("UVW 相序接错 / 缺相", "电源与功率", "换向错误→飞车或过流，无冗余"),
    ("编码器线断 / 干扰", "编码器与反馈", "反馈丢失→位置失控"),
    ("STO 输入断开", "安全与抱闸", "安全链动作→直接切转矩"),
    ("参数密码误设（H02.30=1）", "参数与配置", "面板锁死，无法改参数"),
    ("软限位/超程触发", "运动控制", "指令仍在发→位置溢出"),
    ("制动电阻未接 / 阻值过小", "电源与功率", "减速能量无处释放→母线过压"),
    ("抱闸未打开", "电机与机械", "堵转→过载/过热"),
]
for a, b, c in singles:
    lines.append("| %s | %s | %s |" % (a, b, c))
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 4. 与 rules.csv 的对应关系")
lines.append("")
lines.append("- `category` 列 ↔ 本树一级分支；")
lines.append("- `code` 列 ↔ 叶节点的故障码；")
lines.append("- `cause` 列 ↔ 基本事件描述；")
lines.append("- `verify` / `closed_loop` 列 ↔ 叶节点的「可验证性」标注；")
lines.append("- `page` 列 ↔ 手册原文位置，用于回查。")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 5. 排故时怎么用这棵树")
lines.append("")
lines.append("1. **定位分支**：先看现象属于哪一类（上电类→电源与功率；走不准→运动控制/编码器；通讯断→通讯与总线）；")
lines.append("2. **剪枝**：按「可验证性=实测 → 观察 → 替换」的顺序逐条排除，能测的先测；")
lines.append("3. **禁止跳步**：直接换驱动器（替换类）必须在交叉测试与器件级测量之后；")
lines.append("4. **回流**：每次现场确认的根因，回写到 rules.csv 的 `cause`，树会自动长出来。")
lines.append("")
lines.append("> 本树是「结构性」的，不做概率计算。若要定量（求顶事件概率），需要每个基本事件的失效率 λ，手册不提供，只能用现场统计。")

# ---- section 6: EtherCAT-induced phenomena -------------------------------
if os.path.exists(CASES):
    cases = read_rows(CASES)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. 专题：EtherCAT 设置引起的三类现象")
    lines.append("")
    lines.append("有些「偶发」问题不在伺服本体，而在总线的**时间基准**上。下面三类现象都必须先排除总线因素，再去调增益。")
    lines.append("")
    lines.append("### 6.1 逻辑链")
    lines.append("")
    lines.append("```")
    lines.append("总线因素 ──► 指令/采样时刻不均匀 ──► 速度环看到「假误差」")
    lines.append("   │                 │                      │")
    lines.append("   │                 ├─► 位置指令单周期跳变 ──► 位置偏差过大/超差")
    lines.append("   │                 ├─► 周期性指令阶梯 ──────► 激发机械共振")
    lines.append("   │                 └─► 采样抖动 ───────────► 速度抖动 / 电流波动 → 啸叫异响")
    lines.append("   └─ 丢帧 / DC 失锁 / 看门狗 各自对应不同 AL Status Code")
    lines.append("```")
    lines.append("")
    lines.append("### 6.2 三类现象的总线侧原因")
    lines.append("")
    lines.append("| 现象 | 总线侧原因 | 怎么确认 | 怎么处理 |")
    lines.append("|---|---|---|---|")
    seen = set()
    for c in cases:
        key = (c["phenomenon"], c["cause"])
        if key in seen:
            continue
        seen.add(key)
        lines.append("| %s | %s | %s | %s |"
                     % (c["phenomenon"], c["cause"], c["confirm"], c["action"]))
    lines.append("")
    lines.append("### 6.3 排查顺序（先总线、后参数、再机械）")
    lines.append("")
    lines.append("1. **看总线是否健康**：工作计数器是否增长、丢帧计数、AL Status Code（0x0130/0x0134）；")
    lines.append("2. **看时间是否同步**：DC 是否启用、SYNC0 周期、系统时间差（0x092C）；")
    lines.append("3. **看周期是否对齐**：主站任务周期与总线周期是否整数倍；")
    lines.append("4. **再调参数**：惯量辨识 → 增益 → 前馈；")
    lines.append("5. **最后看机械**：共振频率、联轴器、皮带。")
    lines.append("")
    lines.append("> 顺序反了会怎样：先调增益，可能把总线抖动放大成共振；先换机械，问题依旧——因为时间基准没解决。")
lines.append("")
lines.append("### 6.4 通讯故障的分层定位（H0E.24~29）")
lines.append("")
lines.append("| 层 | 参数 | 能定位的问题 |")
lines.append("|---|---|---|")
lines.append("| 物理层 | H0E.29 端口0链接丢失 | 网线拔插/接头松/断线（EE08.3） |")
lines.append("| 链路层 | H0E.25/26 端口无效帧及错误 | 某端口坏帧（CRC），区分 IN/OUT 定上游还是下游 |")
lines.append("| 转发 | H0E.27 端口转发错误 | 帧经本从站转发出错 |")
lines.append("| 从站内部 | H0E.28 数据帧处理单元错误 | ESC 处理帧出错 |")
lines.append("| 同步层 | H0E.24 同步丢失次数 | 帧太慢/丢帧，SYNC0 已触发但数据没到 |")
lines.append("")
lines.append("> **跨域提醒**：通讯问题不只在通讯层表现，会伪装成运动控制/机械现象——")
lines.append("> 位置指令增量过大（主站任务周期与总线周期不整除）、速度抖动（DC 失锁/周期抖动）、")
lines.append("> 异响啸叫（同步抖动→电流波动）。遇到这三类，先按 §6.4 分层查通讯，再调增益。")


lines.append("")
lines.append("## 7. 现象级故障树（无码，重构版）")
lines.append("")
lines.append("> 树从**现象**进入，每一步是一个可验证的问题，落到具体参数和判据。")
lines.append("> 树根三叉：**配置/参数错（能调）、硬件坏（能换）、选型越界（只能改型号或工况）**。")
lines.append("")
lines.append("### 7.1 精度类：定位不准 / 轨迹偏移 / 重复精度差")
lines.append("")
lines.append("```")
lines.append("① 走固定距离，偏差是否成比例？")
lines.append("   ├─ 是 → 指令链量值错 → H05.07/09 齿轮比、H05.50/51（判据：偏差≈减速比）")
lines.append("   └─ 否 → ② 提高增益，跟随误差是否下降？")
lines.append("          ├─ 是 → H08.02 位置环增益低")
lines.append("          ├─ 否 → ③ 多轴插补时单轴是否都准？")
lines.append("          │        ├─ 是 → 两轴一致性 H08.19/21 前馈、加减速、背隙")
lines.append("          │        └─ 否 → ④ 反向打表，正反向差值？")
lines.append("          │                 ├─ 大 → 机械反向间隙")
lines.append("          │                 └─ 小 → 干扰/噪声（编码器屏蔽、接地）")
lines.append("          └─ 通讯侧：主站周期与总线周期不整除 → 位置指令增量过大（先查 §6）")
lines.append("```")
lines.append("")
lines.append("### 7.2 动作类：不动 / 飞车 / 撞限位")
lines.append("")
lines.append("```")
lines.append("① 看 6041h 状态字，是否 Operation enabled？")
lines.append("   ├─ 否 → 使能链断 → H02.00 模式、急停、STO、H03 DI 条件")
lines.append("   └─ 是 → ② 抱闸是否松开？（失电制动，通 24V 才松）")
lines.append("          ├─ 未松 → H02.09~12 抱闸时序、H04 DO、继电器")
lines.append("          ├─ 已松 → ③ 方向是否反？")
lines.append("          │        ├─ 反 → H02.02 旋转方向（判据：正向指令 CCW）")
lines.append("          │        └─ 正 → ④ 限位是否触发？")
lines.append("          │                 ├─ 是 → 限位开关/逻辑 H03 取反")
lines.append("          │                 └─ 否 → 飞车保护 H0A.12/55/57/58/59")
lines.append("          └─ 通讯侧：总线未进 OP、PDO 映射错（先查 §6）")
lines.append("```")
lines.append("")
lines.append("### 7.3 波动/异响类：抖动 / 啸叫 / 振动")
lines.append("")
lines.append("```")
lines.append("① 脱开负载，现象是否消失？")
lines.append("   ├─ 消失 → 机械侧：联轴器、皮带、减速箱背隙")
lines.append("   └─ 仍在 → ② 特定转速才响？")
lines.append("          ├─ 是 → 机械共振 → H09.23 陷波器 / H08.35 振动抑制")
lines.append("          └─ 否 → ③ 降增益是否消失？")
lines.append("                 ├─ 是 → H08.00/02 增益过高")
lines.append("                 └─ 否 → ④ 惯量比是否正确？")
lines.append("                        ├─ 否 → H08.15 惯量辨识")
lines.append("                        └─ 通讯侧 → DC 失锁/周期抖动（先查 §6）")
lines.append("```")
lines.append("")
lines.append("### 7.4 功能类：按第 11 章功能失效")
lines.append("")
lines.append("| 功能 | 失效表现 | 先查参数 | 判据 |")
lines.append("|---|---|---|---|")
lines.append("| 控制模式 | 指令在变但不动 | H02.00 | 上位机与伺服模式一致 |")
lines.append("| 转换因子/齿轮比 | 距离成比例偏差 | H05.07/09 | 走固定距离实测 |")
lines.append("| 状态机/使能 | 卡在未使能 | H02/H03 | 6040h/6041h |")
lines.append("| 回零 | 回零失败/原点漂 | H05 | 6098h 方式、原点开关、Z 信号 |")
lines.append("| 绝对值保持 | 断电丢原点 | H00/H0d | 电池、标志是否被覆盖 |")
lines.append("| 转矩/速度限制 | 带载无力/误限 | H06/H07 | 限制源、限制值 |")
lines.append("| 自调整 | 整定失败/越调越振 | H09 | 刚性等级、惯量比 |")
lines.append("| 振动抑制 | 特定转速共振 | H08/H09 | 陷波器、低频抑制 |")
lines.append("| STO | 触发不停机 | H03 | 双通道接线、状态字 |")
lines.append("| DO/抱闸 | 抱闸不松/不抱 | H04 | 功能分配、回路电压 |")
lines.append("| 探针/位置比较 | 抓不到位置 | H18 | 比较源、触发沿 |")
lines.append("")
lines.append("> 功能类先查**配置对不对**，再查**硬件坏没坏**。")
lines.append("")
lines.append("### 7.5 选型越界（第三叉，先于一切参数排查）")
lines.append("")
lines.append("| 越界信号 | 判断 | 依据 |")
lines.append("|---|---|---|")
lines.append("| 减速过压、内置电阻过载 | 惯量比超限 | 19.1.5 |")
lines.append("| 长期过载报警 | 容量/工作制选小 | 19.1.2 |")
lines.append("| 高速带不动 | 弱磁未开或转速等级选错 | H07.26、型号④ |")
lines.append("| 电机过热 | 环境高温/海拔高未降额 | 19.1.1 |")
lines.append("| 轴端漏油 | 该用带油封机型 | 3.2.1⑧ |")
lines.append("")
lines.append("> **选型越界靠调参压不住**，只能改型号或改工况；先分清楚再动手。")
lines.append("")

lines.append("")
lines.append("## 8. 排故速查：现象 → 机理 → 参数 → 判据")
lines.append("")
lines.append("| 现象 | 机理 | 先查参数 | 判据 |")
lines.append("|---|---|---|---|")
lines.append("| 定位不准/距离不对 | 指令链量值错 | H05.07/09、H05.50/51 | 走固定距离，偏差成比例=齿轮比错 |")
lines.append("| 到位后抖动/啸叫 | 增益过高或共振 | H08.00/02、H09.23 | 降增益是否消失；FFT 找共振 |")
lines.append("| 减速报过压/过载 | 惯量比超限（选型） | H08.15、H02.25 | 降减速/降转速/外置电阻是否解除 |")
lines.append("| 回零超时 E601 | 原点搜索超时 | H05.35 | 增大搜索时间或查原点开关 |")
lines.append("| STO 后下坠 | 抱闸时序（失电制动） | H02.10 | 测 STO→抱闸闭合时间与下坠距离 |")
lines.append("| 点动无动作 | 使能链/抱闸未松 | H02.00、H02.09~12、H03 | 看 6041h 状态字 |")
lines.append("| 撞限位/飞车 | 方向/限位逻辑/保护 | H02.02、H03、H0A.57 | DI 状态、方向 |")
lines.append("| 指令在变电机不动 | 模式不匹配/齿轮比0 | H02.00、H05.07 | 核对上位机与伺服模式 |")
lines.append("| 断电丢原点 | 绝对值保持失效 | H02.01、电池、H0d.20 | 电池电压、标志被覆盖 |")
lines.append("| 通讯掉站/丢帧 | DC 同步或链路 | H0E.24~29 | 按 29→24 分层定位 |")
lines.append("| 过流 E201 | 增益/接线/驱动/制动电阻 | H08、接线、H02.25 | 规则库四因逐一排除 |")
lines.append("| 带载无力 | 转矩限制 | H07.09/10 | 查限制值 |")
lines.append("")
lines.append("> 先问一句：是配置/参数问题，还是选型越界？越界靠调参压不住。")
lines.append("")

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("written", OUT, len(lines), "lines")
