# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 模块说明
#   职责：增强审计 rules.csv —— 检测「串码」（同一码混入多套原因组）与「未拆分」
#   输入输出：读 rules.csv → 输出可疑清单
# ---------------------------------------------------------------------------
import re
from collections import defaultdict

from csvio import read_rows

rows = read_rows("data/rules.csv")
by = defaultdict(list)
for r in rows:
    by[r["code"]].append(r["cause"])

print("===== ① 串码：同一码下出现多组 '1.' 开头（说明混入了多套原因） =====")
for code, causes in sorted(by.items()):
    groups = sum(1 for c in causes if re.match(r"^\s*1\.", c))
    if groups > 1:
        print("  %s | %d 组 | 共 %d 条" % (code, groups, len(causes)))

print("\n===== ② 未拆分：一个 cause 含两个及以上连续序号 =====")
num_re = re.compile(r"(?<!\d)(\d{1,2})\.")
for r in rows:
    nums = [int(x) for x in num_re.findall(r["cause"])]
    if len(nums) >= 2 and any(nums[i + 1] == nums[i] + 1 for i in range(len(nums) - 1)):
        print("  %s | %s" % (r["code"], r["cause"][:70]))

print("\n审计完成。")
