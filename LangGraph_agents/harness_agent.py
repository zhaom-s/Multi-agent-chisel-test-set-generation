# /root/Test-Agent/LangGraph_agents/harness_agent.py
"""
Harness Agent：系统性评估与迭代建议生成器

职责：
  1. 解析 verify_report.log，统计 T/F 通过率
  2. 汇总本轮所有 Agent 的工作产物（analysis md、enhanced md、错误模式）
  3. 用 LLM 生成一份结构化的"迭代建议报告"
  4. 将报告写入 /root/Test-Agent/harness_reports/iteration_{it}_report.md

报告内容：
  - 本轮指标：pass/all, T_pass/T_all, F_pass/F_all
  - 原始验证日志（完整）
  - 失败模式分类（按 errorMsg 聚类）
  - Agent 工作流建议（框架/提示词/策略层面）
"""

import os
import re
import json
import torch
from datetime import datetime


# ============================================================
# 日志解析
# ============================================================


def parse_verify_log(log_path: str) -> dict:
    """
    解析 verify_report.log，返回统计信息和失败详情。
    """
    result = {
        "t_pass": 0,
        "t_fail": 0,
        "f_pass": 0,
        "f_fail": 0,
        "failures": [],  # [{file, error_type, error_msg}]
        "raw_log": "",
    }

    if not os.path.exists(log_path):
        return result

    with open(log_path, "r", encoding="utf-8") as f:
        raw = f.read()
    result["raw_log"] = raw

    current_file = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("✅ PASS"):
            fname = line.split("|")[-1].strip()
            if "_T.scala" in fname:
                result["t_pass"] += 1
            else:
                result["f_pass"] += 1
        elif line.startswith("❌ FAIL"):
            fname = line.split("|")[-1].strip()
            current_file = fname
            if "_T.scala" in fname:
                result["t_fail"] += 1
            else:
                result["f_fail"] += 1
        elif line.startswith("Reason:") and current_file:
            try:
                detail = json.loads(line[len("Reason:") :].strip())
                error_msg = detail.get("errorMsg", "")
                syntax = detail.get("syntax", None)
                mlir = detail.get("mlir", "")
                if not error_msg and syntax is False:
                    error_type = "syntax_parse_fail"
                elif not error_msg and mlir == "none":
                    error_type = "empty_output"
                elif "not found" in error_msg:
                    error_type = "not_found_api"
                elif "type mismatch" in error_msg:
                    error_type = "type_mismatch"
                elif "overloaded method" in error_msg:
                    error_type = "overloaded_method"
                elif "does not take parameters" in error_msg:
                    error_type = "wrong_type_params"
                else:
                    error_type = "other"
                result["failures"].append(
                    {
                        "file": current_file,
                        "error_type": error_type,
                        "error_msg": error_msg[:300],
                    }
                )
            except Exception:
                result["failures"].append(
                    {
                        "file": current_file,
                        "error_type": "parse_error",
                        "error_msg": line[:200],
                    }
                )
            current_file = None

    return result


def _cluster_failures(failures: list) -> dict:
    """按 error_type 聚类失败项"""
    clusters = {}
    for f in failures:
        et = f["error_type"]
        clusters.setdefault(et, []).append(f["file"])
    return clusters


# ============================================================
# 读取本轮 Agent 工作产物摘要
# ============================================================


def _read_enhanced_summary(enhanced_dir: str) -> str:
    """读取 enhanced md 里的专家建议部分，做摘要"""
    if not os.path.exists(enhanced_dir):
        return "（本轮未生成 enhanced md）"
    lines = []
    for fname in sorted(os.listdir(enhanced_dir)):
        if not fname.endswith(".md"):
            continue
        rule_id = fname.replace("rule_", "").replace(".md", "")
        fpath = os.path.join(enhanced_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        # 只提取专家建议部分
        match = re.search(r"## 专家评审建议.*", content, re.DOTALL)
        if match:
            advice = match.group()[:600]
            lines.append(f"### Rule {rule_id}\n{advice}\n")
    return "\n".join(lines) if lines else "（无 enhanced 内容）"


# ============================================================
# LLM 建议生成
# ============================================================


def _build_harness_prompt(
    stats: dict, clusters: dict, enhanced_summary: str, iteration: int
) -> str:
    t_total = stats["t_pass"] + stats["t_fail"]
    f_total = stats["f_pass"] + stats["f_fail"]
    total = t_total + f_total
    pass_total = stats["t_pass"] + stats["f_pass"]

    cluster_lines = []
    for etype, files in clusters.items():
        cluster_lines.append(f"  - [{etype}] ({len(files)} 个): {', '.join(files[:5])}")
    cluster_str = "\n".join(cluster_lines) if cluster_lines else "  无失败"

    failure_details = []
    for f in stats["failures"][:10]:
        failure_details.append(
            f"  - {f['file']} [{f['error_type']}]: {f['error_msg'][:150]}"
        )
    failure_str = "\n".join(failure_details) if failure_details else "  无"

    # 先在外面算好比率
    pass_rate = (pass_total / total * 100) if total > 0 else 0.0
    t_pass_rate = (stats["t_pass"] / t_total * 100) if t_total > 0 else 0.0
    f_pass_rate = (stats["f_pass"] / f_total * 100) if f_total > 0 else 0.0

    return f"""
# Role
你是一名 AI Agent 系统架构师，专注于 Chisel 测试集生成 Agent 的系统性评估与迭代优化。

# 本轮运行指标（Iteration {iteration}）
- 总体通过率: {pass_total}/{total} ({pass_rate:.1f}%)
- 正例通过率 (T): {stats['t_pass']}/{t_total} ({t_pass_rate:.1f}%)
- 反例通过率 (F): {stats['f_pass']}/{f_total} ({f_pass_rate:.1f}%)

# 失败模式聚类
{cluster_str}

# 典型失败详情（前10条）
{failure_str}

# 本轮双专家评审建议摘要
{enhanced_summary[:2000]}

# 任务
请基于以上信息，从以下三个维度给出具体的、可操作的改进建议：

## 1. 提示词层面建议
针对失败模式，指出当前提示词的哪些约束不够明确，应如何修改。
重点关注：反例生成（既要语法正确又要违反规则）的提示词设计。

## 2. 框架流程层面建议
当前 Agent 流程是否有节点缺失或顺序问题？
是否需要增加新的检查点或反馈机制？

## 3. 下一轮优先修复项
列出 3-5 条最高优先级的具体修复动作（可直接执行的）。

请直接输出建议，结构清晰，每条建议不超过100字：
"""


# ============================================================
# 主入口
# ============================================================


def run_harness(
    iteration: int,
    iter_dir: str,
    model: torch.nn.Module,
    tokenizer: object,
    report_dir: str = "/root/Test-Agent/harness_reports",
) -> str:
    """
    生成本轮 Harness 评估报告，写入 report_dir/iteration_{it}_report.md。
    返回报告文件路径。
    """
    os.makedirs(report_dir, exist_ok=True)

    log_path = os.path.join(iter_dir, "verify_report.log")
    enhanced_dir = os.path.join(iter_dir, "rule_md_enhanced")

    # 1. 解析日志
    stats = parse_verify_log(log_path)
    clusters = _cluster_failures(stats["failures"])

    t_total = stats["t_pass"] + stats["t_fail"]
    f_total = stats["f_pass"] + stats["f_fail"]
    total = t_total + f_total
    pass_total = stats["t_pass"] + stats["f_pass"]

    t_rate = f"{stats['t_pass']}/{t_total}" if t_total else "0/0"
    f_rate = f"{stats['f_pass']}/{f_total}" if f_total else "0/0"
    total_rate = f"{pass_total}/{total}" if total else "0/0"

    print(
        f"\n📊 [Harness] Iteration {iteration} 指标: 总={total_rate}, T={t_rate}, F={f_rate}"
    )

    # 2. 读取 enhanced 摘要
    enhanced_summary = _read_enhanced_summary(enhanced_dir)

    # 3. 调用 LLM 生成建议
    print(f"🤖 [Harness] 正在生成迭代建议报告...")
    prompt = _build_harness_prompt(stats, clusters, enhanced_summary, iteration)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    llm_advice = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # 4. 组装报告
    cluster_lines = []
    for etype, files in clusters.items():
        cluster_lines.append(f"- `{etype}` ({len(files)} 个): {', '.join(files)}")

    failure_lines = []
    for f in stats["failures"]:
        failure_lines.append(
            f"- **{f['file']}** `[{f['error_type']}]`\n  ```\n  {f['error_msg']}\n  ```"
        )

    # --- 修复开始：在组装 report 字符串前先计算 ---
    final_pass_rate = (pass_total / total * 100) if total > 0 else 0.0
    final_t_rate = (stats['t_pass'] / t_total * 100) if t_total > 0 else 0.0
    final_f_rate = (stats['f_pass'] / f_total * 100) if f_total > 0 else 0.0
    
    report = f"""# Harness 评估报告 — Iteration {iteration}

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、本轮指标

| 指标 | 数值 |
|------|------|
| 总体通过率 | {total_rate} ({final_pass_rate:.1f}%) |
| 正例通过率 (T) | {t_rate} ({final_t_rate:.1f}%) |
| 反例通过率 (F) | {f_rate} ({final_f_rate:.1f}%) |

---

## 二、失败模式聚类

{chr(10).join(cluster_lines) if cluster_lines else '无失败'}

---

## 三、失败详情（含原始 errorMsg）

{chr(10).join(failure_lines) if failure_lines else '无失败'}

---

## 四、双专家评审建议摘要

{enhanced_summary[:3000]}

---

## 五、原始验证日志

```
{stats['raw_log']}
```

---

## 六、Agent 迭代建议（LLM 生成）

{llm_advice}
"""

    report_path = os.path.join(report_dir, f"iteration_{iteration}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ [Harness] 报告已写入: {report_path}")
    return report_path, llm_advice, stats
