# /root/Test-Agent/LangGraph_agents/planner_agent.py
"""
Planner Agent：自主感知失败根因，动态输出技能选择方案。

输入 : error_context  —— {rule_id: {error_type, message, file_path, attempt}}
       iteration      —— 当前迭代轮次
       model/tokenizer —— 共享 LLM 资源

输出 : plan = {
    "skill"        : "base" | "cot" | "deep" | "repair",
    "use_rag"      : True | False,
    "repair_ids"   : [rule_id, ...],    # 有明确错误信息 → 定向修复
    "regenerate_ids": [rule_id, ...],   # 无明确错误信息 → 重新生成
    "reasoning"    : "...",             # Planner 的推理摘要（用于日志/论文）
}
"""

import os
import re
import json
import torch
from LangGraph_agents.skill_router import describe_skills, list_skills


# ============================================================
# 启发式规则（快速兜底，减少 LLM 调用）
# ============================================================

def _heuristic_plan(error_context: dict, iteration: int) -> dict | None:
    """
    对于模式非常清晰的失败，直接返回计划，不需要调用 LLM。
    返回 None 表示需要 LLM 决策。
    """
    if not error_context:
        return {
            "skill": "base", "use_rag": False,
            "repair_ids": [], "regenerate_ids": [],
            "reasoning": "无失败项，无需操作"
        }

    total = len(error_context)
    repair_ids = []
    regenerate_ids = []
    mlir_fail = 0
    syntax_fail = 0
    empty_fail = 0

    for rid, info in error_context.items():
        etype = info.get("error_type", "unknown")
        attempt = info.get("attempt", 1)
        has_msg = bool(info.get("message") and str(info.get("message")) not in ("None", "{}"))

        if etype == "empty" or not has_msg:
            empty_fail += 1
            regenerate_ids.append(rid)
        elif etype == "mlir":
            mlir_fail += 1
            repair_ids.append(rid)
        elif etype == "syntax":
            syntax_fail += 1
            if attempt >= 2:
                repair_ids.append(rid)
            else:
                regenerate_ids.append(rid)

    # 简单启发式：全是第一次 syntax 失败 → base 重生成
    if syntax_fail == total and iteration <= 1:
        return {
            "skill": "base", "use_rag": False,
            "repair_ids": [], "regenerate_ids": list(error_context.keys()),
            "reasoning": f"首轮 syntax 失败 {total} 条，使用 base 技能全量重生成"
        }

    # 有大量 MLIR 失败 → deep + RAG
    if mlir_fail > total * 0.5:
        return {
            "skill": "deep", "use_rag": True,
            "repair_ids": repair_ids, "regenerate_ids": regenerate_ids,
            "reasoning": f"MLIR 失败占多数({mlir_fail}/{total})，切换 deep 技能并激活 RAG"
        }

    return None  # 需要 LLM 决策


# ============================================================
# LLM 驱动的 Planner
# ============================================================

def _build_planner_prompt(error_context: dict, iteration: int) -> str:
    skills_desc = describe_skills()
    available_skills = list_skills() + ["repair"]

    error_summary_lines = []
    for rid, info in list(error_context.items())[:20]:  # 最多展示 20 条，避免超出 context
        etype = info.get("error_type", "unknown")
        msg = str(info.get("message", ""))[:200]  # 截断过长消息
        attempt = info.get("attempt", 1)
        error_summary_lines.append(f"  - Rule {rid}: [{etype}] attempt={attempt} | {msg}")
    error_summary = "\n".join(error_summary_lines)
    total_fail = len(error_context)

    return f"""
# Role
你是一个自主决策的 Planner Agent，负责分析 Chisel 代码生成失败的根因，并为下游 Agent 选择最优技能方案。

# 当前状态
- 当前迭代轮次: {iteration}
- 失败规则总数: {total_fail}
- 可用技能列表:
{skills_desc}
  - `repair`: 定向修复（适用于有明确编译错误信息的规则，不重新解析，直接修改代码）

# 失败详情（最多展示20条）
{error_summary}

# 错误类型说明
- `syntax`: Scala/Chisel 语法错误，通常可通过更详细的提示词修复
- `mlir`: FIRRTL/MLIR 转换失败，通常需要深度架构分析和 RAG 辅助
- `empty`: 生成内容为空或占位符，需要重新生成
- `unknown`: 未知原因，按 syntax 处理

# 任务
请分析上述失败信息，输出一个 JSON 格式的决策方案，包含：
- `skill`: 选择的技能（{"/".join(available_skills)}）
- `use_rag`: 是否激活 RAG 检索（true/false）
- `repair_ids`: 需要定向修复的规则 ID 列表（有具体错误信息的）
- `regenerate_ids`: 需要重新解析+生成的规则 ID 列表
- `reasoning`: 一句话解释你的决策逻辑

# 输出格式（严格 JSON，不含其他文字）
```json
{{
  "skill": "cot",
  "use_rag": false,
  "repair_ids": [3, 7, 12],
  "regenerate_ids": [1, 5, 9],
  "reasoning": "多数为 syntax 失败且已尝试过一次，切换 CoT 技能；有明确错误信息的走修复路径"
}}
```

请直接输出 JSON，不要有任何前置说明。
"""


def _parse_plan_from_output(text: str, fallback_ids: list) -> dict:
    """从 LLM 输出中提取 JSON plan，失败则返回安全默认值"""
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            plan = json.loads(match.group())
            plan.setdefault("skill", "cot")
            plan.setdefault("use_rag", False)
            plan.setdefault("repair_ids", [])
            plan.setdefault("regenerate_ids", fallback_ids)
            plan.setdefault("reasoning", "LLM 决策（已解析）")
            return plan
    except Exception as e:
        print(f"⚠️ [Planner] JSON 解析失败: {e}")
    return {
        "skill": "cot", "use_rag": False,
        "repair_ids": [], "regenerate_ids": fallback_ids,
        "reasoning": "LLM 输出解析失败，回退到 cot 技能全量重生成"
    }


def run_planner(
    error_context: dict,
    iteration: int,
    model: torch.nn.Module,
    tokenizer: object,
) -> dict:
    """
    主入口：返回 plan 字典。
    优先使用启发式规则，复杂情况调用 LLM。
    """
    all_ids = list(error_context.keys())
    print(f"\n🧭 [Planner] 开始分析失败根因 (iteration={iteration}, 失败数={len(all_ids)})...")

    # 1. 先尝试启发式决策（快速、零 LLM 开销）
    heuristic = _heuristic_plan(error_context, iteration)
    if heuristic is not None:
        print(f"✅ [Planner] 启发式决策: skill={heuristic['skill']}, "
              f"repair={len(heuristic['repair_ids'])}, "
              f"regenerate={len(heuristic['regenerate_ids'])}")
        print(f"   推理: {heuristic['reasoning']}")
        return heuristic

    # 2. 复杂情况：调用 LLM 进行推理
    print(f"🤖 [Planner] 启发式无法覆盖，调用 LLM 进行决策...")
    prompt = _build_planner_prompt(error_context, iteration)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    plan = _parse_plan_from_output(output_text, all_ids)
    print(f"✅ [Planner] LLM 决策: skill={plan['skill']}, "
          f"use_rag={plan['use_rag']}, "
          f"repair={len(plan['repair_ids'])}, "
          f"regenerate={len(plan['regenerate_ids'])}")
    print(f"   推理: {plan['reasoning']}")
    return plan
