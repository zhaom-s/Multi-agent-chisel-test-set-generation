# graph/build_graph.py

import os
import config
from langgraph.graph import StateGraph, END
from state import PipelineState
from nodes import (
    verifier_node,
    verifier_v2_node,
    harness_node,
    planner_node,
    repair_node,
    rule_parser_node,
    codegen_node,
    syntax_review_node,
    compliance_review_node,
    reviewed_codegen_node,
    update_iteration_node,
)

# ============================================================
# ⚙️ 路由函数 1：Harness 后 → 判断是否需要继续
# ============================================================
def after_harness(state: PipelineState) -> str:
    it = state.get("iteration", 0)
    fail_count = state.get("fail_count", 0)
    max_it = getattr(config, "MAX_ITERATIONS", 5)

    base_root = config.BASE_ROOT
    curr_t_dir = os.path.join(base_root, f"iteration_{it}", "scala_T")
    has_files = (
        os.path.exists(curr_t_dir) and
        len([f for f in os.listdir(curr_t_dir) if f.endswith(".scala")]) > 0
    )

    # 当前轮次没有代码 → 先生成
    if not has_files:
        print(f"🚩 [Decision] Iteration {it} 尚未检测到代码，启动生成流程...")
        return "plan"

    # 所有通过
    if fail_count == 0:
        print(f"🎉 [Decision] 所有规则验证通过，任务圆满结束。")
        return "end"

    # 达到迭代上限
    if it >= max_it:
        print(f"🏁 [Decision] 已完成 {it} 轮迭代（上限 {max_it}），停止运行。")
        return "end"

    # 继续：升级轮次
    print(f"🔄 [Decision] Iteration {it} 未通过（{fail_count} 失败），准备升级...")
    return "next_it"


# ============================================================
# ⚙️ 路由函数 2：Planner 后 → repair 或 parse+gen
# ============================================================
def after_planner(state: PipelineState) -> str:
    plan = state.get("plan", {})
    skill = plan.get("skill", "base")
    repair_ids = plan.get("repair_ids", [])
    regenerate_ids = plan.get("regenerate_ids", [])

    if skill == "repair" and repair_ids and not regenerate_ids:
        print(f"🔧 [Decision] Planner 决策: 纯修复路径 ({len(repair_ids)} 条)")
        return "repair_only"

    if repair_ids:
        print(f"🔧 [Decision] Planner 决策: 修复 {len(repair_ids)} 条 + 重生成 {len(regenerate_ids)} 条")
        return "repair_then_parse"

    print(f"🚀 [Decision] Planner 决策: 纯重生成路径 ({len(regenerate_ids)} 条), skill={skill}")
    return "parse_only"


# ============================================================
# 🏗️ 构建多 Agent 流水线
# ============================================================
def build_pipeline():
    workflow = StateGraph(PipelineState)

    # 注册节点
    workflow.add_node("verifier",           verifier_node)       # 第一轮编译验证
    workflow.add_node("verifier_v2",        verifier_v2_node)    # 第二轮编译验证
    workflow.add_node("harness",            harness_node)
    workflow.add_node("planner",            planner_node)
    workflow.add_node("repair",             repair_node)
    workflow.add_node("rule_parser",        rule_parser_node)
    workflow.add_node("codegen",            codegen_node)
    workflow.add_node("syntax_review",      syntax_review_node)
    workflow.add_node("compliance_review",  compliance_review_node)
    workflow.add_node("reviewed_codegen",   reviewed_codegen_node)
    workflow.add_node("update_iter",        update_iteration_node)

    # 入口：第一轮验证（初次运行时 scala_T 不存在，会直接跳过）
    workflow.set_entry_point("verifier")

    # 正确流水线：
    # codegen → verifier(v1) → syntax_review → compliance_review
    #         → reviewed_codegen → verifier_v2 → harness → [route]
    workflow.add_edge("verifier",           "syntax_review")
    workflow.add_edge("syntax_review",      "compliance_review")
    workflow.add_edge("compliance_review",  "reviewed_codegen")
    workflow.add_edge("reviewed_codegen",   "verifier_v2")
    workflow.add_edge("verifier_v2",        "harness")

    # Harness → 路由判断
    workflow.add_conditional_edges(
        "harness",
        after_harness,
        {
            "plan":    "planner",
            "next_it": "update_iter",
            "end":     END,
        }
    )

    # UpdateIter → Planner
    workflow.add_edge("update_iter", "planner")

    # Planner → 分叉路由
    workflow.add_conditional_edges(
        "planner",
        after_planner,
        {
            "repair_only":       "repair",
            "repair_then_parse": "repair",
            "parse_only":        "rule_parser",
        }
    )

    # 生成流水线：repair → rule_parser → codegen → verifier(v1)
    workflow.add_edge("repair",      "rule_parser")
    workflow.add_edge("rule_parser", "codegen")
    workflow.add_edge("codegen",     "verifier")

    return workflow.compile()
