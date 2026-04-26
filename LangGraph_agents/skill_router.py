# /root/Test-Agent/LangGraph_agents/skill_router.py
"""
技能注册表：统一映射 skill 名称 → (parser_fn, codegen_fn)
替代 nodes.py 中的 if-elif-else 硬编码路由。
新增技能只需在 SKILL_MAP 中注册，无需修改节点逻辑。
"""

from LangGraph_agents.rule_analysis import run_rule_analysis
from LangGraph_agents.code_generate import run_code_generate
from LangGraph_agents.retry_rule_parser import run_retry_rule_analysis
from LangGraph_agents.retry_code_generate import run_retry_code_generate
from LangGraph_agents.deep_analysis import run_deep_rule_analysis
from LangGraph_agents.deep_generate import run_deep_code_generate

# ============================================================
# 技能注册表
# key   : skill 名称（由 Planner Agent 输出）
# value : (parse_fn, codegen_fn)
# ============================================================
SKILL_MAP = {
    "base":  (run_rule_analysis,       run_code_generate),
    "cot":   (run_retry_rule_analysis, run_retry_code_generate),
    "deep":  (run_deep_rule_analysis,  run_deep_code_generate),
}

SKILL_DESCRIPTIONS = {
    "base":  "基础解析+生成（适用于首次或规则语义清晰的场景）",
    "cot":   "思维链强化解析+多模块生成（适用于 syntax 反复失败）",
    "deep":  "架构师视角深度解析+工程化生成（适用于 MLIR/topModule 失败）",
}


def get_skill(skill: str):
    """
    返回 (parser_fn, codegen_fn) 元组。
    若 skill 不在注册表中，回退到 base。
    """
    if skill not in SKILL_MAP:
        print(f"⚠️ [SkillRouter] 未知技能 '{skill}'，回退到 base")
        skill = "base"
    return SKILL_MAP[skill]


def list_skills() -> list:
    """返回所有已注册的技能名称列表（供 Planner prompt 使用）"""
    return list(SKILL_MAP.keys())


def describe_skills() -> str:
    """返回技能描述字符串（直接嵌入 Planner prompt）"""
    lines = []
    for name, desc in SKILL_DESCRIPTIONS.items():
        lines.append(f"  - `{name}`: {desc}")
    return "\n".join(lines)
