# graph/nodes.py

import os
import re
import torch
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import config
from state import PipelineState

# ============================================================
# 🔗 导入 Agent
# ============================================================
from LangGraph_agents.skill_router import get_skill
from LangGraph_agents.planner_agent import run_planner
from LangGraph_agents.repair_agent import run_repair
from LangGraph_agents.verifier import run_verifier
from LangGraph_agents.syntax_reviewer import run_syntax_review
from LangGraph_agents.rule_compliance_reviewer import run_compliance_review
from LangGraph_agents.harness_agent import run_harness
from RAG_Service_v2 import ChiselHybridRAGService

# ============================================================
# 🔧 资源加载 (LLM + RAG)
# ============================================================
MODEL = None
TOKENIZER = None
RAG_SERVICE = None


def load_resources_once(use_rag: bool = False):
    global MODEL, TOKENIZER, RAG_SERVICE

    if MODEL is None:
        print(f"🚀 [Model] 正在加载模型: {os.path.basename(config.BASE_MODEL)}...")
        TOKENIZER = AutoTokenizer.from_pretrained(config.BASE_MODEL, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        if hasattr(config, "USE_LORA") and config.USE_LORA:
            print(f"✨ [LoRA] 挂载适配器模式: {os.path.basename(config.LORA_PATH)}")
            MODEL = PeftModel.from_pretrained(base, config.LORA_PATH, is_trainable=False)
        else:
            MODEL = base
        MODEL.eval()
        print("✅ [Model] 模型加载成功")

    if use_rag and RAG_SERVICE is None:
        print(f"📚 [RAG] 激活 RAG 检索服务...")
        RAG_SERVICE = ChiselHybridRAGService(db_path="/root/chisel-RAG/vector_db", device="cuda:0")

    return MODEL, TOKENIZER, RAG_SERVICE if use_rag else None


# ============================================================
# 🧪 Node 1: Verifier — 验证 + 收集错误上下文
# ============================================================
def verifier_node(state: PipelineState) -> PipelineState:
    """
    第一轮编译验证（codegen 之后）。
    结果写入 verified_pass/ + verified_fail/ + verify_report.log。
    同时把 v1 统计存入 state["v1_stats"] 供 progress 报告使用。
    """
    it = state["iteration"]
    base_root = config.BASE_ROOT
    current_iter_dir = os.path.join(base_root, f"iteration_{it}")

    t_dir = os.path.join(current_iter_dir, "scala_T")
    if not os.path.exists(t_dir) or not any(f.endswith(".scala") for f in os.listdir(t_dir)):
        # 还没有代码，跳过验证
        state["failed_rules"] = list(range(1, _count_rules() + 1))
        state["fail_count"] = _count_rules()
        state["error_context"] = {}
        state["v1_stats"] = None
        return state

    print(f"\n🔍 --- [Node: Verifier-V1] 第一轮编译验证 (Iteration {it}) ---")

    from LangGraph_agents.harness_agent import parse_verify_log
    failed_ids, error_context = run_verifier(
        scala_t_dir=os.path.join(current_iter_dir, "scala_T"),
        scala_f_dir=os.path.join(current_iter_dir, "scala_F"),
        api_url=config.API_URL,
        iter_dir=current_iter_dir,
        attempt_count=state.get("attempt_count", {}),
    )

    attempt_count = dict(state.get("attempt_count", {}))
    for rid in error_context:
        attempt_count[rid] = attempt_count.get(rid, 0) + 1

    state["failed_rules"] = failed_ids
    state["fail_count"] = len(failed_ids)
    state["error_context"] = error_context
    state["attempt_count"] = attempt_count

    # 解析 v1 统计供 progress 报告使用
    log_path = os.path.join(current_iter_dir, "verify_report.log")
    state["v1_stats"] = parse_verify_log(log_path)
    return state


# ============================================================
# 📝 Node 2: Harness — 评估报告生成（每轮验证后自动触发）
# ============================================================
def harness_node(state: PipelineState) -> PipelineState:
    """
    在两轮验证都完成后生成 Harness 报告，并写 progress 文件。
    v1_stats 来自第一轮验证，v2_stats 来自第二轮验证。
    """
    it = state["iteration"]
    base_root = config.BASE_ROOT
    current_iter_dir = os.path.join(base_root, f"iteration_{it}")

    v2_log = os.path.join(current_iter_dir, "verifiedv2_report.log")
    v1_log = os.path.join(current_iter_dir, "verify_report.log")

    # 两轮日志都不存在 → 本轮还没有任何代码，跳过
    if not os.path.exists(v1_log) and not os.path.exists(v2_log):
        print(f"ℹ️ [Harness] Iteration {it} 无验证日志，跳过报告生成")
        return state

    model, tokenizer, _ = load_resources_once(use_rag=False)
    report_dir = os.path.join(base_root, "harness_reports")

    # 用 v2 日志（更完整）生成 Harness 报告；若 v2 不存在则用 v1
    primary_log_dir = current_iter_dir  # run_harness 内部会找 verify_report.log
    print(f"\n📝 --- [Node: Harness] 生成评估报告 (Iteration {it}) ---")
    report_path, llm_advice, v2_stats = run_harness(
        iteration=it,
        iter_dir=current_iter_dir,
        model=model,
        tokenizer=tokenizer,
        report_dir=report_dir,
    )
    if llm_advice:
        state["last_harness_advice"] = llm_advice

    # 写 progress 报告（两阶段真实数据）
    v1_stats = state.get("v1_stats")
    v2_stats_final = state.get("v2_stats") or v2_stats
    _save_progress_both(it, v1_stats, v2_stats_final, base_root)
    return state


# ============================================================
# 🧭 Node 3: Planner — 感知失败根因，输出技能选择方案
# ============================================================
def planner_node(state: PipelineState) -> PipelineState:
    it = state["iteration"]
    error_context = state.get("error_context", {})
    all_ids = list(range(1, _count_rules() + 1))

    # FORCE_SKILL 模式：跳过 LLM，直接使用配置指定的 skill
    force_skill = getattr(config, "FORCE_SKILL", "")
    if force_skill:
        print(f"🎯 [Planner] FORCE_SKILL={force_skill}，跳过 LLM 决策，全量重生成")
        plan = {
            "skill": force_skill,
            "use_rag": False,
            "repair_ids": [],
            "regenerate_ids": all_ids,
            "reasoning": f"FORCE_SKILL={force_skill}（调试模式）",
        }
        state["plan"] = plan
        state["skill_used"] = force_skill
        state["repair_ids"] = []
        return state

    # 无失败时也不需要 LLM
    if not error_context and state.get("fail_count", 0) == 0:
        state["plan"] = {"skill": "base", "use_rag": False,
                         "repair_ids": [], "regenerate_ids": [], "reasoning": "无失败"}
        return state

    # 无错误上下文但有失败（初始生成阶段）→ 默认 deep skill
    if not error_context:
        print(f"🚀 [Planner] 无错误上下文（初始生成），使用 deep skill 全量生成")
        plan = {
            "skill": "deep",
            "use_rag": False,
            "repair_ids": [],
            "regenerate_ids": all_ids,
            "reasoning": "初始生成阶段，无错误上下文",
        }
        state["plan"] = plan
        state["skill_used"] = "deep"
        state["repair_ids"] = []
        return state

    model, tokenizer, _ = load_resources_once(use_rag=False)
    plan = run_planner(
        error_context=error_context,
        iteration=it,
        model=model,
        tokenizer=tokenizer,
    )
    state["plan"] = plan
    state["skill_used"] = plan.get("skill", "base")
    state["repair_ids"] = plan.get("repair_ids", [])
    return state


# ============================================================
# 🔧 Node 3: RepairAgent — 定向修复失败代码
# ============================================================
def repair_node(state: PipelineState) -> PipelineState:
    it = state["iteration"]
    repair_ids = state.get("repair_ids", [])
    if not repair_ids:
        print("ℹ️ [RepairNode] 无需修复，跳过")
        return state

    base_root = config.BASE_ROOT
    current_iter_dir = os.path.join(base_root, f"iteration_{it}")

    model, tokenizer, _ = load_resources_once(use_rag=False)

    print(f"\n🔧 --- [Node: RepairAgent] 定向修复 (Iteration {it}) ---")
    repair_results = run_repair(
        repair_ids=repair_ids,
        iter_dir=current_iter_dir,
        rule_md_dir=state["retry_rule_md_dir"],
        error_context=state.get("error_context", {}),
        model=model,
        tokenizer=tokenizer,
    )

    # 记录修复历史（论文用：展示 Agent 修复轨迹）
    history = list(state.get("repair_history", []))
    for rid, result in repair_results.items():
        history.append({
            "rule_id": rid,
            "iteration": it,
            "skill": "repair",
            "result": result,
        })
    state["repair_history"] = history
    return state


# ============================================================
# 🧠 Node 4: RuleParser — 根据 Planner 技能动态调度解析 Agent
# ============================================================
def rule_parser_node(state: PipelineState) -> PipelineState:
    it = state["iteration"]
    plan = state.get("plan", {})
    skill = plan.get("skill", "base")
    use_rag = plan.get("use_rag", False)

    # repair 路径不走 parser（直接修复原代码）
    if skill == "repair":
        print(f"ℹ️ [ParserNode] skill=repair，本轮跳过全量解析，仅修复失败项")
        return state

    model, tokenizer, rag = load_resources_once(use_rag=use_rag)
    parse_fn, _ = get_skill(skill)

    pending = sorted(list(range(1, _count_rules() + 1)))
    print(f"\n📖 --- [Node: RuleParser] skill={skill}, rag={use_rag is not None} (Iteration {it}) ---")

    parse_fn(
        pending_rules=pending,
        output_dir=state["retry_rule_md_dir"],
        rules_txt_path=config.RULES_FILE,
        model=model,
        tokenizer=tokenizer,
        rag=rag,
        harness_advice=state.get("last_harness_advice", ""),
    )
    return state


# ============================================================
# ⚙️ Node 5: CodeGen — 根据 Planner 技能动态调度代码生成 Agent
# ============================================================
def codegen_node(state: PipelineState) -> PipelineState:
    it = state["iteration"]
    plan = state.get("plan", {})
    skill = plan.get("skill", "base")
    use_rag = plan.get("use_rag", False)

    if skill == "repair":
        print(f"ℹ️ [CodeGenNode] skill=repair，本轮跳过全量代码生成")
        return state

    model, tokenizer, rag = load_resources_once(use_rag=use_rag)
    _, codegen_fn = get_skill(skill)

    print(f"\n🛠️  --- [Node: CodeGen] skill={skill}, rag={use_rag is not None} (Iteration {it}) ---")

    codegen_fn(
        rule_md_dir=state["retry_rule_md_dir"],
        scala_t_dir=state["retry_scala_T_dir"],
        scala_f_dir=state["retry_scala_F_dir"],
        model=model,
        tokenizer=tokenizer,
        rag=rag,
        harness_advice=state.get("last_harness_advice", ""),
    )
    return state


# ============================================================
# 🔬 Node 6: SyntaxReviewer — 语法专家预检
# ============================================================
def syntax_review_node(state: PipelineState) -> PipelineState:
    """只对第一轮验证失败的文件做语法预检（verified_fail/ 目录）。"""
    it = state["iteration"]
    base_root = config.BASE_ROOT
    current_iter_dir = os.path.join(base_root, f"iteration_{it}")
    fail_dir = os.path.join(current_iter_dir, "verified_fail")

    # 从 verified_fail 目录推断失败的 rule_id 列表（T/F 分开）
    failed_t_ids, failed_f_ids = _failed_ids_from_dir(fail_dir)
    all_failed_ids = sorted(failed_t_ids | failed_f_ids)

    if not all_failed_ids:
        print(f"ℹ️ [SyntaxReviewer] 无失败文件，跳过语法预检")
        state["syntax_advice"] = {}
        state["review_done"] = False
        return state

    model, tokenizer, _ = load_resources_once(use_rag=False)
    print(f"\n🔬 --- [Node: SyntaxReviewer] 语法预检 {len(all_failed_ids)} 条失败规则 (Iteration {it}) ---")

    # 专家只看 verified_fail 里的文件（真实失败的）
    syntax_advice = run_syntax_review(
        rule_ids=all_failed_ids,
        scala_t_dir=fail_dir,
        scala_f_dir=fail_dir,
        model=model,
        tokenizer=tokenizer,
    )
    state["syntax_advice"] = syntax_advice
    state["review_done"] = False
    return state


# ============================================================
# 📋 Node 7: ComplianceReviewer — 规则合规专家审查
# ============================================================
def compliance_review_node(state: PipelineState) -> PipelineState:
    """只对第一轮验证失败的文件做规则合规审查（verified_fail/ 目录）。"""
    it = state["iteration"]
    base_root = config.BASE_ROOT
    current_iter_dir = os.path.join(base_root, f"iteration_{it}")
    fail_dir = os.path.join(current_iter_dir, "verified_fail")

    failed_t_ids, failed_f_ids = _failed_ids_from_dir(fail_dir)
    all_failed_ids = sorted(failed_t_ids | failed_f_ids)

    if not all_failed_ids:
        print(f"ℹ️ [ComplianceReviewer] 无失败文件，跳过合规审查")
        state["compliance_advice"] = {}
        state["review_done"] = True
        return state

    model, tokenizer, _ = load_resources_once(use_rag=False)
    print(f"\n📋 --- [Node: ComplianceReviewer] 合规审查 {len(all_failed_ids)} 条失败规则 (Iteration {it}) ---")

    compliance_advice = run_compliance_review(
        rule_ids=all_failed_ids,
        scala_t_dir=fail_dir,
        scala_f_dir=fail_dir,
        rule_md_dir=state["retry_rule_md_dir"],
        model=model,
        tokenizer=tokenizer,
    )
    state["compliance_advice"] = compliance_advice
    state["review_done"] = True
    return state


# ============================================================
# 🔄 Node 8: ReviewedCodeGen — 结合双专家建议重新生成
# ============================================================
def _adv_has_t_issue(adv: str) -> bool:
    """判断建议字符串是否涉及正例（T）问题"""
    if not adv:
        return False
    ok_markers = ("无明显", "均符合", "跳过", "审查异常")
    if any(m in adv for m in ok_markers):
        return False
    if "[正例]" in adv:
        return True
    # 无标签前缀时视为两侧都有问题
    if "[反例]" not in adv:
        return True
    return False


def _adv_has_f_issue(adv: str) -> bool:
    """判断建议字符串是否涉及反例（F）问题"""
    if not adv:
        return False
    ok_markers = ("无明显", "均符合", "跳过", "审查异常")
    if any(m in adv for m in ok_markers):
        return False
    if "[反例]" in adv:
        return True
    # 无标签前缀时视为两侧都有问题
    if "[正例]" not in adv:
        return True
    return False


def reviewed_codegen_node(state: PipelineState) -> PipelineState:
    """
    基于双专家建议，对第一轮失败的文件做二次生成。
    - 只重生成有失败的那一侧（T/F 独立）
    - 输出到 verifiedv2_T/ 和 verifiedv2_F/（不覆盖已通过的原始文件）
    - 不需要重生成的一侧直接从 verified_pass/ 复制过来
    """
    it = state["iteration"]
    plan = state.get("plan", {})
    skill = plan.get("skill", "base")
    use_rag = plan.get("use_rag", False)

    base_root = config.BASE_ROOT
    current_iter_dir = os.path.join(base_root, f"iteration_{it}")
    fail_dir = os.path.join(current_iter_dir, "verified_fail")
    pass_dir = os.path.join(current_iter_dir, "verified_pass")

    # 从 verified_fail 中推断需要重生成的 T/F 文件
    failed_t_ids, failed_f_ids = _failed_ids_from_dir(fail_dir)

    # 输出目录：二次生成的新文件放在独立目录
    v2_t_dir = os.path.join(current_iter_dir, "verifiedv2_T")
    v2_f_dir = os.path.join(current_iter_dir, "verifiedv2_F")
    os.makedirs(v2_t_dir, exist_ok=True)
    os.makedirs(v2_f_dir, exist_ok=True)

    if not failed_t_ids and not failed_f_ids:
        print(f"ℹ️ [ReviewedCodeGen] 无失败文件，跳过二次生成")
        return state

    syntax_advice = state.get("syntax_advice", {})
    compliance_advice = state.get("compliance_advice", {})

    # 构建 enhanced md（对每个有失败文件的规则）
    rule_md_dir = state["retry_rule_md_dir"]
    enhanced_md_dir = os.path.join(current_iter_dir, "rule_md_enhanced")
    os.makedirs(enhanced_md_dir, exist_ok=True)

    all_failed_ids = sorted(failed_t_ids | failed_f_ids)
    for rid in all_failed_ids:
        src_md = os.path.join(rule_md_dir, f"rule_{rid}.md")
        dst_md = os.path.join(enhanced_md_dir, f"rule_{rid}.md")
        try:
            with open(src_md, "r", encoding="utf-8") as f:
                original = f.read()
        except Exception:
            continue

        syn_adv = syntax_advice.get(rid, "")
        com_adv = compliance_advice.get(rid, "")
        enhanced = original + f"""

---
## 专家评审建议（请在重新生成时严格遵守）

### 语法专家建议
{syn_adv or "无"}

### 规则合规专家建议
{com_adv or "无"}
"""
        with open(dst_md, "w", encoding="utf-8") as f:
            f.write(enhanced)

    model, tokenizer, rag = load_resources_once(use_rag=use_rag)
    _, codegen_fn = get_skill(skill if skill != "repair" else "base")

    print(f"\n🔄 --- [Node: ReviewedCodeGen] T侧失败 {len(failed_t_ids)} 条 / F侧失败 {len(failed_f_ids)} 条 → 二次生成 (Iteration {it}) ---")

    # 把 T 侧失败文件先放入 v2_t_dir（标记为待重生成），T 侧通过的也复制过去
    # codegen 只会生成缺少的文件，所以把通过的先复制好，失败的不复制 → 会被重新生成
    for fname in os.listdir(pass_dir):
        if fname.endswith("_T.scala"):
            src = os.path.join(pass_dir, fname)
            dst = os.path.join(v2_t_dir, fname)
            if not os.path.exists(dst):
                import shutil; shutil.copy(src, dst)
        elif fname.endswith("_F.scala"):
            src = os.path.join(pass_dir, fname)
            dst = os.path.join(v2_f_dir, fname)
            if not os.path.exists(dst):
                import shutil; shutil.copy(src, dst)

    # 生成（增量：只生成 v2 目录里缺少的文件）
    codegen_fn(
        rule_md_dir=enhanced_md_dir,
        scala_t_dir=v2_t_dir,
        scala_f_dir=v2_f_dir,
        model=model,
        tokenizer=tokenizer,
        rag=rag,
        harness_advice=state.get("last_harness_advice", ""),
    )
    return state


# ============================================================
# ✅ Node 8b: Verifier-V2 — 二次生成后的编译验证
# ============================================================
def verifier_v2_node(state: PipelineState) -> PipelineState:
    """
    对 reviewed_codegen 输出的文件（verifiedv2_T/ + verifiedv2_F/）做第二轮编译验证。
    结果写入 verifiedv2_pass/ + verifiedv2_fail/ + verifiedv2_report.log。
    把 v2 统计存入 state["v2_stats"] 供 harness/progress 使用。
    """
    it = state["iteration"]
    base_root = config.BASE_ROOT
    current_iter_dir = os.path.join(base_root, f"iteration_{it}")

    v2_t_dir = os.path.join(current_iter_dir, "verifiedv2_T")
    v2_f_dir = os.path.join(current_iter_dir, "verifiedv2_F")

    has_t = os.path.exists(v2_t_dir) and any(f.endswith(".scala") for f in os.listdir(v2_t_dir))
    has_f = os.path.exists(v2_f_dir) and any(f.endswith(".scala") for f in os.listdir(v2_f_dir))

    if not has_t and not has_f:
        print(f"ℹ️ [Verifier-V2] 无二次生成文件，跳过")
        state["v2_stats"] = None
        return state

    print(f"\n🔍 --- [Node: Verifier-V2] 二次编译验证 (Iteration {it}) ---")

    from LangGraph_agents.harness_agent import parse_verify_log

    # 将 v2_T/ + v2_F/ 作为输入目录，验证结果放在独立子目录
    v2_iter_dir_fake = current_iter_dir  # 传入 iter_dir 只是为了写 log
    failed_ids_v2, error_context_v2 = run_verifier(
        scala_t_dir=v2_t_dir,
        scala_f_dir=v2_f_dir,
        api_url=config.API_URL,
        iter_dir=current_iter_dir,
        attempt_count=state.get("attempt_count", {}),
        pass_subdir="verifiedv2_pass",
        fail_subdir="verifiedv2_fail",
        log_name="verifiedv2_report.log",
    )

    # 将 v2 的错误上下文合并（覆盖 v1 的）供 Planner 下一轮使用
    merged_ctx = dict(state.get("error_context", {}))
    merged_ctx.update(error_context_v2)
    state["error_context"] = merged_ctx

    # 用 v2 报告重算 fail_count
    v2_log = os.path.join(current_iter_dir, "verifiedv2_report.log")
    state["v2_stats"] = parse_verify_log(v2_log)

    # fail_count = 仍然失败的文件数
    # v2 验证的是 verifiedv2_T/ + verifiedv2_F/，其中包含了 v1 已通过的文件（复制过来的）
    # 所以直接用 v2 的 fail 数即可（v2 fail = 仍未修复的文件）
    v2_fail = (state["v2_stats"]["t_fail"] + state["v2_stats"]["f_fail"]) if state["v2_stats"] else 0
    state["fail_count"] = v2_fail
    state["failed_rules"] = sorted(set(failed_ids_v2))
    return state


# ============================================================
# 🔁 Node 9: UpdateIteration
# ============================================================
def update_iteration_node(state: PipelineState) -> PipelineState:
    state["iteration"] += 1
    it = state["iteration"]

    new_dir = os.path.join(config.BASE_ROOT, f"iteration_{it}")
    state["retry_rule_md_dir"] = os.path.join(new_dir, "rule_md_analysis")
    state["retry_scala_T_dir"] = os.path.join(new_dir, "scala_T")
    state["retry_scala_F_dir"] = os.path.join(new_dir, "scala_F")

    os.makedirs(state["retry_rule_md_dir"], exist_ok=True)
    os.makedirs(state["retry_scala_T_dir"], exist_ok=True)
    os.makedirs(state["retry_scala_F_dir"], exist_ok=True)

    print(f"\n📂 [Update] 升级至 Iteration {it}。")
    return state



# ============================================================
# 🛠️ 辅助工具：从 verified_fail/ 目录推断失败的 rule_id 集合
# ============================================================
def _failed_ids_from_dir(fail_dir: str):
    """
    扫描 fail_dir，返回 (failed_t_ids: set, failed_f_ids: set)。
    例如 rule_3_T.scala → failed_t_ids.add(3)
         rule_3_F.scala → failed_f_ids.add(3)
    """
    import re as _re
    failed_t = set()
    failed_f = set()
    if not os.path.exists(fail_dir):
        return failed_t, failed_f
    for fname in os.listdir(fail_dir):
        if not fname.endswith(".scala"):
            continue
        m = _re.search(r"rule_(\d+)_(T|F)\.scala", fname)
        if not m:
            continue
        rid = int(m.group(1))
        side = m.group(2)
        if side == "T":
            failed_t.add(rid)
        else:
            failed_f.add(rid)
    return failed_t, failed_f


# ============================================================
# 📊 进度报告辅助函数
# ============================================================
def _save_progress_both(it: int, v1_stats, v2_stats, base_root: str):
    """
    写进度报告：
    - 第一阶段 = 第一轮编译器真实验证结果（v1_stats）
    - 第二阶段 = 二次生成后编译器真实验证结果（v2_stats）
    - 综合总结：合并两轮后的最终通过率
    """
    from datetime import datetime
    progress_dir = os.path.join(base_root, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    report_path = os.path.join(progress_dir, f"iteration_{it}_progress.md")

    def _fmt(stats):
        if not stats:
            return {"t_p": 0, "t_t": 0, "f_p": 0, "f_t": 0,
                    "t_rate": "N/A", "f_rate": "N/A", "all": "0/0", "all_rate": "N/A"}
        tt = stats["t_pass"] + stats["t_fail"]
        ft = stats["f_pass"] + stats["f_fail"]
        tot = tt + ft
        p = stats["t_pass"] + stats["f_pass"]
        return {
            "t_p": stats["t_pass"], "t_t": tt,
            "f_p": stats["f_pass"], "f_t": ft,
            "t_rate": f"{stats['t_pass']/tt*100:.1f}" if tt else "0.0",
            "f_rate": f"{stats['f_pass']/ft*100:.1f}" if ft else "0.0",
            "all": f"{p}/{tot}",
            "all_rate": f"{p/tot*100:.1f}" if tot else "0.0",
        }

    v1 = _fmt(v1_stats)
    v2 = _fmt(v2_stats)

    # 综合总结：v1 通过 的文件 + v2 通过 的文件（互补，不重复）
    # v1 通过的 T 文件数 + v2 通过的 T 文件数（v2 只验证 v1 失败的，所以不重叠）
    total_rules = _count_rules()
    final_t_pass = v1["t_p"] + v2["t_p"]
    final_f_pass = v1["f_p"] + v2["f_p"]
    final_all_pass = final_t_pass + final_f_pass
    final_total = total_rules * 2
    final_rate = f"{final_all_pass/final_total*100:.1f}" if final_total else "0.0"

    # 收集失败的规则 ID 列表
    v1_fail_t_ids_list = "（数据未记录）"
    v1_fail_f_ids_list = "（数据未记录）"
    v1_fail_dir = os.path.join(base_root, f"iteration_{it}", "verified_fail")
    if os.path.exists(v1_fail_dir):
        ft_ids, ff_ids = _failed_ids_from_dir(v1_fail_dir)
        v1_fail_t_ids_list = ", ".join(f"Rule {r}" for r in sorted(ft_ids)) or "无"
        v1_fail_f_ids_list = ", ".join(f"Rule {r}" for r in sorted(ff_ids)) or "无"

    v2_fail_t_ids_list = "（数据未记录）"
    v2_fail_f_ids_list = "（数据未记录）"
    v2_fail_dir = os.path.join(base_root, f"iteration_{it}", "verifiedv2_fail")
    if os.path.exists(v2_fail_dir):
        ft_ids2, ff_ids2 = _failed_ids_from_dir(v2_fail_dir)
        v2_fail_t_ids_list = ", ".join(f"Rule {r}" for r in sorted(ft_ids2)) or "无"
        v2_fail_f_ids_list = ", ".join(f"Rule {r}" for r in sorted(ff_ids2)) or "无"

    report = f"""# 迭代进度报告 — Iteration {it}

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 第一阶段：初次代码生成 → 编译器验证结果

| 指标 | 数值 |
|------|------|
| 正例通过率 (T) | {v1["t_p"]}/{v1["t_t"]} ({v1["t_rate"]}%) |
| 反例通过率 (F) | {v1["f_p"]}/{v1["f_t"]} ({v1["f_rate"]}%) |
| 总体通过率 | {v1["all"]} ({v1["all_rate"]}%) |

### 第一阶段失败的规则
- T侧失败: {v1_fail_t_ids_list}
- F侧失败: {v1_fail_f_ids_list}

---

## 第二阶段：双专家增强重生成 → 编译器验证结果

| 指标 | 数值 |
|------|------|
| 正例通过率 (T) | {v2["t_p"]}/{v2["t_t"]} ({v2["t_rate"]}%) |
| 反例通过率 (F) | {v2["f_p"]}/{v2["f_t"]} ({v2["f_rate"]}%) |
| 总体通过率 | {v2["all"]} ({v2["all_rate"]}%) |

### 第二阶段失败的规则（仍未修复）
- T侧仍失败: {v2_fail_t_ids_list}
- F侧仍失败: {v2_fail_f_ids_list}

---

## 综合总结（两轮合并最终通过率）

| 指标 | 数值 |
|------|------|
| 最终正例通过数 (T) | {final_t_pass}/{total_rules} ({final_t_pass/total_rules*100:.1f}%) |
| 最终反例通过数 (F) | {final_f_pass}/{total_rules} ({final_f_pass/total_rules*100:.1f}%) |
| **最终总体通过率** | **{final_all_pass}/{final_total} ({final_rate}%)** |

> 本轮结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📊 [Progress] 完整进度报告已写入: {report_path}")


# ============================================================
# 🛠️ 工具：统计规则总数
# ============================================================
def _count_rules() -> int:
    try:
        with open(config.RULES_FILE, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return len(lines)
    except Exception:
        return 65
