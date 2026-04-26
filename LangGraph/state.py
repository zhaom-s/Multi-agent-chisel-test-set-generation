# graph/state.py

import os
import re
from typing import TypedDict, List, Optional, Dict, Any
import config  # 确保读取 config.BASE_ROOT, config.MAX_ITERATIONS 等

class PipelineState(TypedDict):
    """
    全局状态：LangGraph 在各个节点之间传递的唯一数据载体
    """
    # ======================
    # 🔁 循环控制
    # ======================
    iteration: int                 # 当前第几轮
    max_iterations: int            # 最大允许轮次
    stop_reason: Optional[str]     # 停止原因

    # ======================
    # 📊 统计信息
    # ======================
    total_files: int               # 当前参与验证的文件总数
    pass_count: int                # 通过数量
    fail_count: int                # 失败数量
    pass_rate: float               # 通过率
    target_pass_rate: float        # 目标通过率
    
    prev_pass_count: int           # 上一轮通过数
    improvement: int               # 本轮提升
    min_improvement: int           # 最小提升阈值

    # ======================
    # 📂 路径信息 (由 BASE_ROOT 动态驱动)
    # ======================
    rule_md_dir: str               # 初始规则解析目录
    scala_T_dir: str               # 初始正例目录
    scala_F_dir: str               # 初始反例目录
    
    # 存档总目录
    pass_dir: str                  
    fail_dir: str                  
    
    # 📂 动态重试路径：指向当前轮次内部文件夹
    retry_rule_md_dir: str         # 当前轮次解析重试目录
    retry_scala_T_dir: str         # 当前轮次代码重试目录-T
    retry_scala_F_dir: str         # 当前轮次代码重试目录-F

    # ======================
    # ❌ 失败驱动数据
    # ======================
    failed_rules: List[int]        # 失败的规则编号 (续传核心)
    failed_files: List[str]        # 失败文件路径列表

    log_file: str                  # 日志路径

    # ======================
    # 🧠 V2.0 多Agent 感知与决策字段
    # ======================
    error_context: Dict[int, Any]  # {rule_id: {error_type, message, file_path, attempt}}
    repair_history: List[dict]     # 历史修复记录 [{rule_id, iteration, skill, result}]
    plan: Dict[str, Any]           # Planner 当前决策 {skill, use_rag, repair_ids, ...}
    skill_used: str                # 本轮实际使用的技能名称
    repair_ids: List[int]          # 本轮走修复路径的规则 ID
    attempt_count: Dict[int, int]  # {rule_id: 已尝试次数}（驱动 Planner 升级策略）

    # ======================
    # 🔬 V3.0 双专家评审字段
    # ======================
    syntax_advice: Dict[int, str]      # {rule_id: "语法专家建议"}
    compliance_advice: Dict[int, str]  # {rule_id: "合规专家建议"}
    review_done: bool                  # 本轮是否已完成评审（防止重复评审）

    # ======================
    # 📊 V4.0 两轮验证统计
    # ======================
    v1_stats: Optional[Dict]           # 第一轮编译验证统计（来自 verify_report.log）
    v2_stats: Optional[Dict]           # 第二轮编译验证统计（来自 verifiedv2_report.log）

    # ======================
    # 📋 V3.1 Harness 反馈字段
    # ======================
    last_harness_advice: str           # 上一轮 Harness 生成的迭代建议（供下一轮 Agent 参考）


# ==========================================
# ✨ 核心函数：智能初始化 (Auto-Resume Logic)
# ==========================================
def init_state() -> PipelineState:
    """
    智能初始化：
    1. 扫描磁盘寻找最大的 iteration_X 文件夹。
    2. 从该轮次的 verified_fail 存档中提取规则 ID 和 文件完整路径。
    """
    base_root = config.BASE_ROOT
    if not os.path.exists(base_root):
        os.makedirs(base_root, exist_ok=True)
    
    # --- 1. 深度探测轮次 ---
    all_dirs = os.listdir(base_root)
    iter_nums = []
    for d in all_dirs:
        if d.startswith("iteration_"):
            nums = re.findall(r'\d+', d)
            if nums: 
                iter_nums.append(int(nums[0]))
    
    # 探测到的最高轮次 (例如：2)
    current_iter = max(iter_nums) if iter_nums else 0

    # --- 2. 提取失败规则 ID 和 路径 ---
    detected_failed_rules = set()
    detected_failed_files = []
    
    # 路径对齐：当前轮次的结果目录
    search_path = os.path.join(base_root, f"iteration_{current_iter}", "verified_fail")

    if os.path.exists(search_path):
        for f in os.listdir(search_path):
            if f.endswith(".scala"):
                full_path = os.path.abspath(os.path.join(search_path, f))
                detected_failed_files.append(full_path)
                
                # 提取 Rule ID
                m = re.search(r"rule_(\d+)[_.]", f)
                if m: 
                    detected_failed_rules.add(int(m.group(1)))

    failed_list = sorted(list(detected_failed_rules))

    # --- 3. 打印探测报告 ---
    print("\n" + "="*60)
    print(f"🔎 [智能探测] 根目录: {base_root}")
    print(f"📈 [智能探测] 当前进度: Iteration {current_iter}")
    
    if not failed_list and current_iter == 0:
        print(f"🌟 [智能探测] 未发现历史进度，将从 Iteration 0 全量开始。")
        # 初始默认 65 条规则全量（nodes.py 也会有兜底判断）
        failed_list = [] 
    else:
        print(f"❌ [智能探测] 发现待修复文件: {len(detected_failed_files)} 个")
        print(f"🆔 [智能探测] 涉及规则 ID: {len(failed_list)} 条")
    print("="*60 + "\n")

    # --- 4. 组装初始状态 ---
    return {
        "iteration": current_iter,
        "max_iterations": config.MAX_ITERATIONS,
        "stop_reason": None,

        # 统计数据
        "total_files": 0,
        "pass_count": 0,
        "fail_count": len(failed_list), 
        "pass_rate": 0.0,
        "target_pass_rate": getattr(config, "TARGET_PASS_RATE", 0.98),
        
        "prev_pass_count": 0,
        "improvement": 999,
        "min_improvement": getattr(config, "MIN_IMPROVEMENT", 3),

        # 初始路径
        "rule_md_dir": os.path.join(base_root, "rule_md_analysis"),
        "scala_T_dir": os.path.join(base_root, "scala_T"),
        "scala_F_dir": os.path.join(base_root, "scala_F"),
        "pass_dir": os.path.join(base_root, "all_verified_pass"),
        "fail_dir": os.path.join(base_root, "all_verified_fail"),
        
        # 📂 动态重试路径：指向当前轮次内部文件夹
        "retry_rule_md_dir": os.path.join(base_root, f"iteration_{current_iter}", "rule_md_analysis"),
        "retry_scala_T_dir": os.path.join(base_root, f"iteration_{current_iter}", "scala_T"),
        "retry_scala_F_dir": os.path.join(base_root, f"iteration_{current_iter}", "scala_F"),

        # 失败数据记录
        "failed_rules": failed_list,
        "failed_files": detected_failed_files,

        "log_file": config.LOG_FILE,

        # V2.0 多Agent 感知与决策字段（初始化为空）
        "error_context": {},
        "repair_history": [],
        "plan": {},
        "skill_used": "base",
        "repair_ids": [],
        "attempt_count": {rid: 0 for rid in failed_list},

        # V3.0 双专家评审字段
        "syntax_advice": {},
        "compliance_advice": {},
        "review_done": False,

        # V3.1 Harness 反馈
        "last_harness_advice": "",

        # V4.0 两轮验证统计
        "v1_stats": None,
        "v2_stats": None,
    }