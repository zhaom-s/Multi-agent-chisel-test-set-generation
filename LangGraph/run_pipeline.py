# /root/Test-Agent/LangGraph/run_pipeline.py

import os
import sys
import torch
import re

# ============================================================
# 🛡️ 1. 环境防御与路径修复
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 确保当前目录和父目录都在 Python 路径中
sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 物理清理编译缓存
os.system('find . -name "__pycache__" -type d -exec rm -rf {} + > /dev/null 2>&1')

# ============================================================
# 📦 2. 导入内部模块
# ============================================================
import config
from state import init_state
from build_graph import build_pipeline

def inspect_progress(base_root):
    """辅助函数：探测磁盘上的进度"""
    if not os.path.exists(base_root):
        return None
    iters = [d for d in os.listdir(base_root) if d.startswith("iteration_")]
    if not iters:
        return None
    it_nums = [int(re.findall(r'\d+', d)[0]) for d in iters if re.findall(r'\d+', d)]
    return max(it_nums) if it_nums else None

def run_main():
    print("\n" + "="*70)
    print("🚀 [LangGraph] Chisel 代码生成强化迭代流水线 (V2.0) 启动...")
    print(f"📍 根目录: {config.BASE_ROOT}")
    print("="*70)
    
    # ---------------------------------------------------------
    # 🤖 3. 状态初始化与断点检测
    # ---------------------------------------------------------
    initial_state = init_state()
    latest_it_on_disk = inspect_progress(config.BASE_ROOT)

    rule_count = sum(1 for l in open(config.RULES_FILE) if l.strip())

    # 如果是全新任务且 initial_state 没拿到规则，手动初始化全量
    if not initial_state.get("failed_rules"):
        initial_state["failed_rules"] = list(range(1, rule_count + 1))
        initial_state["fail_count"] = rule_count

    it = initial_state["iteration"]
    
    # ---------------------------------------------------------
    # 🛡️ 4. 路径强制加固
    # ---------------------------------------------------------
    curr_it_dir = os.path.join(config.BASE_ROOT, f"iteration_{it}")
    os.makedirs(os.path.join(curr_it_dir, "rule_md_analysis"), exist_ok=True)
    os.makedirs(os.path.join(curr_it_dir, "scala_T"), exist_ok=True)
    os.makedirs(os.path.join(curr_it_dir, "scala_F"), exist_ok=True)

    initial_state["retry_rule_md_dir"] = os.path.join(curr_it_dir, "rule_md_analysis")
    initial_state["retry_scala_T_dir"] = os.path.join(curr_it_dir, "scala_T")
    initial_state["retry_scala_F_dir"] = os.path.join(curr_it_dir, "scala_F")

    # 🏗️ 编译流水线图
    app = build_pipeline()
    
    # ---------------------------------------------------------
    # 🏁 5. 执行工作流
    # ---------------------------------------------------------
    try:
        force_skill = getattr(config, "FORCE_SKILL", "") or "auto"
        print(f"\n📊 [任务配置预览]:")
        print(f"   - 当前轮次: Iteration {it}")
        print(f"   - Skill 模式: {force_skill}")
        print(f"   - 迭代上限: {config.MAX_ITERATIONS}")
        print(f"   - 规则数量: {rule_count}")
        print(f"   - 显存状态: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print("-" * 70)

        # 核心：启动 LangGraph
        final_state = app.invoke(initial_state)
        
        # ---------------------------------------------------------
        # 📈 6. 结果汇报
        # ---------------------------------------------------------
        print("\n" + "="*70)
        print("✅ [任务完成] 全流程迭代结束！")
        
        it_final = final_state.get('iteration', it)
        fail_cnt = final_state.get('fail_count', 0)
        
        print(f"📈 最终总结数据 (结束于 Iteration {it_final}):")
        print(f"   - 剩余失败项: {fail_cnt}")
        
        if fail_cnt == 0:
            print(f"🎯 最终结果: 【完美通关】所有规则均已通过编译验证！")
        else:
            print(f"🎯 最终结果: 剩余 {fail_cnt} 条规则未能修复，建议查看最后一次迭代的分析报告。")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ [核心运行时异常]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_main()