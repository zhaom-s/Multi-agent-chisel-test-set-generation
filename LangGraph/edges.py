# graph/edges.py
from state import PipelineState
import config

def should_continue(state: PipelineState):
    """
    决策函数：控制 LangGraph 的流向
    适配三阶段进化：
    Iteration 0-1: Base
    Iteration 2-3: Retry
    Iteration 4-5: Deep
    """
    it = state.get("iteration", 0)
    fail_count = state.get("fail_count", 0)
    pass_rate = state.get("pass_rate", 0.0)
    
    # 从 state 或 config 获取目标
    max_it = state.get("max_iterations", getattr(config, "MAX_ITERATIONS", 5))
    target_pass_rate = state.get("target_pass_rate", getattr(config, "TARGET_PASS_RATE", 0.98))

    # 1. 【通关判定】如果失败数为 0，且已经不是初始化状态
    if fail_count == 0 and it > 0:
        print(f"🎉 [Decision] 所有规则均已通过验证（Pass Rate: 100%），任务圆满完成！")
        return "end"

    # 2. 【上限判定】如果达到最大迭代轮次
    if it >= max_it:
        print(f"🛑 [Decision] 已达到实验预设的最大迭代轮次 ({max_it})，停止运行。")
        state["stop_reason"] = "Max iterations reached"
        return "end"

    # 3. 【达标判定】如果通过率已经达到预期目标 (例如 98%)
    if pass_rate >= target_pass_rate and it > 0:
        print(f"🎯 [Decision] 当前通过率 {pass_rate:.2%} 已达标 ({target_pass_rate:.2%})，提前结束。")
        return "end"

    # 4. 【演进判定】继续进行
    # 在全量生成模式下，我们通常不因为“单轮无进步”而停止，
    # 因为 It 2 和 It 4 会更换更强大的 Agent 策略，必须给它们尝试的机会。
    
    # 确定下一轮将采用的阶段（仅用于日志打印）
    next_stage = "Base"
    if 2 <= (it + 1) <= 3:
        next_stage = "Retry (CoT + Multi-Module)"
    elif (it + 1) >= 4:
        next_stage = "Deep (Architectural Guide)"

    print(f"🔄 [Decision] Iteration {it} 结束，失败项: {fail_count}。")
    print(f"🚀 [Decision] 准备进入 Iteration {it+1}，切换至策略: 【{next_stage}】")
    
    return "continue"