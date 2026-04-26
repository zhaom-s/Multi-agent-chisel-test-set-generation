import os
import torch
import re

# =============================
# 构建强化版 Prompt (保留你的思维链协议)
# =============================
def build_reinforced_prompt(rule_text: str) -> str:
    # 保持你原始的 Prompt 逻辑不变，这是核心灵魂
    return f"""
# Role
你是一名资深的 Chisel/Scala 编译器前端工程师与硬件验证专家。你的核心任务是协助下游的“代码生成智能体”构建高质量的测试用例数据集。

# Instruction (The Chain of Thought Protocol)
在处理自然语言规则时，你**必须**严格执行以下**两个输出阶段**（缺一不可）：

### 阶段一：思维链推演 (<thought_process>)
首先，请在 `<thought_process>` 标签内进行深度推理，回答以下问题：
1.  **语义分析**: 规则解决了什么硬件设计痛点？
2.  **代码映射**: 对应的 Scala/Chisel 代码写法（正/误）是什么？
3.  **AST 提取**: 如果我编写一个静态分析工具，我应该捕捉什么样的语法树节点（如 `Defn.Var`, `Term.Apply`, `Type.Name`）？
4.  **正反例构造**: 
    * 最典型的**错误写法**是什么？（新手最容易犯的错）
    * 最标准的**修正写法**是什么？（官方文档推荐的写法）

### 阶段二：结构化输出 (Final Blueprint)
**在闭合 `</thought_process>` 标签后，你必须立即生成最终的解析报告**。报告需包含以下章节：
1. 规则核心定义
2. 违规模式生成指南
3. 合规模式生成指南
4. 推荐变量与上下文

# Few-Shot Examples

## 示例 1
**输入规则**:
在Chisel硬件构造中使用var变量会破坏函数式编程原则，导致状态难以追踪，应基于不可变值（val）和组合/时序逻辑原语（如Reg、Wire）。

**模型输出**:
<thought_process>
1.  **语义分析**: `var` 是 Scala 的可变变量，虽然 Chisel 允许写，但这通常意味着设计者试图用软件思维写硬件（例如用 var 做累加），这会导致生成的电路不可预测或难以分析。
2.  **代码映射**: 错误写法是 `var count = 0` 然后 `count += 1`。正确写法是 `val reg = RegInit(0.U)` 然后 `reg := reg + 1.U`。
3.  **AST 提取**: Scalameta 中变量定义是 `Defn.Var`，而常量定义是 `Defn.Val`。这是最显著的区别。
4.  **正反例构造**: 反例是使用 `var` 声明信号。正例是使用 `val` 配合 `Reg/Wire`。
</thought_process>

## 1. 规则核心定义
禁止使用 Scala 可变变量 `var` 定义硬件信号，强制使用 `val` 配合 `Reg/Wire` 类型以保持硬件描述的不可变性与可分析性。

## 2. 违规模式生成指南
* **代码特征**: 使用 `var` 关键字声明变量，并对其进行多次重赋值（Reassignment）。
* **触发场景**: 尝试在 `Module` 中使用软件风格的循环或条件累加。
* **AST 关键指纹**: `Defn.Var` 节点。

## 3. 合规模式生成指南
* **修复策略**: 将状态存储迁移至 `RegInit`，将逻辑连接迁移至 `Wire`。
* **代码特征**: 使用 `val reg = RegInit(...)` 存储状态，使用 `when(...) {{ reg := ... }}` 更新状态。
* **预期行为**: 明确的时序逻辑（寄存器）或组合逻辑（线网），生成确定的电路结构。

## 4. 推荐变量与上下文
* **推荐变量名**: `accumulator`, `state_reg`, `counter_val`
* **数据类型建议**: `UInt(32.W)`, `Bool`

# Start Task
**待分析规则**: 
{rule_text}

**请开始执行阶段一，并在完成后立即执行阶段二：**
"""

# =============================
# 工具函数
# =============================
def save_output_md(output_text, output_path):
    # 增加一个鲁棒性处理：确保思维链标签闭合，移除多余的 markdown 标记
    cleaned = (
        output_text.replace("```json", "")
        .replace("```markdown", "")
        .replace("```", "")
        .strip()
    )
    # 如果模型生成了多个重复的阶段，只截取到第一次出现的完整结构（可选）
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

# ==========================================================
# 【核心修改】接收外部传入的共享模型，不再在内部初始化
# ==========================================================
def run_retry_rule_analysis(
    pending_rules: list,    # 由 Graph 传入：全量时传入 range(1,66)
    output_dir: str,        # iteration_1/rule_md_analysis
    rules_txt_path: str,    
    model: torch.nn.Module, 
    tokenizer: object,      
    rag=None,               # 预留 RAG 接口
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(rules_txt_path):
        print(f"❌ [Retry-Agent] 找不到规则文件: {rules_txt_path}")
        return

    with open(rules_txt_path, "r", encoding="utf-8") as f:
        # 过滤掉可能的空行，并建立索引
        all_rules = [line.strip() for line in f if line.strip()]

    print(f"🚨 [Retry-Agent] 启动强化解析引擎，目标规则数: {len(pending_rules)}")

    for idx in pending_rules:
        # 1. 检查是否已经生成过（断点续传逻辑，如果不需要可删掉）
        output_path = os.path.join(output_dir, f"rule_{idx}.md")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            print(f"⏩ [Retry-Agent] 跳过已存在的强化解析: rule_{idx}")
            continue

        # 2. 索引校验
        if idx < 1 or idx > len(all_rules):
            print(f"⚠️ [Retry-Agent] 跳过无效索引: {idx}")
            continue

        rule_text = all_rules[idx - 1]
        print(f"🧠 [Retry-Agent] 正在执行【思维链协议】深度解析：rule_{idx} ...")

        # 3. 构造 Prompt
        prompt = build_reinforced_prompt(rule_text)
        
        # 4. 推理
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1500,  # 思维链解析通常比普通解析长，调大到 1500
                temperature=0.3,     
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1 # 增加轻微惩罚，防止思维链陷入死循环
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # 5. 保存
        save_output_md(output_text, output_path)
        print(f"✅ [Retry-Agent] 强化解析完成: rule_{idx}")



