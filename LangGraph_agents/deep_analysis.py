# /root/Test-Agent/LangGraph_agents/deep_analysis.py

import os
import torch
import re

# ============================================================
# 【核心部分】构建 Deep Analysis 专用 Prompt
# 这里预留了三个深度解析维度：硬件语义、边界冲突、以及隐蔽路径
# ============================================================
def build_deep_prompt(rule_text: str) -> str:
    return f"""
# Role
你是一名精通 Chisel 大型项目开发的【系统架构师与高级 QA 专家】。你的任务是编写一份《高覆盖率测试用例开发指南》，指导后续的编程 Agent 开发出既符合工程实践、又能触达边界条件的测试代码。

# Instruction (The Engineering Guidance Protocol)
你生成的解析必须直接服务于“代码实现”。你需要告诉开发者：在真实的 SoC 或复杂 IP 开发中，这个规则通常在什么场景下被**模范遵循**，又在什么路径下被**隐蔽违反**。

请严格按以下结构输出（禁止废话）：

---

### 🗂️ [工程场景画像：Rule in Context]
<描述该规则在大型项目中的存在感。为什么要在这个场景下测试这条规则？>

### 🏗️ [指南 A：如何构造“模范遵循”的工程用例]
**1. 推荐的架构拓扑**:
- <描述推荐的模块组织形式，使规则的遵循变得自然且健壮。>
**2. 核心开发逻辑**:
- <给出实现该规则时的“最佳实践”思考路径。>
**3. 成功要素标签**: [标签1] [标签2]

### 🚧 [指南 B：如何构造“具有挑战性”的违反用例]
**1. 隐蔽违规的注入点**:
- <指导如何生成具有误导性的雷点。>
**2. 违规逻辑的伪装术**:
- <如何通过复杂连线或深层嵌套掩盖违反行为。>
**3. 典型失败模式**: [模式1] [模式2]

### 📝 [开发者便签：编程 Agent 专用指令]
- **建议的 Mock 对象**: <建议加入的模拟信号或状态机。>
- **命名诱导**: <3-5 个容易出现在真实项目中的变量名。>
- **代码复杂度建议**: <例如：嵌套层数或继承关系要求。>

---

# Few-Shot Example (以“不可达代码”规则为例)

**输入规则**: 
代码不可达（Unreachable Code）：函数中 return/throw 后、死循环（while(true)）后、或恒为假（when(false.B)）的条件分支内部代码无法执行。会导致逻辑错误与资源浪费。

**模型输出示例**:

### 🗂️ [工程场景画像：Rule in Context]
在 SoC 的状态机控制（FSM）或总线仲裁协议中，由于复杂的嵌套逻辑和参数化配置（Parameters），容易出现因逻辑冗余导致的“暗码”。这类不可达代码不仅增加维护成本，还可能导致仿真器与综合工具的行为差异（Mismatch）。

### 🏗️ [指南 A：如何构造“模范遵循”的工程用例]
**1. 推荐的架构拓扑**:
- **配置化分支**: 使用 Scala 的 if/else 处理静态参数配置，使生成的 Chisel 代码在静态期即确定路径。
**2. 核心开发逻辑**:
- 所有的状态跳转必须基于动态输入信号。在编写长函数时，通过逻辑分片代替 return 中断，确保硬件描述的连贯性。
**3. 成功要素标签**: [显式数据流] [参数化有效性检查] [FSM 完备性]

### 🚧 [指南 B：如何构造“具有挑战性”的违反用例]
**1. 隐蔽违规的注入点**:
- **参数化陷阱**: 构造一个依赖外部 Parameter 的模块，当参数在特定边界（如 Width=0）时，导致 `when(params.active.B)` 变为恒假。
- **防御性编程过度**: 在一个已全覆盖的状态机中，添加永远无法触达的 default 分支并写入关键业务逻辑。
**2. 违规逻辑的伪复术**:
- **影子更新 (Shadow Update)**: 在一个 while(true.B) 循环后放置复位清理逻辑，使该逻辑永远失效。
- **逻辑矛盾掩护**: 使用复杂的布尔算子组合 `when(a && !a)`，其中 a 是经过多层传递后的别名信号。
**3. 典型失败模式**: [静态分支被阉割] [影子逻辑失效] [复位路径中断]

### 📝 [开发者便签：编程 Agent 专用指令]
- **建议的 Mock 对象**: 构造一个名为 DeadlockMonitor 的状态机，在特定状态注入不可达跳转。
- **命名诱导**: emergencyShutdown, finalValidator, shadow_ctrl_path
- **代码复杂度建议**: 建议将违规点埋藏在 3 层 when...elsewhen 的嵌套结构中，且最外层由一个 val 常量控制。

---

# Start Task
**待分析规则**: 
{rule_text}

**请直接输出针对该规则的《高覆盖率测试用例开发指南》：**
"""

# ============================================================
# 工具函数：保存与清洗
# ============================================================
def save_output_md(output_text, output_path):
    # 移除可能存在的 Markdown 标签
    cleaned = (
        output_text.replace("```json", "")
        .replace("```markdown", "")
        .replace("```", "")
        .strip()
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

# ============================================================
# 【执行引擎】由 Node 调用的主函数
# ============================================================
def run_deep_rule_analysis(
    pending_rules: list,
    output_dir: str,
    rules_txt_path: str,
    model: torch.nn.Module,
    tokenizer: object,
    rag=None,
    harness_advice: str = "",
):
    """
    深度解析执行引擎。
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(rules_txt_path):
        print(f"❌ [Deep-Analysis] 找不到规则文件: {rules_txt_path}")
        return

    with open(rules_txt_path, "r", encoding="utf-8") as f:
        all_rules = [line.strip() for line in f if line.strip()]

    print(f"🧬 [Deep-Analysis] 启动深度解析路线，目标数量: {len(pending_rules)}")

    for idx in pending_rules:
        output_path = os.path.join(output_dir, f"rule_{idx}.md")
        
        # 断点续传逻辑
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            print(f"⏩ [Deep-Analysis] 跳过已存在的解析: rule_{idx}")
            continue

        if idx < 1 or idx > len(all_rules):
            print(f"⚠️ [Deep-Analysis] 跳过无效索引: {idx}")
            continue

        rule_text = all_rules[idx - 1]
        print(f"🧪 [Deep-Analysis] 正在挖掘规则 {idx} 的深层边界...")

        # 1. 构造 Prompt
        prompt = build_deep_prompt(rule_text)
        if harness_advice:
            prompt += f"""

---
# 📊 上一轮 Harness 评估建议（请在解析时重点关注以下失败模式）
{harness_advice[:1200]}
---
请在你的解析中，针对上述建议中提到的失败模式，提供更具体的规避指导。
"""
        
        # 2. 推理
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1200,   # 深度解析需要一定的字数空间
                temperature=0.4,       # 稍微调高一点温度，增加发散性思考
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # 3. 保存结果
        save_output_md(output_text, output_path)
        print(f"✅ [Deep-Analysis] 规则 {idx} 深度拆解完成。")

print("💡 Deep Analysis Agent 框架加载完毕。")