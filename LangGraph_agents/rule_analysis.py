import os
import torch

# ============================
# 构建 prompt (核心逻辑保持不动)
# ============================
def build_prompt(rule_text: str) -> str:
    return f"""
你是一个专注于对语言文本格式的chisel规则进行解析的专家。你的任务是把自然文本形式的规则，把它扩充为一份解析。接下来我会给你对于一份代码规则生成的示例。请你仿照生成这样的解析内容。
你的唯一任务是：
【针对“本次规则”生成且仅生成一份解析结果】。

⚠️  【强制约束 - 违者必究】
1. **单次输出协议**：你【有且仅有一次】机会生成解析结果。
2. **禁止循环**：严禁在输出结束后重新开始或重复生成。
3. **静默执行**：禁止任何前导开场白（如“好的，这是解析...”）或后验总结。
4. **结构对齐**：必须严格遵守 1, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 4.3 的编号格式。

================================================================
===========================【示例参考输出内容格式 】============================
================================================================

【规则】
在Chisel模块实现中调用print语句属于严重反模式，此类语句在FIRRTL综合阶段会被忽略，仅在仿真时生效，导致综合与仿真行为不一致，且可能泄露敏感信息或干扰日志系统，应改用专用调试机制如printf或Diplomacy调试接口。

【解析】
1. 规则核心
在 Chisel 模块代码中禁止使用 Scala 的 print，应使用 Chisel 提供的调试原语，以避免综合与仿真行为不一致。

2. 正例关键
2.1
- 使用 printf 进行调试输出
- 所有日志类行为通过 Chisel 内建调试机制完成
- 调试语句仅用于仿真，不影响硬件结构
2.2
这些特征能确保硬件综合行为与仿真一致，并遵守 Chisel 的硬件构造语义。

3. 反例关键
3.1
- 直接调用 print 输出调试信息
- 在硬件逻辑路径上混入 Scala 侧副作用
- 用 print 模拟硬件状态变化或监控
3.2
这些问题会导致综合阶段语句被忽略，硬件与仿真结果不一致。

4. 生成指导
4.1 **推荐模式**: 使用 printf(io.signal) 或使用 Chisel Debug utilities
4.2 **反例模式**: 直接在模块中写 print("xx")
4.3 **推荐变量**:
- statusFlag
- debugValue
- cycleCounter



================================================================
===========================【针对规则的解析文件的输出 】============================
================================================================
【任务明确】
接下来将向你提供一条“规则”。请你阅读该条规则，并“严格模仿上方示例的结构与编号格式”生成该规则的解析内容。

**待分析规则**：
{rule_text}


**执行要求**：
请直接开始输出解析内容。**在你输出完“4.3 推荐变量”列表的最后一个字符后，必须立即停止，严禁再次输出“【解析】”或任何字符。**

请注意：
- 你只能输出解析内容，不能输出示例内容。
- 禁止输出上述示例 任何复述或引用。
- 你的输出必须严格符合以下编号结构,然后效仿上方示例补充该条规则对应的每条内容：


1. 规则核心
<一句话总结规则核心思想，不可直接照抄原文>

2. 正例关键
2.1 <列出2~3条正例遵守规则的关键特征>
2.2 <描述这些特征体现了规则要求>

3. 反例关键
3.1 <列出2~3条反例常见问题>
3.2 <描述这些问题违反了规则精神>

4. 生成指导
4.1 **推荐模式**: <给出能体现该规则的命名或结构模板，以供大模型用于代码生成>
4.2 **反例模式**: <给出违反该规则的典型错误模板，以供大模型用于代码生成>
4.3 **推荐变量**:
- <变量名示例 1>
- <变量名示例 2>
- <变量名示例 3>



"""

# ============================
# 保存 Markdown 输出
# ============================
def save_output_md(output_text, output_path):
    cleaned = (
        output_text.replace("```json", "")
        .replace("```markdown", "")
        .replace("```", "")
        .strip()
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

# ==========================================================
# 【核心修改】不再在内部加载模型，而是接收外部传入的 model 和 tokenizer
# ==========================================================
def run_rule_analysis(
    pending_rules: list,    # 由 Graph 传入
    output_dir: str,        # 由 Graph 传入
    rules_txt_path: str,    # config.py 里的路径
    model: torch.nn.Module, # 【新增】接收已加载的模型
    tokenizer: object,       # 【新增】接收已加载的 tokenizer
    rag=None,
    **kwargs,
):
    """
    该函数由 graph/nodes.py 调用，直接使用传入的模型进行推理。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取所有原始规则
    if not os.path.exists(rules_txt_path):
        print(f"❌ [Agent] 找不到规则文件: {rules_txt_path}")
        return

    with open(rules_txt_path, "r", encoding="utf-8") as f:
        all_rules = [line.strip() for line in f if line.strip()]

    print(f"📘 [Node] 本轮需解析规则数量: {len(pending_rules)}")

    for idx in pending_rules:
        # 确保索引合法
        if idx < 1 or idx > len(all_rules):
            print(f"⚠️ 跳过无效索引: {idx}")
            continue
            
        rule_text = all_rules[idx - 1]
        print(f"▶️ [Agent] 正在解析规则 {idx}...")

        prompt = build_prompt(rule_text)
        
        # 使用模型自带的 device 确保张量位置正确
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        output_path = os.path.join(output_dir, f"rule_{idx}.md")
        save_output_md(output_text, output_path)
        print(f"✅ [Agent] 规则 {idx} 解析完成并保存到 {output_path}")

