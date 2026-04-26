# /root/Test-Agent/LangGraph_agents/deep_generate.py

import os
import re
import torch

# ==========================================================
# [Prompt 构建] - 这里预留了与“架构师指南”对接的逻辑
# ==========================================================

def build_deep_generate_prompt(rule_md_text: str) -> str:
    return f"""
# Role
你是一名资深的 Chisel 硬件架构师与验证专家。你的任务是根据提供的《高覆盖率测试用例开发指南》，生成两段高度工程化、可独立编译的 Chisel 代码。

# Metadata Instruction (元指令)
⚠️ **【重要约束】**：
1. 第一段以 //正例 开头；第二段以 //反例 开头。
2. 仅生成两段独立的代码。无任何额外解释文字（禁止出现 # Task Output, # Explanation 等）。
3. 严禁照抄**：下方示例仅展示结构。你必须根据本次输入的《指南》中的“工程场景”和“注入点”重新构思逻辑。
4. 严禁导入任何package。
5. 每个文件必须是完整的（包含 import）。

# Output Specification
1. **结构分明**：第一段以 `//正例` 开头；第二段以 `//反例` 开头。
2. **独立编译**：包含 `import chisel3._` 和 `import chisel3.util._`。
3. **精准注释**：必须标注 `// [预期通过]` 或 `// 违反规则：...`。

# [理想输出风格示例] (仅供参考，请勿照抄)
================================================================
========================【理想输出风格示例】===========================
================================================================

以下示例展示了理想格式、结构、缩进和注释方式。你的输出应严格模仿示例的风格，但严禁照抄内容。
------------------------- 示例1开始 -------------------------
//正例
import chisel3._
import chisel3.util._

// 场景：高性能流水线中的数据校验器
class DataValidator(val width: Int) extends Module {{
  val io = IO(new Bundle {{
    val dataIn  = Input(UInt(width.W))
    val isValid = Input(Bool())
    val dataOut = Output(UInt(width.W))
  }})

  // [预期通过] 遵循指南 A：使用显式寄存器流，确保逻辑路径完备
  val resultReg = RegInit(0.U(width.W))
  when(io.isValid) {{
    resultReg := io.dataIn
  }} .otherwise {{
    resultReg := 0.U
  }}
  io.dataOut := resultReg
}}

//反例
import chisel3._
import chisel3.util._

// 场景：带有参数化陷阱的接口协议转换器
class ProtocolBridge(val bypassMode: Boolean) extends Module {{
  val io = IO(new Bundle {{
    val rawIn    = Input(UInt(8.W))
    val protoOut = Output(UInt(8.W))
  }})

  // [预期报错] 遵循指南 B：利用 Scala 逻辑提前中断硬件构造
  // 注入点：bypassMode 为 true 时，Scala 的 return 导致 io.protoOut 的硬件连接不可达
  def connectLogic(): Unit = {{
    if (bypassMode) {{
      return // 违反规则：在 Scala 函数中使用 return 提前退出，导致后续硬件端口未连接
    }}
    io.protoOut := io.rawIn + 1.U
  }}

  connectLogic()
}}
// === End of Deep Test Case ===


------------------------- 示例2开始 -------------------------
//正例
import chisel3._
import chisel3.util._
class CompliantModule extends Module {{
  val io = IO(new Bundle {{ val out = Output(UInt(8.W)) }})
  io.out := 1.U
}}

//反例
import chisel3._
import chisel3.util._
class ViolationModule extends Module {{
  val io = IO(new Bundle {{ val out = Output(UInt(8.W)) }})
  // 违反规则：此处演示了错误的逻辑连接
  io.out := 0.U 
}}
------------------------- 示例结束 -------------------------

# Task Input
以下是本次任务需要适配的《高覆盖率测试用例开发指南》：
----------------------------------------
{rule_md_text}
----------------------------------------

**请根据上述指南，开始生成两段完整代码：**
//正例
//反例（含“// 违反规则：...”）
"""

# ==========================================================
# [工具函数]
# ==========================================================

def clean_markdown_blocks(text: str) -> str:
    """清理模型可能输出的 Markdown 代码块标签"""
    return (
        text.replace("```markdown", "")
        .replace("```scala", "")
        .replace("```chisel", "")
        .replace("```", "")
        .strip()
    )

def extract_examples(output_text: str):
    """提取正例和反例代码"""
    pos_match = re.search(
        r"(//\s*正例|【\s*正例\s*】|正例[:：])\s*(.*?)"
        r"(?=(//\s*反例|【\s*反例\s*】|反例[:：]|$))",
        output_text, re.DOTALL,
    )
    pos_code = pos_match.group(2).strip() if pos_match else ""

    neg_match = re.search(
        r"(//\s*反例|【\s*反例\s*】|反例[:：])\s*(.*)",
        output_text, re.DOTALL,
    )
    neg_code = neg_match.group(2).strip() if neg_match else ""
    return pos_code, neg_code

# ==========================================================
# [核心执行引擎]
# ==========================================================

PLACEHOLDER = "⚠️ Deep-CodeGen 生成失败，内容缺失\n"

def run_deep_code_generate(
    rule_md_dir: str,
    scala_t_dir: str,
    scala_f_dir: str,
    model: torch.nn.Module,
    tokenizer: object,
    rag=None,
    harness_advice: str = "",
):
    """
    深度代码生成引擎。
    支持增量生成，并针对复杂架构代码调优了生成参数。
    """
    os.makedirs(scala_t_dir, exist_ok=True)
    os.makedirs(scala_f_dir, exist_ok=True)

    if not os.path.exists(rule_md_dir):
        print(f"❌ [Deep-CodeGen] 找不到解析目录: {rule_md_dir}")
        return

    md_files = [f for f in os.listdir(rule_md_dir) if f.endswith(".md")]
    print(f"🚀 [Deep-CodeGen] 准备处理 {len(md_files)} 个深度解析指南...")

    for md_filename in sorted(md_files):
        base_name = os.path.splitext(md_filename)[0]
        md_path = os.path.join(rule_md_dir, md_filename)
        
        pos_path = os.path.join(scala_t_dir, f"{base_name}_T.scala")
        neg_path = os.path.join(scala_f_dir, f"{base_name}_F.scala")
        
        # 增量判定
        if os.path.exists(pos_path) and os.path.exists(neg_path):
            print(f"⏭️  [Deep-CodeGen] 跳过已存在用例: {base_name}")
            continue

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                rule_md_text = f.read()

            md_with_advice = rule_md_text
            if harness_advice:
                md_with_advice += f"""

---
## 📊 上一轮 Harness 评估建议（生成时请严格规避以下错误模式）
{harness_advice[:1000]}
---
"""
            full_prompt = build_deep_generate_prompt(md_with_advice)
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            print(f"🛠️  [Deep-CodeGen] 正在根据系统架构指南构造: {base_name} ...")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3000,    # 深度用例结构复杂，大幅提升 Token 上限
                    temperature=0.5,        # 适度提高温度以增加代码结构的多样性
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.05
                )

            input_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            
            cleaned_output = clean_markdown_blocks(decoded)
            pos_code, neg_code = extract_examples(cleaned_output)

            # 写入正例
            with open(pos_path, "w", encoding="utf-8") as f:
                f.write((pos_code if pos_code else PLACEHOLDER) + "\n")
            
            # 写入反例
            with open(neg_path, "w", encoding="utf-8") as f:
                f.write((neg_code if neg_code else PLACEHOLDER) + "\n")

            print(f"✅ [Deep-CodeGen] {base_name} (正例/反例) 生成成功")

        except Exception as e:
            print(f"❌ [Deep-CodeGen] {base_name} 发生异常: {str(e)}")
            if "CUDA" in str(e): raise e

    print(f"🏁 [Deep-CodeGen] 全量深度生成任务结束。")