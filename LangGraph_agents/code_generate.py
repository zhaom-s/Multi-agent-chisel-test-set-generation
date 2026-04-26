import os
import re
import torch

# =============================
# 构造输入 prompt
# =============================
USER_PROMPT = """
你是一名 Chisel 代码生成顾问。
你的任务：根据给定的“规则可编码行为解析文件”，输出两段可独立编译的 Chisel 代码（正例 / 反例）。

================================================================
============================【生成规范】============================
================================================================

-输出两段代码，无任何额外解释文字
-第一段以 //正例 开头；第二段以 //反例 开头
-每一段均为【完整、可独立编译】的 Scala 文件（仅包含模块和相关类）
-正例：严格符合解析文件描述的推荐行为与约束
-反例：Scala 语法与 Chisel 语法均合法，但明确违反解析文件中的规则
-必须在违规位置添加注释：// 违反规则：...
-两段代码之间不得共享任何定义（类、对象、函数、参数等）
-两段代码必须均可在【Chisel 3.x】环境下成功编译
-不生成任何入口函数或 ChiselStage 调用

================================================================
======================【严格禁止与黑名单】========================
================================================================
【禁止发明 API】
-严禁创建解析文件中未出现的 API、函数、方法、类或字段
-只能使用以下来源：
-Chisel 标准库（chisel3、chisel3.util）
-解析文件中明确提到的 API
-若解析文件未提及某 API，仅允许使用最基础、确定性的构造：
-Reg / Wire / IO / Module / UInt / Bool / Vec / Bundle / Mux / when


【黑名单：完全禁止出现以下元素】
-不允许出现：Option / implicit / lazy val / RawModule / MultiIOModule
-不允许生成自定义 DSL、虚构类型、虚构方法
-不允许创建不存在的硬件语义或协议（如虚构握手信号）
-不允许自动生成测试代码（testbench）
-不允许使用 chiseltest、PeekPokeTester 或任何测试框架


================================================================
========================【理想输出示例】===========================
================================================================

以下示例展示了理想格式、结构、缩进和注释方式。你的输出应严格模仿示例的风格。

------------------------- 示例 1 -------------------------
//正例
import chisel3._

case class Parameters(total: Int, sum: Int) {
  require(total > 0)
  val result = total + sum  // 使用普通 val，确定性初始化
}

class ModuleA extends Module {
  val params = Parameters(10, 20)
  val io = IO(new Bundle {
    val out = Output(UInt(8.W))
  })
  io.out := params.result.U
}


//反例
import chisel3._

case class Parameters(total: Int, sum: Int) {
  lazy val result = total + sum  // 违反规则：lazy val 导致不确定初始化
}

class ModuleB extends Module {
  val params = Parameters(10, 20)
  val io = IO(new Bundle {
    val out = Output(UInt(8.W))
  })
  io.out := params.result.U
}



------------------------- 示例 2 -------------------------
//正例
import chisel3._

class RandomGenerator(seed: Long) {
  private val rnd = new scala.util.Random(seed)
  def nextInt(max: Int): Int = rnd.nextInt(max)
}

class TestModule extends Module {
  val io = IO(new Bundle {
    val out = Output(UInt(8.W))
  })
  val randomGenerator = new RandomGenerator(42L)
  io.out := randomGenerator.nextInt(256).U
}



//反例
import chisel3._

class TestModule extends Module {
  val io = IO(new Bundle {
    val out = Output(UInt(8.W))
  })
  val rnd = new scala.util.Random(42L)
  io.out := rnd.nextInt(256).U  // 违反规则：在硬件模块中直接使用 Scala 随机数
}




================================================================
==========================【任务输入】============================
================================================================

你将收到规则对应的“可编码行为解析文件”，并需要根据解析内容生成代码。


"""

def build_prompt(rule_md_text: str) -> str:
    return f"""{USER_PROMPT}

以下是规则的可编码行为解析文件：
----------------------------------------
{rule_md_text}
----------------------------------------

仅根据解析文件生成两段完整代码：
- //正例
- //反例（含“// 违反规则：...”）
"""

# =============================
# 工具函数 (保持原样)
# =============================
def clean_markdown_blocks(text: str) -> str:
    return (
        text.replace("```markdown", "").replace("```json", "").replace("```scala", "")
        .replace("```chisel", "").replace("```", "").strip()
    )

def extract_examples(output_text: str):
    # 这里的正则保持你原来的逻辑
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
# 【核心修改】不再加载模型，直接接收外部传入的 model 和 tokenizer
# ==========================================================
PLACEHOLDER = "⚠️ 生成失败，规则 MD 文件格式异常或内容缺失\n"

def run_code_generate(
    rule_md_dir: str,       # 由 Graph 传入
    scala_t_dir: str,       # 由 Graph 传入
    scala_f_dir: str,       # 由 Graph 传入
    model: torch.nn.Module, # 接收已加载的模型
    tokenizer: object,      # 接收已加载的 tokenizer
    rag=None,
    mode="base",
    **kwargs,
):
    """
    该函数由 graph/nodes.py 调用。
    具备增量生成能力：仅生成并写入目标文件夹中缺失的 .scala 文件。
    """
    # 1. 确保输出目录存在
    os.makedirs(scala_t_dir, exist_ok=True)
    os.makedirs(scala_f_dir, exist_ok=True)

    # 2. 获取所有待处理的 MD 文件
    if not os.path.exists(rule_md_dir):
        print(f"❌ [Agent] 找不到解析目录: {rule_md_dir}")
        return

    md_files = [f for f in os.listdir(rule_md_dir) if f.endswith(".md")]
    print(f"📘 [Node] 发现 {len(md_files)} 个解析文件，准备执行增量生成...")

    for md_filename in sorted(md_files):
        # 提取编号，如从 rule_1.md 提取 rule_1
        base_name = os.path.splitext(md_filename)[0] 
        md_path = os.path.join(rule_md_dir, md_filename)
        
        # --- 🔍 增量判定：检查目标文件是否已存在 ---
        pos_path = os.path.join(scala_t_dir, f"{base_name}_T.scala")
        neg_path = os.path.join(scala_f_dir, f"{base_name}_F.scala")
        
        need_t = not os.path.exists(pos_path)
        need_f = not os.path.exists(neg_path)

        if not need_t and not need_f:
            print(f"⏭️  [Agent] {base_name} 的正反例代码均已存在，跳过 LLM 生成。")
            continue

        # --- 🚀 执行生成 ---
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                rule_md_text = f.read()

            full_prompt = build_prompt(rule_md_text) # 假设 build_prompt 已在上下文定义
            
            # 使用 model.device 确保数据分发到正确的 GPU
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            print(f"🧩 [Agent] 正在为 {md_filename} 生成缺失代码 (T:{need_t}, F:{need_f})...")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # 清洗并提取代码
            cleaned_output = clean_markdown_blocks(output_text) # 假设已在上下文定义
            pos_code, neg_code = extract_examples(cleaned_output) # 假设已在上下文定义

            # --- 💾 原子化写入：只写缺失的文件 ---
            if need_t:
                with open(pos_path, "w", encoding="utf-8") as f:
                    content = pos_code if (pos_code and pos_code.strip()) else PLACEHOLDER
                    f.write(content + "\n")
                print(f"    ✅ 已生成正例: {os.path.basename(pos_path)}")

            if need_f:
                with open(neg_path, "w", encoding="utf-8") as f:
                    content = neg_code if (neg_code and neg_code.strip()) else PLACEHOLDER
                    f.write(content + "\n")
                print(f"    ✅ 已生成反例: {os.path.basename(neg_path)}")

        except Exception as e:
            print(f"❌ [Agent] {base_name} 生成过程出错: {str(e)}")
            if "CUDA" in str(e):
                raise e

    print(f"🏁 [Agent] Iteration 任务处理完毕。")

# if __name__ == "__main__" 部分在共享模式下已废弃