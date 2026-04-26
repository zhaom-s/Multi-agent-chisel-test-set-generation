import os
import re
import torch


# ==========================================================
#  prompt构建
# ==========================================================

USER_PROMPT = """
你是一名资深的 Chisel 硬件验证工程师。
你的任务是根据给定的「规则可编码行为解析文件」，生成 **两段完全独立、可被前端成功解析的 Chisel 代码**（正例 / 反例）。

【核心策略：多模块拆分】
为了保证代码清晰且无语法错误，**请不要**将所有测试点写在一个模块里。请生成以下三个独立的类：
1. 正例的一个类：`class CompliantModule`
   - 专门用于测试 **合规与边界场景**（标准写法、易误判写法）
2. 反例的第一个类：`class CoreViolationModule`
   - 专门用于测试 **核心报错场景**（最直接、最典型的违规）
3. 反例的第二个类：`class HiddenViolationModule`
   - 专门用于测试 **隐蔽报错场景**（私有作用域、嵌套、别名等）


==================================================
【文件结构与 Import 的绝对约束（必须遵守）】
==================================================

1. 【禁止 package】
   - **严禁出现任何 package 声明**
   - 输出必须是一个“无 package 的顶层 Scala 文件”

2. 【Import 白名单（只能使用下面两条）】
   - 只允许且必须使用：
     import chisel3._
     import chisel3.util._
   - **禁止出现任何其他 import**
     （包括但不限于 scala.io、scala.reflect、java.io、Source、util 等）

3. 【违规测试的构造方式限制】
   - 违规必须通过 **代码结构 / 语义本身** 触发
   - 不允许通过以下方式制造违规：
     - 文件 IO
     - 资源加载
     - 反射
     - 运行时库调用
   - 若某种写法需要额外 import 才能成立，请放弃该写法，改用结构等价的方式


==================================================
【语法防火墙 (Syntax Firewall) - 必须严格遵守】
==================================================

之前的代码常因混淆软件 / 硬件概念导致编译失败。请牢记：

1. **严禁跨界赋值**
   - Scala 运行时对象（如 `String`, `Type`, `Symbol`, `Int`）
     **绝对不能**直接连接到 Chisel 硬件端口（如 `io.out`）
   - 错误示例：
     io.out := someScalaValue
   - 正确做法：
     val _ = someScalaValue   // 仅用于触发规则检查，不连接硬件

2. **严禁虚构方法**
   - Scala 的 `String` 类型没有 `.asUInt` 等方法，请勿捏造 API

3. **必须使用 new**
   - 实例化普通类时，必须使用 `new` 关键字
     （如：`val x = new MyClass()`）

4. **禁止“为了更真实”而引入额外 Scala 依赖**
   - 这是规则测试代码，不是工程示例代码


==================================================
【输出严格约束（违反即视为错误输出）】
==================================================

1. **只输出代码**
   - 严禁包含 markdown 标记
   - 严禁包含解释性文字或标题（如 Generated Code）

2. **注释规范**
   - 每个测试点必须明确标注：
     - `// [预期通过]`
     - 或 `// [预期报错 X]`

3. **结构要求**
   - 必须包含：
     - 一段 `//正例`
     - 一段 `//反例`
   - 反例中必须至少包含一条：
     `// 违反规则：...`

4. **停止机制**
   - 代码最后一行必须是：
     `// === End of Test File ===`


================================================================
========================【严格遵循输出风格示例】========================
================================================================

------------------------- 示例1 -------------------------
//正例
import chisel3._
import chisel3.util._

// 模块 1: 合规与边界测试
class CompliantModule extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })

  // [预期通过] 标准合规：使用普通 Wrapper 类
  class ExplicitWrapper(u: UInt) {
    def addOne: UInt = u + 1.U
  }

  // [预期通过] 边界测试：隐式参数是合规的（不是隐式类）
  def addWithImplicit(x: UInt)(implicit v: Int): UInt = x + v.U

  // 逻辑实现
  val wrapper = new ExplicitWrapper(io.in)
  implicit val inc: Int = 1
  io.out := wrapper.addOne + addWithImplicit(io.in)
}


//反例
import chisel3._
import chisel3.util._


// 模块 1: 核心违规测试
class CoreViolationModule extends Module {
  val io = IO(new Bundle {
    val out = Output(Bool())
  })

  // [预期报错 1] 核心违规：定义隐式类扩展硬件类型
  implicit class BadExtension(u: UInt) {
    def isAllOnes: Bool = u === 255.U
  }

  // 触发逻辑：只要定义即违规
  io.out := false.B
}

// 模块 2: 隐蔽违规测试
class HiddenViolationModule extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(8.W))
    val out = Output(Bool())
  })

  // [预期报错 2] 隐蔽违规：私有作用域中的隐式类
  private implicit class HiddenOps(b: Bool) {
    def unsafeInt: Int = if (b.litToBoolean) 1 else 0
  }

  io.out := true.B
}

------------------------- 示例2 -------------------------
//正例
import chisel3._
import chisel3.util._

// 安全域接口定义
class SecureDomainInterface extends Bundle {
  val publicData = UInt(8.W)   // 公开数据
  val privateState = UInt(16.W) // 私有状态（不应暴露）
  val controlSignal = Bool()    // 控制信号
}

class PublicDomainInterface extends Bundle {
  val publicData = UInt(8.W)   // 公开数据
  val controlSignal = Bool()    // 控制信号
}

// 模块 1: 安全连接实现
class SecureConnectionModule extends Module {
  val io = IO(new Bundle {
    val highSecurityIn = Input(new SecureDomainInterface)
    val lowSecurityOut = Output(new PublicDomainInterface)
  })

  // [预期通过] 安全连接：手动映射，仅传递授权字段
  io.lowSecurityOut.publicData := io.highSecurityIn.publicData
  io.lowSecurityOut.controlSignal := io.highSecurityIn.controlSignal
  // privateState 字段被明确忽略，不会泄露
}

// 模块 2: 完备Switch语句实现
class CompleteSwitchModule extends Module {
  val io = IO(new Bundle {
    val selector = Input(UInt(2.W))
    val output = Output(UInt(8.W))
  })

  val stateReg = RegInit(0.U(8.W))

  // [预期通过] 完备Switch：所有合法case覆盖，无重叠
  switch(io.selector) {
    is(0.U) { stateReg := 10.U }  // case 0
    is(1.U) { stateReg := 20.U }  // case 1
    is(2.U) { stateReg := 30.U }  // case 2
    is(3.U) { stateReg := 40.U }  // case 3 - 覆盖所有2位输入
  }
  io.output := stateReg
}

//反例
import chisel3._
import chisel3.util._

// 模块 1: 不安全的<>连接
class InsecureDomainInterface extends Bundle {
  val publicData = UInt(8.W)
  val privateState = UInt(16.W)  // 私有状态字段
  val controlSignal = Bool()
}

class PublicTargetInterface extends Bundle {
  val publicData = UInt(8.W)
  val privateState = UInt(16.W)  // 接收方意外接收私有状态
  val controlSignal = Bool()
}

class UnsafeConnectionModule extends Module {
  val io = IO(new Bundle {
    val highSecurityIn = Input(new InsecureDomainInterface)
    val lowSecurityOut = Output(new PublicTargetInterface)
  })

  // [预期报错 1] 不安全连接：直接使用<>连接，可能导致私有状态泄露
  val tempBundle = Wire(new PublicTargetInterface)
  tempBundle.publicData := io.highSecurityIn.publicData
  tempBundle.privateState := io.highSecurityIn.privateState  // 危险：私有状态被传出
  tempBundle.controlSignal := io.highSecurityIn.controlSignal
  
  io.lowSecurityOut <> tempBundle
}

// 模块 2: 不完备Switch语句
class IncompleteSwitchModule extends Module {
  val io = IO(new Bundle {
    val selector = Input(UInt(3.W))  // 3位输入，理论上应有8种状态
    val output = Output(UInt(8.W))
  })

  val stateReg = RegInit(0.U(8.W))

  // [预期报错 2] 不完备Switch：只处理了部分case，缺少default或完整覆盖
  switch(io.selector) {
    is(0.U) { stateReg := 10.U }
    is(1.U) { stateReg := 20.U }
    is(2.U) { stateReg := 30.U }
    // 缺少其他5种状态的处理 (3-7)，也无default分支
    // 这会导致某些输入下stateReg保持不变，产生不确定行为
  }
  io.output := stateReg
}

// 模块 3: 重叠Case Switch（故意优先级编码除外的情况）
class OverlappingCaseModule extends Module {
  val io = IO(new Bundle {
    val input = Input(UInt(2.W))
    val output = Output(UInt(8.W))
  })

  val result = Wire(UInt(8.W))

  // [预期报错 3] 重叠Case：两个case可能同时匹配某些输入值
  switch(io.input) {
    is(0.U) { result := 99.U }
    is(1.U) { result := 88.U }
    is(0.U) { result := 77.U }  // 错误：重复case 0，违反互斥原则
    is(3.U) { result := 66.U }
  }
  io.output := result
}



================================================================
==========================【任务输入】============================
================================================================

你将收到一份「规则的解析蓝图」，请严格依照其内容生成：

- 一段 //正例
- 一段 //反例（必须包含“// 违反规则：...”）

除代码本身外，不得输出任何内容。
"""


def build_prompt(rule_md_text: str) -> str:
    return f"""{USER_PROMPT}

以下是规则的解析蓝图：
----------------------------------------
{rule_md_text}
----------------------------------------

- 一段 //正例
- 一段 //反例（必须包含“// 违反规则：...”）

除代码本身外，不得输出任何内容。
"""


def build_retry_prompt(rule_md_text: str, rag_context: str = "") -> str:
    ref_section = f"\n【RAG 参考代码库】\n{rag_context}\n" if rag_context else ""
    
    # 显式引导模型从 //正例 开始输出
    return f"""{USER_PROMPT}
{ref_section}
以下是规则的解析蓝图：
----------------------------------------
{rule_md_text}
----------------------------------------

请根据蓝图直接开始生成两段代码。
必须严格以 "//正例" 作为输出的第一行，然后输出第一段代码。
必须在正例结束后紧跟 "//反例"，开始第二段的代码输出。
"""

def clean_markdown_blocks(text: str) -> str:
    return text.replace("```markdown", "").replace("```json", "").replace("```scala", "").replace("```chisel", "").replace("```", "").strip()

def extract_examples(output_text: str):
    # 先尝试标准正则
    pos_match = re.search(r"(//\s*正例|正例[:：])\s*(.*?)(?=(//\s*反例|反例[:：]|$))", output_text, re.DOTALL)
    neg_match = re.search(r"(//\s*反例|反例[:：])\s*(.*)", output_text, re.DOTALL)
    
    pos_code = pos_match.group(2).strip() if pos_match else ""
    neg_code = neg_match.group(2).strip() if neg_match else ""
    
    # --- 【保底逻辑】 ---
    # 如果正例为空但反例有内容，且反例之前有很长的文本，那很可能正例就在那段文本里
    if not pos_code and neg_code:
        # 寻找“反例”标签的位置
        neg_start = re.search(r"(//\s*反例|反例[:：])", output_text)
        if neg_start:
            pre_text = output_text[:neg_start.start()].strip()
            # 如果预留文本里包含 import，说明这其实就是没贴标签的正例
            if "import chisel3" in pre_text:
                pos_code = pre_text
                
    return pos_code, neg_code

def run_retry_code_generate(
    rule_md_dir: str, 
    scala_t_dir: str, 
    scala_f_dir: str, 
    model: torch.nn.Module, 
    tokenizer: object, 
    failed_map: dict = None,  # 改为可选
    rag=None, 
    mode="rag_fix",
    **kwargs,
):
    """
    全量/增量兼容的代码生成引擎。
    如果 failed_map 为 None，则进入全量模式，扫描 rule_md_dir 下所有 .md 文件。
    """
    os.makedirs(scala_t_dir, exist_ok=True)
    os.makedirs(scala_f_dir, exist_ok=True)
    PLACEHOLDER = " ⚠️ 重试生成失败\n"

    # --- 逻辑核心：确定任务目标 ---
    task_rules = []
    if failed_map:
        # 增量模式：只处理失败的
        for rid, targets in failed_map.items():
            task_rules.append((rid, targets))
        print(f"🩹 [Retry-CodeGen] 增量修复模式：共 {len(task_rules)} 条规则")
    else:
        # 全量模式：扫描文件夹下所有 rule_n.md
        for fname in os.listdir(rule_md_dir):
            if fname.endswith(".md"):
                m = re.search(r"rule_(\d+)", fname)
                if m:
                    # 全量模式下，默认 T 和 F 都要生成
                    task_rules.append((int(m.group(1)), ["T", "F"]))
        task_rules.sort(key=lambda x: x[0])
        print(f"🚀 [Retry-CodeGen] 全量强化生成模式：共 {len(task_rules)} 条规则")

    # --- 开始循环处理 ---
    for rule_idx, targets in task_rules:
        rule_md_file = os.path.join(rule_md_dir, f"rule_{rule_idx}.md")
        if not os.path.exists(rule_md_file): 
            continue
        
        with open(rule_md_file, "r", encoding="utf-8") as f:
            rule_md_text = f.read()

        # RAG 检索 (逻辑保持)
        context = ""
        if rag:
            try:
                query = rag.extract_query_from_blueprint(rule_md_text)
                res = rag.search_context(query, filter_type="code", top_k=2)
                context = res.get("context_str", "")
            except Exception as e:
                print(f"⚠️ RAG 检索异常 (rule_{rule_idx}): {e}")

        # 构造强化版 Prompt
        prompt = build_retry_prompt(rule_md_text, context)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 执行推理 (调高 tokens 限制，因为三个 Module 长度较长)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=2500, # 强化版三个类，上限提高到 2500
                temperature=0.2, 
                top_p=0.9, 
                do_sample=True,
                repetition_penalty=1.0
            )

        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pos_code, neg_code = extract_examples(clean_markdown_blocks(decoded))

        # --- 写入文件 ---
        if "T" in targets:
            path_t = os.path.join(scala_t_dir, f"rule_{rule_idx}_T.scala")
            with open(path_t, "w", encoding="utf-8") as f:
                f.write((pos_code if pos_code else PLACEHOLDER) + "\n")
        
        if "F" in targets:
            path_f = os.path.join(scala_f_dir, f"rule_{rule_idx}_F.scala")
            with open(path_f, "w", encoding="utf-8") as f:
                f.write((neg_code if neg_code else PLACEHOLDER) + "\n")
        
        print(f"✅ [Retry-CodeGen] Rule {rule_idx} 完成 (Targets: {targets})")

    print(f"🏁 [Retry-CodeGen] 任务流水线结束。")



# USER_PROMPT = """
# 你是一名资深的 Chisel 硬件验证工程师。
# 你的任务是根据给定的「规则可编码行为解析文件」，生成 **两段完全独立、可被前端成功解析的 Chisel 代码**（正例 / 反例）。

# 【核心策略：多模块拆分】
# 为了保证代码清晰且无语法错误，**请不要**将所有测试点写在一个模块里。请生成以下三个独立的类：
# 1. 正例的一个类：`class CompliantModule`
#    - 专门用于测试 **合规与边界场景**（标准写法、易误判写法）
# 2. 反例的第一个类：`class CoreViolationModule`
#    - 专门用于测试 **核心报错场景**（最直接、最典型的违规）
# 3. 反例的第二个类：`class HiddenViolationModule`
#    - 专门用于测试 **隐蔽报错场景**（私有作用域、嵌套、别名等）


# ==================================================
# 【文件结构与 Import 的绝对约束（必须遵守）】
# ==================================================

# 1. 【禁止 package】
#    - **严禁出现任何 package 声明**
#    - 输出必须是一个“无 package 的顶层 Scala 文件”

# 2. 【Import 白名单（只能使用下面两条）】
#    - 只允许且必须使用：
#      import chisel3._
#      import chisel3.util._
#    - **禁止出现任何其他 import**
#      （包括但不限于 scala.io、scala.reflect、java.io、Source、util 等）



# 提示：接下来你将会看到一个严格遵循输出风格示例：它只是为了告诉你输出代码的风格（正例 + 反例），严禁你直接照抄输出。
#      例如你在反例代码中：//[预期报错 1]...   和//[预期报错 2].. 这两个地方的代码构造应该是结合规则解析蓝图输出该规则特定的报错，严禁直接照抄风格示例代码。

# ================================================================
# ========================【严格遵循输出风格示例】========================
# ================================================================

# ------------------------- 示例1 -------------------------
# //正例
# import chisel3._
# import chisel3.util._

# // 模块 1: 合规与边界测试
# class CompliantModule extends Module {
#   val io = IO(new Bundle {
#     val in  = Input(UInt(8.W))
#     val out = Output(UInt(8.W))
#   })

#   // [预期通过] 标准合规：使用普通 Wrapper 类
#   class ExplicitWrapper(u: UInt) {
#     def addOne: UInt = u + 1.U
#   }

#   // [预期通过] 边界测试：隐式参数是合规的（不是隐式类）
#   def addWithImplicit(x: UInt)(implicit v: Int): UInt = x + v.U

#   // 逻辑实现
#   val wrapper = new ExplicitWrapper(io.in)
#   implicit val inc: Int = 1
#   io.out := wrapper.addOne + addWithImplicit(io.in)
# }


# //反例
# import chisel3._
# import chisel3.util._


# // 模块 1: 核心违规测试
# class CoreViolationModule extends Module {
#   val io = IO(new Bundle {
#     val out = Output(Bool())
#   })

#   // [预期报错 1] 核心违规：定义隐式类扩展硬件类型
#   implicit class BadExtension(u: UInt) {
#     def isAllOnes: Bool = u === 255.U
#   }

#   // 触发逻辑：只要定义即违规
#   io.out := false.B
# }

# // 模块 2: 隐蔽违规测试
# class HiddenViolationModule extends Module {
#   val io = IO(new Bundle {
#     val in  = Input(UInt(8.W))
#     val out = Output(Bool())
#   })

#   // [预期报错 2] 隐蔽违规：私有作用域中的隐式类
#   private implicit class HiddenOps(b: Bool) {
#     def unsafeInt: Int = if (b.litToBoolean) 1 else 0
#   }

#   io.out := true.B
# }

# ------------------------- 示例2 -------------------------
# //正例
# import chisel3._
# import chisel3.util._

# // 安全域接口定义
# class SecureDomainInterface extends Bundle {
#   val publicData = UInt(8.W)   // 公开数据
#   val privateState = UInt(16.W) // 私有状态（不应暴露）
#   val controlSignal = Bool()    // 控制信号
# }

# class PublicDomainInterface extends Bundle {
#   val publicData = UInt(8.W)   // 公开数据
#   val controlSignal = Bool()    // 控制信号
# }

# // 模块 1: 安全连接实现
# class SecureConnectionModule extends Module {
#   val io = IO(new Bundle {
#     val highSecurityIn = Input(new SecureDomainInterface)
#     val lowSecurityOut = Output(new PublicDomainInterface)
#   })

#   // [预期通过] 安全连接：手动映射，仅传递授权字段
#   io.lowSecurityOut.publicData := io.highSecurityIn.publicData
#   io.lowSecurityOut.controlSignal := io.highSecurityIn.controlSignal
#   // privateState 字段被明确忽略，不会泄露
# }

# // 模块 2: 完备Switch语句实现
# class CompleteSwitchModule extends Module {
#   val io = IO(new Bundle {
#     val selector = Input(UInt(2.W))
#     val output = Output(UInt(8.W))
#   })

#   val stateReg = RegInit(0.U(8.W))

#   // [预期通过] 完备Switch：所有合法case覆盖，无重叠
#   switch(io.selector) {
#     is(0.U) { stateReg := 10.U }  // case 0
#     is(1.U) { stateReg := 20.U }  // case 1
#     is(2.U) { stateReg := 30.U }  // case 2
#     is(3.U) { stateReg := 40.U }  // case 3 - 覆盖所有2位输入
#   }
#   io.output := stateReg
# }

# //反例
# import chisel3._
# import chisel3.util._

# // 模块 1: 不安全的<>连接
# class InsecureDomainInterface extends Bundle {
#   val publicData = UInt(8.W)
#   val privateState = UInt(16.W)  // 私有状态字段
#   val controlSignal = Bool()
# }

# class PublicTargetInterface extends Bundle {
#   val publicData = UInt(8.W)
#   val privateState = UInt(16.W)  // 接收方意外接收私有状态
#   val controlSignal = Bool()
# }

# class UnsafeConnectionModule extends Module {
#   val io = IO(new Bundle {
#     val highSecurityIn = Input(new InsecureDomainInterface)
#     val lowSecurityOut = Output(new PublicTargetInterface)
#   })

#   // [预期报错 1] 不安全连接：直接使用<>连接，可能导致私有状态泄露
#   val tempBundle = Wire(new PublicTargetInterface)
#   tempBundle.publicData := io.highSecurityIn.publicData
#   tempBundle.privateState := io.highSecurityIn.privateState  // 危险：私有状态被传出
#   tempBundle.controlSignal := io.highSecurityIn.controlSignal
  
#   io.lowSecurityOut <> tempBundle
# }

# // 模块 2: 不完备Switch语句
# class IncompleteSwitchModule extends Module {
#   val io = IO(new Bundle {
#     val selector = Input(UInt(3.W))  // 3位输入，理论上应有8种状态
#     val output = Output(UInt(8.W))
#   })

#   val stateReg = RegInit(0.U(8.W))

#   // [预期报错 2] 不完备Switch：只处理了部分case，缺少default或完整覆盖
#   switch(io.selector) {
#     is(0.U) { stateReg := 10.U }
#     is(1.U) { stateReg := 20.U }
#     is(2.U) { stateReg := 30.U }
#     // 缺少其他5种状态的处理 (3-7)，也无default分支
#     // 这会导致某些输入下stateReg保持不变，产生不确定行为
#   }
#   io.output := stateReg
# }

# // 模块 3: 重叠Case Switch（故意优先级编码除外的情况）
# class OverlappingCaseModule extends Module {
#   val io = IO(new Bundle {
#     val input = Input(UInt(2.W))
#     val output = Output(UInt(8.W))
#   })

#   val result = Wire(UInt(8.W))

#   // [预期报错 3] 重叠Case：两个case可能同时匹配某些输入值
#   switch(io.input) {
#     is(0.U) { result := 99.U }
#     is(1.U) { result := 88.U }
#     is(0.U) { result := 77.U }  // 错误：重复case 0，违反互斥原则
#     is(3.U) { result := 66.U }
#   }
#   io.output := result
# }



# ================================================================
# ==========================【任务输入】============================
# ================================================================

# 你将收到一份「规则的解析蓝图」，请严格依照其内容生成：

# - 一段 //正例
# - 一段 //反例（必须包含“// 违反规则：...”）

# 除代码本身外，不得输出任何内容。
# 严禁：直接照抄示例代码里的代码内容，你应该参照解析蓝图自己构建代码。
#      例如你在反例代码中：//[预期报错 1]...   和//[预期报错 2].. 这两个地方的代码构造应该是结合规则解析蓝图输出该规则特定的报错，严禁直接照抄风格示例代码。
# """
