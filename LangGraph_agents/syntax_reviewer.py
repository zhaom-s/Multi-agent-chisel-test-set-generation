# /root/Test-Agent/LangGraph_agents/syntax_reviewer.py
"""
语法专家 Agent：在提交编译器验证之前，用 LLM 模拟 Scala 编译器视角，
对生成的 Chisel 代码进行语法预检，输出简短的修复建议。

核心价值：
  - 在真实编译之前发现明显的 API 错误（虚构 API、错误 import 等）
  - 输出的建议会和规则解析 md 一起，作为 ReviewedCodeGen 的输入
  - 不替代编译器，只做"第一道防线"

输入:
  - rule_ids     : 需要评审的规则 ID 列表
  - scala_t_dir  : 正例目录
  - scala_f_dir  : 反例目录
  - model/tokenizer

输出:
  - syntax_advice: {rule_id: "建议字符串"}
"""

import os
import re
import torch


# ============================================================
# 已知的虚构/非标准 API 黑名单（从历史错误中总结）
# ============================================================
KNOWN_FAKE_APIS = [
    "XSDebug", "XSError", "XSWarn",
    "chisel3.util.experimental.BaseKey",
    "chisel3.util.experimental.Parameter",
    "chisel3.experimental.ChiselEnum",   # 应从 chisel3.util 导入
    "chisel3.experimental.FixedPoint",   # 应从 chisel3.experimental 导入（版本相关）
    "chisel3.experimental.debug",
    "Float(", "FixedPoint(",             # 常见类型误用
]

BLACKLIST_PATTERN = re.compile(
    r"(XSDebug|XSError|XSWarn"
    r"|chisel3\.util\.experimental\.(BaseKey|Parameter)"
    r"|chisel3\.experimental\.(ChiselEnum|debug)"
    r"|Float\s*\(\s*\d+\.W\s*\)"
    r"|FixedPoint\s*\()"
)


def _quick_static_check(code: str) -> list:
    """
    快速静态扫描：不调用 LLM，直接检测已知黑名单 API。
    返回发现的问题列表（字符串）。
    """
    issues = []
    for line_no, line in enumerate(code.splitlines(), 1):
        m = BLACKLIST_PATTERN.search(line)
        if m:
            issues.append(f"第{line_no}行: 使用了非标准/虚构 API `{m.group()}`，请替换为 chisel3 标准库")
    # 检查 Wire() 无参数调用
    if re.search(r"Wire\s*\(\s*\)", code):
        issues.append("存在 Wire() 无类型参数调用，应改为 Wire(UInt(n.W)) 等完整形式")
    # 检查 implicit 关键字
    if re.search(r"\bimplicit\b", code):
        issues.append("存在 implicit 关键字，属于黑名单，请改用显式参数传递")
    return issues


def _build_syntax_review_prompt(code: str, code_type: str, static_issues: list) -> str:
    label = "正例" if code_type == "T" else "反例"
    static_hint = ""
    if static_issues:
        static_hint = "\n**静态扫描已发现以下问题（必须修复）**:\n" + "\n".join(f"- {i}" for i in static_issues) + "\n"

    return f"""
# Role
你是一名 Chisel 3.x / Scala 2.x 编译器专家。你的任务是对下面这段 Chisel {label}代码进行语法预检，
找出会导致编译失败的问题，并给出简短的修复建议（不超过3条）。

# 检查重点
1. import 语句是否引用了不存在的包或类（如 `chisel3.util.experimental.BaseKey`、`chisel3.experimental.debug.XSDebug`）
2. 类型使用是否正确（如 `Float(32.W)` 是错误的，应使用 `UInt`/`SInt`/`FixedPoint`）
3. 方法调用是否完整（如 `Wire()` 缺少类型参数）
4. 是否使用了黑名单 API：`XSDebug`、`XSError`、`XSWarn`、`implicit`、`lazy val`、`Option`
5. 是否有明显的类型不匹配（如把 `chisel3.Bool` 用在需要 `Boolean` 的地方）
{static_hint}
# 待检查代码
```scala
{code[:2500]}
```

# 输出要求
- 如果代码没有明显语法问题，输出：`无明显语法问题`
- 如果有问题，输出不超过3条简短建议，每条一行，格式：`[语法] 问题描述 → 修复方向`
- 禁止输出修复后的完整代码，只输出建议文字

请直接输出建议，不要有任何前置说明：
"""


def _read_file_safe(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def run_syntax_review(
    rule_ids: list,
    scala_t_dir: str,
    scala_f_dir: str,
    model: torch.nn.Module,
    tokenizer: object,
) -> dict:
    """
    对指定规则的正/反例代码进行语法预检。
    返回 {rule_id: "综合语法建议字符串"}
    """
    syntax_advice = {}
    print(f"\n🔬 [SyntaxReviewer] 开始语法预检，共 {len(rule_ids)} 条规则...")

    for rule_id in rule_ids:
        advices = []

        for code_type, scala_dir in [("T", scala_t_dir), ("F", scala_f_dir)]:
            file_path = os.path.join(scala_dir, f"rule_{rule_id}_{code_type}.scala")
            code = _read_file_safe(file_path)
            if not code or "⚠️ 生成失败" in code:
                continue

            # 1. 快速静态扫描（零 LLM 开销）
            static_issues = _quick_static_check(code)

            # 2. 如果静态扫描已发现问题，直接用静态结果（节省 LLM 调用）
            if static_issues and len(static_issues) >= 2:
                label = "正例" if code_type == "T" else "反例"
                advice_text = f"[{label}] " + "; ".join(static_issues[:3])
                advices.append(advice_text)
                print(f"  ⚡ [SyntaxReviewer] Rule {rule_id} {label}: 静态扫描发现 {len(static_issues)} 个问题")
                continue

            # 3. 调用 LLM 做深度语法检查
            prompt = _build_syntax_review_prompt(code, code_type, static_issues)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=True,
                    )
                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                advice_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                # 截断过长输出
                advice_text = advice_text[:400]
                label = "正例" if code_type == "T" else "反例"
                advices.append(f"[{label}] {advice_text}")
                print(f"  ✅ [SyntaxReviewer] Rule {rule_id} {label}: 评审完成")
            except Exception as e:
                print(f"  ⚠️ [SyntaxReviewer] Rule {rule_id} {code_type} 评审异常: {e}")
                if "CUDA" in str(e):
                    raise e

        syntax_advice[rule_id] = "\n".join(advices) if advices else "无明显语法问题"

    print(f"🏁 [SyntaxReviewer] 语法预检完成，共 {len(syntax_advice)} 条规则")
    return syntax_advice
