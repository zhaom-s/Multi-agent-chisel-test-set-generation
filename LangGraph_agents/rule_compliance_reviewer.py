# /root/Test-Agent/LangGraph_agents/rule_compliance_reviewer.py
"""
规则合规专家 Agent：检查生成的正/反例代码是否真正符合规则意图。

核心价值：
  - 正例检查：代码是否真正遵守了规则描述的推荐行为
  - 反例检查：代码是否真正违反了规则，且违反点是否有 `// 违反规则：...` 注释
  - 输出的建议会和语法建议一起，作为 ReviewedCodeGen 的输入

输入:
  - rule_ids     : 需要评审的规则 ID 列表
  - scala_t_dir  : 正例目录
  - scala_f_dir  : 反例目录
  - rule_md_dir  : 规则解析 md 目录
  - model/tokenizer

输出:
  - compliance_advice: {rule_id: "合规建议字符串"}
"""

import os
import re
import torch


def _read_file_safe(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _build_compliance_prompt(
    rule_md_text: str,
    pos_code: str,
    neg_code: str,
) -> str:
    return f"""
# Role
你是一名 Chisel 硬件设计规则审查专家。你的任务是检查给定的正例和反例代码是否真正符合规则要求，
并给出简短的改进建议（每类最多2条）。

# 规则解析文档（规则的权威来源）
```
{rule_md_text[:1200]}
```

# 待审查：正例代码（应遵守规则）
```scala
{pos_code[:1200]}
```

# 待审查：反例代码（应违反规则）
```scala
{neg_code[:1200]}
```

# 审查任务
**正例审查**：
- 正例是否真正体现了规则要求的推荐行为？
- 正例中是否存在意外违反规则的代码？

**反例审查**：
- 反例是否真正违反了规则（而不是随机写错）？
- 违反点是否有 `// 违反规则：...` 注释标注？
- 反例的 Scala 语法本身是否合法（反例只需违反规则，不能有语法错误）？

# 输出要求
- 如果正例和反例都符合要求，输出：`正反例均符合规则要求`
- 如果有问题，分别输出：
  `[正例] 问题描述 → 修复方向`
  `[反例] 问题描述 → 修复方向`
- 每类最多2条，每条不超过50字
- 禁止输出修复后的完整代码

请直接输出建议，不要有任何前置说明：
"""


def run_compliance_review(
    rule_ids: list,
    scala_t_dir: str,
    scala_f_dir: str,
    rule_md_dir: str,
    model: torch.nn.Module,
    tokenizer: object,
) -> dict:
    """
    对指定规则的正/反例代码进行规则合规性检查。
    返回 {rule_id: "合规建议字符串"}
    """
    compliance_advice = {}
    print(f"\n📋 [ComplianceReviewer] 开始合规审查，共 {len(rule_ids)} 条规则...")

    for rule_id in rule_ids:
        # 读取规则解析 md
        md_path = os.path.join(rule_md_dir, f"rule_{rule_id}.md")
        rule_md_text = _read_file_safe(md_path)
        if not rule_md_text:
            compliance_advice[rule_id] = "缺少规则解析文档，跳过合规审查"
            continue

        pos_code = _read_file_safe(os.path.join(scala_t_dir, f"rule_{rule_id}_T.scala"))
        neg_code = _read_file_safe(os.path.join(scala_f_dir, f"rule_{rule_id}_F.scala"))

        if not pos_code and not neg_code:
            compliance_advice[rule_id] = "正反例文件均不存在，跳过"
            continue

        # 快速检查：反例是否有违规注释
        quick_issues = []
        if neg_code and "// 违反规则" not in neg_code and "//违反规则" not in neg_code:
            quick_issues.append("[反例] 缺少 `// 违反规则：...` 注释，请在违规行添加")

        # 如果只有快速问题且不需要深度检查，直接返回
        if quick_issues and not pos_code:
            compliance_advice[rule_id] = "\n".join(quick_issues)
            continue

        # 调用 LLM 做深度合规检查
        prompt = _build_compliance_prompt(
            rule_md_text,
            pos_code or "（文件不存在）",
            neg_code or "（文件不存在）",
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                )
            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            advice_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            advice_text = advice_text[:500]

            # 合并快速检查结果
            if quick_issues:
                advice_text = "\n".join(quick_issues) + "\n" + advice_text

            compliance_advice[rule_id] = advice_text
            print(f"  ✅ [ComplianceReviewer] Rule {rule_id}: 合规审查完成")

        except Exception as e:
            print(f"  ⚠️ [ComplianceReviewer] Rule {rule_id} 审查异常: {e}")
            if "CUDA" in str(e):
                raise e
            compliance_advice[rule_id] = "\n".join(quick_issues) if quick_issues else "审查异常，跳过"

    print(f"🏁 [ComplianceReviewer] 合规审查完成，共 {len(compliance_advice)} 条规则")
    return compliance_advice
