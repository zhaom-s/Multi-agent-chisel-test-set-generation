# /root/Test-Agent/LangGraph_agents/repair_agent.py
"""
Repair Agent：接收编译失败的代码 + 错误信息，定向修复后重新写入文件。

与重新生成的核心区别：
  - 重新生成：重走 rule_analysis → code_generate 完整流程
  - 定向修复：保留原始规则解析，仅针对编译错误修改代码本身
            → 更快、更精准、体现 Agent 自我修复能力（论文亮点）

输入（由 nodes.py 调用时传入）:
  - repair_ids     : 需要修复的规则 ID 列表
  - iter_dir       : 当前迭代目录（含 scala_T/scala_F）
  - rule_md_dir    : 规则解析 md 目录
  - error_context  : {rule_id: {error_type, message, file_path, code_type}}
  - model/tokenizer: 共享 LLM 资源

输出：
  - 直接将修复后的代码写回原 .scala 文件
  - 返回 {rule_id: "success"/"failed"} 修复结果摘要
"""

import os
import re
import torch


# ============================================================
# Prompt 构造
# ============================================================

def _build_repair_prompt(
    rule_md_text: str,
    broken_code: str,
    error_message: str,
    code_type: str,  # "T"(正例) or "F"(反例)
) -> str:
    code_label = "正例" if code_type == "T" else "反例"
    return f"""
# Role
你是一名 Chisel 代码修复专家。你的任务是根据编译器报告的错误信息，精准修复一段 Chisel/{code_label}代码，使其能通过 Scala 编译并满足原始规则要求。

# 原始规则解析（规则约束来源）
```
{rule_md_text[:1500]}
```

# 待修复代码（{code_label}）
```scala
{broken_code[:2000]}
```

# 编译器错误信息
```
{error_message[:800]}
```

# 修复要求
1. **仅修复错误**：保留代码的整体结构和意图，不要大幅重写
2. **保持规则符合性**：
   - 若为正例（//正例）：修复后必须仍然符合规则要求
   - 若为反例（//反例）：修复后 Scala 语法须合法，但仍须违反规则（保留 // 违反规则：... 注释）
3. **可编译**：修复后必须是完整、可独立编译的 Scala 文件
4. **禁止使用**：Option / implicit / lazy val / RawModule / MultiIOModule / chiseltest
5. **必须包含**: `import chisel3._`

# 输出格式
直接输出修复后的完整 Scala 代码，以 `//{code_label}` 开头，不含任何解释文字。

//{code_label}
"""


# ============================================================
# 工具函数
# ============================================================

def _read_file_safe(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _write_file_safe(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def _clean_output(text: str) -> str:
    return (
        text.replace("```scala", "").replace("```chisel", "")
        .replace("```", "").strip()
    )


def _extract_repaired_code(text: str, code_type: str) -> str:
    """从 LLM 输出中提取修复后的代码块"""
    label = "正例" if code_type == "T" else "反例"
    match = re.search(
        rf"(//\s*{label}[\s\S]*)",
        text
    )
    if match:
        return match.group(1).strip()
    # 若没有匹配到标签，直接返回清洗后的全文
    return _clean_output(text)


# ============================================================
# 核心：单文件修复
# ============================================================

def _repair_single_file(
    file_path: str,
    rule_md_text: str,
    error_message: str,
    code_type: str,
    model: torch.nn.Module,
    tokenizer: object,
) -> bool:
    broken_code = _read_file_safe(file_path)
    if not broken_code or "⚠️ 生成失败" in broken_code:
        print(f"    ⚠️ [Repair] 原代码为空或占位符，跳过修复: {os.path.basename(file_path)}")
        return False

    prompt = _build_repair_prompt(rule_md_text, broken_code, error_message, code_type)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1200,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        repaired_code = _extract_repaired_code(output_text, code_type)
        if not repaired_code or len(repaired_code) < 50:
            print(f"    ❌ [Repair] 修复输出过短，放弃写入: {os.path.basename(file_path)}")
            return False

        _write_file_safe(file_path, repaired_code)
        print(f"    ✅ [Repair] 修复完成: {os.path.basename(file_path)}")
        return True

    except Exception as e:
        print(f"    💥 [Repair] 修复异常: {os.path.basename(file_path)} | {e}")
        if "CUDA" in str(e):
            raise e
        return False


# ============================================================
# 主入口
# ============================================================

def run_repair(
    repair_ids: list,
    iter_dir: str,
    rule_md_dir: str,
    error_context: dict,
    model: torch.nn.Module,
    tokenizer: object,
) -> dict:
    """
    对 repair_ids 中的每条规则，针对失败文件进行定向修复。
    返回 {rule_id: "success"/"failed"/"skipped"} 摘要。
    """
    scala_t_dir = os.path.join(iter_dir, "scala_T")
    scala_f_dir = os.path.join(iter_dir, "scala_F")
    results = {}

    print(f"\n🔧 [RepairAgent] 开始定向修复，共 {len(repair_ids)} 条规则...")

    for rule_id in repair_ids:
        info = error_context.get(rule_id, {})
        error_message = str(info.get("message", "编译失败，无详细信息"))

        # 读取规则解析 md
        md_path = os.path.join(rule_md_dir, f"rule_{rule_id}.md")
        rule_md_text = _read_file_safe(md_path)
        if not rule_md_text:
            print(f"  ⚠️ [Repair] Rule {rule_id} 缺少解析 md，跳过")
            results[rule_id] = "skipped"
            continue

        print(f"\n  🛠️  [Repair] 正在修复 Rule {rule_id}...")

        success_t = True
        success_f = True

        # 修复正例（_T.scala）
        t_path = os.path.join(scala_t_dir, f"rule_{rule_id}_T.scala")
        if os.path.exists(t_path):
            success_t = _repair_single_file(t_path, rule_md_text, error_message, "T", model, tokenizer)

        # 修复反例（_F.scala）
        f_path = os.path.join(scala_f_dir, f"rule_{rule_id}_F.scala")
        if os.path.exists(f_path):
            success_f = _repair_single_file(f_path, rule_md_text, error_message, "F", model, tokenizer)

        if success_t and success_f:
            results[rule_id] = "success"
        elif success_t or success_f:
            results[rule_id] = "partial"
        else:
            results[rule_id] = "failed"

    success_count = sum(1 for v in results.values() if v in ("success", "partial"))
    print(f"\n🏁 [RepairAgent] 修复完成: {success_count}/{len(repair_ids)} 条成功或部分成功")
    return results
