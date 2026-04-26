# Multi-agent-chisel-test-set-generation
zms的第一个发布项目：基于多智能体的chisel测试集生成，支持传入文件格式的规则描述，支持对规则解析文件的可视化提供。 并提供compare支持不同模型选型和agent协同设计进行数据比较。提供collect 对同一工作流的多轮循环做统计。
# Chisel 测试用例生成多智能体系统

> 基于 LangGraph 的自主迭代式 Chisel 硬件描述语言测试用例生成框架

## 📋 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [模块说明](#模块说明)
- [核心流程详解](#核心流程详解)
- [关键技术实现](#关键技术实现)
- [操作手册](#操作手册)
- [常见问题排查](#常见问题排查)

---

## 项目概述

本项目是一个基于大语言模型（LLM）的多智能体协作系统，用于自动生成 Chisel 3.x 硬件描述语言的测试用例。系统采用 **LangGraph** 状态图框架，实现了多个专业 Agent 的协同工作，通过迭代优化的方式生成高质量的正例（符合规则）和反例（违反规则但语法正确）测试代码。

### 核心特性

- **双轮验证机制**：初次生成 + 专家增强重生成，确保代码质量
- **专家评审系统**：语法专家 + 规则合规专家双重把关
- **自主迭代优化**：Harness 评估 Agent 提供跨轮次改进建议
- **技能路由系统**：根据失败模式动态选择生成策略（base/cot/deep）
- **增量修复能力**：RepairAgent 针对性修复失败用例
- **RAG 增强检索**：混合 BM25 + FAISS 向量检索提供上下文
- **完整可追溯性**：每轮生成 Harness 报告 + Progress 进度文件

---

## 系统架构

### 目录结构

```
Test-Agent/
├── LangGraph/                    # 核心流程控制
│   ├── config.py                 # 全局配置
│   ├── state.py                  # 状态定义
│   ├── nodes.py                  # 节点实现
│   ├── build_graph.py            # 流程图构建
│   ├── run_pipeline.py           # 主入口
│   └── RAG_Service_v2.py         # RAG 检索服务
│
├── LangGraph_agents/             # Agent 实现
│   ├── skill_router.py           # 技能路由
│   ├── planner_agent.py          # 规划 Agent
│   ├── rule_analysis.py          # 基础规则解析
│   ├── deep_analysis.py          # 深度规则解析
│   ├── code_generate.py          # 基础代码生成
│   ├── deep_generate.py          # 深度代码生成
│   ├── retry_rule_parser.py      # CoT 规则解析
│   ├── retry_code_generate.py    # CoT 代码生成
│   ├── repair_agent.py           # 修复 Agent
│   ├── verifier.py               # 编译验证器
│   ├── syntax_reviewer.py        # 语法专家
│   ├── rule_compliance_reviewer.py  # 合规专家
│   └── harness_agent.py          # 评估 Agent
│
├── output*/                      # 输出目录（按版本）
│   └── iteration_N/              # 每轮迭代目录
│       ├── rule_md_analysis/     # 规则解析结果
│       ├── rule_md_enhanced/     # 增强解析（含专家建议）
│       ├── scala_T/              # 正例代码
│       ├── scala_F/              # 反例代码
│       ├── verified_pass/        # 第一轮通过文件
│       ├── verified_fail/        # 第一轮失败文件
│       ├── verifiedv2_T/         # 第二轮正例
│       ├── verifiedv2_F/         # 第二轮反例
│       ├── verifiedv2_pass/      # 第二轮通过文件
│       ├── verifiedv2_fail/      # 第二轮失败文件
│       ├── verify_report.log     # 第一轮验证日志
│       └── verifiedv2_report.log # 第二轮验证日志
│
├── harness_reports/              # Harness 评估报告
│   └── iteration_N_report.md
│
├── progress/                     # 进度追踪
│   └── iteration_N_progress.md
│
├── rule10.txt / rule30.txt       # 规则描述文件
└── structure.md                  # 架构演进文档
```

---

## 模块说明

### 1. LangGraph 核心模块

#### `config.py` - 全局配置
```python
BASE_ROOT = "/root/Test-Agent/output30_deep_v5"  # 输出根目录
RULES_FILE = "/root/Test-Agent/rule30.txt"       # 规则文件路径
MAX_ITERATIONS = 3                                # 最大迭代轮次
FORCE_SKILL = ""                                  # 强制技能（调试用）
BASE_MODEL = "Qwen2.5-Coder-32B-Instruct"        # 基础模型
USE_LORA = False                                  # 是否使用 LoRA
API_URL = "http://localhost:8080/api/..."        # Scala 编译器 API
```

#### `state.py` - 状态管理
定义 `PipelineState` TypedDict，包含：
- **循环控制**：`iteration`, `max_iterations`, `fail_count`
- **路径信息**：`retry_rule_md_dir`, `retry_scala_T_dir`, `retry_scala_F_dir`
- **失败驱动**：`failed_rules`, `error_context`, `attempt_count`
- **Agent 决策**：`plan`, `skill_used`, `repair_ids`
- **专家评审**：`syntax_advice`, `compliance_advice`
- **验证统计**：`v1_stats`, `v2_stats`（两轮真实编译结果）
- **跨轮反馈**：`last_harness_advice`

#### `nodes.py` - 节点实现
包含 10 个核心节点：
1. **verifier_node**：第一轮编译验证
2. **syntax_review_node**：语法专家预检
3. **compliance_review_node**：规则合规审查
4. **reviewed_codegen_node**：结合专家建议重生成
5. **verifier_v2_node**：第二轮编译验证
6. **harness_node**：生成评估报告
7. **planner_node**：感知失败根因，输出技能选择
8. **repair_node**：定向修复失败代码
9. **rule_parser_node**：动态调度规则解析
10. **codegen_node**：动态调度代码生成
11. **update_iteration_node**：升级迭代轮次

#### `build_graph.py` - 流程图构建
定义 LangGraph 状态图：
```python
verifier → syntax_review → compliance_review 
        → reviewed_codegen → verifier_v2 → harness → [路由]
        
[路由决策]
- 无代码 → planner（启动生成）
- 全通过 → END
- 达到上限 → END
- 有失败 → update_iter → planner（下一轮）

[生成流程]
planner → rule_parser → codegen → verifier（回到顶部）
```

---

### 2. LangGraph_agents 智能体模块

#### 技能路由系统

**`skill_router.py`**
```python
SKILL_MAP = {
    "base":  (run_rule_analysis,       run_code_generate),
    "cot":   (run_retry_rule_analysis, run_retry_code_generate),
    "deep":  (run_deep_rule_analysis,  run_deep_code_generate),
}
```
- **base**：基础解析 + 生成（首次或规则清晰）
- **cot**：思维链强化（syntax 反复失败）
- **deep**：架构师视角深度解析（MLIR/topModule 失败）

#### 规则解析 Agent

**`rule_analysis.py`** - 基础解析
- 输入：规则文本
- 输出：结构化解析（规则核心、正例关键、反例关键、实现要点）
- 特点：快速、直接

**`deep_analysis.py`** - 深度解析
- 输入：规则文本
- 输出：《高覆盖率测试用例开发指南》
- 包含：工程场景画像、模范遵循指南、违反用例指南、开发者便签
- 特点：架构师视角，强调边界条件和隐蔽违规

**`retry_rule_parser.py`** - CoT 解析
- 输入：规则文本 + 历史失败信息
- 输出：思维链推理 + 结构化解析
- 特点：显式推理过程，适合复杂规则

#### 代码生成 Agent

**`code_generate.py`** - 基础生成
- 输入：规则解析 md
- 输出：正例 + 反例 Scala 代码
- 特点：增量生成（跳过已存在文件）

**`deep_generate.py`** - 深度生成
- 输入：深度解析指南
- 输出：工程化正反例代码
- 参数：max_new_tokens=3000, temperature=0.5
- 特点：结构复杂，多样性高

**`retry_code_generate.py`** - CoT 生成
- 输入：规则解析 + 失败映射
- 输出：针对性修复的代码
- 特点：全量/增量兼容


#### 专家评审 Agent

**`syntax_reviewer.py`** - 语法专家
- 输入：失败的 Scala 代码文件
- 输出：{rule_id: "语法建议"}
- 功能：
  - 静态黑名单检测（XSDebug, Float(n.W), implicit 等）
  - LLM 深度语法检查
  - 分 T/F 两侧独立评审
- 特点：快速预检，避免明显错误

**`rule_compliance_reviewer.py`** - 合规专家
- 输入：失败的代码 + 规则解析 md
- 输出：{rule_id: "合规建议"}
- 功能：
  - 正例：是否真正遵守规则
  - 反例：是否真正违反规则 + 是否有 `// 违反规则：...` 注释
- 特点：确保测试用例符合规则意图

#### 其他关键 Agent

**`planner_agent.py`** - 规划 Agent
- 输入：error_context（失败根因）
- 输出：plan = {skill, use_rag, repair_ids, regenerate_ids, reasoning}
- 决策逻辑：
  - 启发式优先（明显模式直接决策）
  - LLM 兜底（复杂情况调用模型）
  - 感知 attempt_count（多次失败升级策略）

**`repair_agent.py`** - 修复 Agent
- 输入：repair_ids + error_context + 原代码
- 输出：修复后的代码
- 特点：定向修复，保留原结构

**`verifier.py`** - 编译验证器
- 输入：Scala 文件目录
- 输出：(failed_ids, error_context)
- 功能：
  - 调用 Scala 编译器 API
  - 分类错误类型（syntax/mlir/empty/timeout）
  - 写入 verified_pass/ + verified_fail/
  - 生成 verify_report.log

**`harness_agent.py`** - 评估 Agent
- 输入：verify_report.log + enhanced md
- 输出：Harness 报告 + LLM 改进建议
- 功能：
  - 解析日志统计 T/F 通过率
  - 聚类失败模式
  - LLM 生成三维度建议（提示词/框架/优先修复项）
  - 写入 harness_reports/iteration_N_report.md

---

## 核心流程详解

### 单轮 Iteration 完整流程

```
┌─────────────────────────────────────────────────────────────┐
│  Iteration N 开始                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  1. Planner 决策                                             │
│     - 分析 error_context（上一轮失败信息）                   │
│     - 选择 skill（base/cot/deep）                            │
│     - 决定是否使用 RAG                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  2. RuleParser 规则解析                                      │
│     - 读取 rule30.txt                                        │
│     - 调用对应 skill 的 parse_fn                             │
│     - 输出到 iteration_N/rule_md_analysis/                   │
│     - 注入 last_harness_advice（上一轮建议）                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  3. CodeGen 代码生成（第一次）                               │
│     - 读取 rule_md_analysis/*.md                             │
│     - 调用对应 skill 的 codegen_fn                           │
│     - 输出到 iteration_N/scala_T/ + scala_F/                 │
│     - 增量生成（跳过已存在文件）                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Verifier-V1 第一轮编译验证                               │
│     - 调用 Scala 编译器 API 验证所有文件                     │
│     - 通过 → verified_pass/                                  │
│     - 失败 → verified_fail/                                  │
│     - 生成 verify_report.log                                 │
│     - 存储 v1_stats 到 state                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  5. SyntaxReviewer 语法专家预检                              │
│     - 只读取 verified_fail/ 中的失败文件                     │
│     - 静态黑名单检测 + LLM 深度检查                          │
│     - 输出 syntax_advice = {rule_id: "[T/F] 建议"}           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  6. ComplianceReviewer 规则合规审查                          │
│     - 只读取 verified_fail/ 中的失败文件                     │
│     - 检查正例是否遵守规则、反例是否违反规则                  │
│     - 输出 compliance_advice = {rule_id: "[T/F] 建议"}       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  7. ReviewedCodeGen 结合专家建议重生成（第二次）             │
│     - 从 verified_fail/ 推断需要重生成的 T/F 文件            │
│     - 将双专家建议追加到 rule_md_enhanced/*.md               │
│     - 从 verified_pass/ 复制已通过文件到 verifiedv2_T/F/    │
│     - 只重新生成失败的那一侧（T 或 F）                        │
│     - 输出到 verifiedv2_T/ + verifiedv2_F/                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  8. Verifier-V2 第二轮编译验证                               │
│     - 验证 verifiedv2_T/ + verifiedv2_F/ 中的所有文件        │
│     - 通过 → verifiedv2_pass/                                │
│     - 失败 → verifiedv2_fail/                                │
│     - 生成 verifiedv2_report.log                             │
│     - 存储 v2_stats 到 state                                 │
│     - 更新 fail_count = v2_fail（仍失败的文件数）            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  9. Harness 评估报告生成                                     │
│     - 解析 verify_report.log + verifiedv2_report.log         │
│     - 统计 T/F 通过率（两轮）                                │
│     - 聚类失败模式                                            │
│     - LLM 生成改进建议                                        │
│     - 输出 harness_reports/iteration_N_report.md             │
│     - 输出 progress/iteration_N_progress.md                  │
│     - 存储 last_harness_advice 到 state                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  10. 路由决策                                                │
│      - fail_count == 0 → END（全部通过）                     │
│      - iteration >= MAX_ITERATIONS → END（达到上限）         │
│      - 否则 → update_iter → 下一轮 Iteration                 │
└─────────────────────────────────────────────────────────────┘
```


### 关键设计亮点

#### 1. 双轮验证机制
每轮 iteration 包含两次独立的代码生成和编译验证：
- **第一轮**：全量生成 → 真实编译 → 识别失败文件
- **第二轮**：专家增强 → 只重生成失败文件 → 再次编译

这种设计确保：
- 第一轮快速覆盖所有规则
- 第二轮针对性修复，不浪费资源重新生成已通过的文件
- 两轮统计独立记录，便于分析专家系统的有效性

#### 2. T/F 独立追踪
正例（T）和反例（F）完全独立处理：
- 专家评审时分别标注 `[正例]` / `[反例]`
- 重生成时只删除有问题的那一侧文件
- 统计时分别计算 T 通过率和 F 通过率

避免了"T 通过就认为 F 也通过"的错误假设。

#### 3. 增量生成策略
所有代码生成函数都支持增量模式：
```python
if os.path.exists(pos_path) and os.path.exists(neg_path):
    print(f"⏭️ 跳过已存在用例: {base_name}")
    continue
```
- 同一 iteration 内重启不会重复生成
- 第二轮重生成时，已通过文件从 verified_pass/ 复制到 verifiedv2_T/F/，codegen 自动跳过

#### 4. 跨轮次学习机制
通过 `last_harness_advice` 实现：
```python
# harness_node 存储建议
state["last_harness_advice"] = llm_advice

# rule_parser_node 和 codegen_node 注入建议
parse_fn(..., harness_advice=state.get("last_harness_advice", ""))
```
每轮的 Harness 报告会总结失败模式和改进建议，下一轮的 Agent 会读取这些建议并调整生成策略。

---

## 关键技术实现

### 1. 状态图路由机制

#### 条件路由示例
```python
def after_harness(state: PipelineState) -> str:
    it = state.get("iteration", 0)
    fail_count = state.get("fail_count", 0)
    max_it = getattr(config, "MAX_ITERATIONS", 5)
    
    # 无代码 → 启动生成
    if not has_files:
        return "plan"
    
    # 全通过 → 结束
    if fail_count == 0:
        return "end"
    
    # 达到上限 → 结束
    if it >= max_it:
        return "end"
    
    # 继续下一轮
    return "next_it"
```

#### 技能路由示例
```python
def after_planner(state: PipelineState) -> str:
    plan = state.get("plan", {})
    skill = plan.get("skill", "base")
    repair_ids = plan.get("repair_ids", [])
    
    if skill == "repair" and repair_ids:
        return "repair_only"
    elif repair_ids:
        return "repair_then_parse"
    else:
        return "parse_only"
```

### 2. 错误上下文提取

从编译器 API 返回的 JSON 中提取关键信息：
```python
def _classify_error(detail) -> str:
    if isinstance(detail, dict):
        syntax = detail.get("syntax")
        mlir = str(detail.get("mlir", "")).lower()
        
        if syntax is not True:
            return "syntax"
        if mlir not in ("true", "none"):
            return "mlir"
        if not detail.get("topModules"):
            return "mlir"
    
    return "unknown"

# 存储到 error_context
error_context[rule_id] = {
    "error_type": error_type,
    "message": detail.get("errorMsg", ""),
    "file_path": file_path,
    "code_type": "T" or "F",
    "attempt": attempt_count + 1,
}
```

### 3. 专家建议解析

从专家 advice 字符串中判断 T/F 侧是否有问题：
```python
def _adv_has_t_issue(adv: str) -> bool:
    if not adv:
        return False
    # 明确标记为无问题
    if any(m in adv for m in ("无明显", "均符合", "跳过")):
        return False
    # 明确标记为正例问题
    if "[正例]" in adv:
        return True
    # 无标签前缀 → 视为两侧都有问题
    if "[反例]" not in adv:
        return True
    return False
```

### 4. 增强 Markdown 生成

将专家建议追加到规则解析文档：
```python
enhanced = original + f"""

---
## 专家评审建议（请在重新生成时严格遵守）

### 语法专家建议
{syntax_advice.get(rid, "无")}

### 规则合规专家建议
{compliance_advice.get(rid, "无")}
"""
```

### 5. 进度报告生成

合并两轮验证统计：
```python
def _save_progress_both(it, v1_stats, v2_stats, base_root):
    # 第一阶段：v1 真实编译结果
    v1_t_pass = v1_stats["t_pass"]
    v1_f_pass = v1_stats["f_pass"]
    
    # 第二阶段：v2 真实编译结果
    v2_t_pass = v2_stats["t_pass"]
    v2_f_pass = v2_stats["f_pass"]
    
    # 综合总结：v1 通过 + v2 通过（不重复计数）
    final_t_pass = v1_t_pass + v2_t_pass
    final_f_pass = v1_f_pass + v2_f_pass
    final_rate = (final_t_pass + final_f_pass) / (total_rules * 2) * 100
```

### 6. RAG 检索增强

混合 BM25 + FAISS 检索：
```python
class ChiselHybridRAGService:
    def retrieve(self, query: str, top_k: int = 5):
        # BM25 关键词检索
        bm25_results = self.bm25.get_top_n(query, self.corpus, n=top_k)
        
        # FAISS 向量检索
        query_vec = self.encoder.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        
        # 混合排序
        combined = self._merge_results(bm25_results, faiss_results)
        return combined[:top_k]
```


---

## 操作手册

### 环境准备

#### 1. 依赖安装
```bash
pip install langgraph transformers torch peft requests
pip install faiss-cpu rank-bm25  # RAG 依赖
```

#### 2. 模型准备
```bash
# 下载 Qwen2.5-Coder-32B-Instruct
# 或使用 ModelScope
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-Coder-32B-Instruct', 
                  cache_dir='/root/.cache/modelscope/hub/models')
```

#### 3. Scala 编译器 API
确保 Scala 编译服务运行在 `http://localhost:8080`：
```bash
# 启动 Spring Boot Chisel 编译服务
cd /root/springboot_chisel
./gradlew bootRun
```

### 快速开始

#### 1. 配置修改
编辑 `LangGraph/config.py`：
```python
BASE_ROOT = "/root/Test-Agent/output_my_experiment"  # 输出目录
RULES_FILE = "/root/Test-Agent/rule30.txt"           # 规则文件
MAX_ITERATIONS = 3                                    # 迭代轮次（0, 1, 2）
FORCE_SKILL = ""                                      # 留空=自动决策
USE_LORA = False                                      # 是否使用 LoRA
```

#### 2. 运行流水线
```bash
cd /root/Test-Agent/LangGraph
CUDA_VISIBLE_DEVICES=0,1 python run_pipeline.py
```

#### 3. 查看结果
```bash
# 查看 Harness 报告
cat ../harness_reports/iteration_0_report.md

# 查看进度统计
cat ../progress/iteration_0_progress.md

# 查看生成的代码
ls ../output_my_experiment/iteration_0/scala_T/
ls ../output_my_experiment/iteration_0/scala_F/

# 查看验证日志
cat ../output_my_experiment/iteration_0/verify_report.log
```

### 高级配置

#### 1. 强制使用特定技能
```python
# config.py
FORCE_SKILL = "deep"  # 强制使用 deep 技能，跳过 Planner 决策
```

#### 2. 启用 LoRA 微调模型
```python
# config.py
USE_LORA = True
LORA_PATH = "/root/Qwen-2.5-coder/finetune/lora_ckpt1"
```

#### 3. 启用 RAG 检索
```python
# config.py
RAG_START_ITERATION = 1  # 从第 1 轮开始启用 RAG
RAG_END_ITERATION = 3    # 到第 3 轮结束
```

#### 4. 调整迭代上限
```python
# config.py
MAX_ITERATIONS = 5  # 最多跑 iteration 0~4（共 5 轮）
```

注意：`MAX_ITERATIONS = N` 会跑 iteration 0 到 N-1，共 N 轮。

#### 5. 修改生成参数
编辑对应的生成文件（如 `deep_generate.py`）：
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=3000,    # 增加输出长度
    temperature=0.5,        # 调整温度（0.1~1.0）
    top_p=0.9,              # 调整采样概率
    do_sample=True,
    repetition_penalty=1.05 # 重复惩罚
)
```

### 输出文件说明

#### 每轮 Iteration 目录结构
```
output_xxx/iteration_N/
├── rule_md_analysis/          # 第一次规则解析
│   └── rule_*.md
├── rule_md_enhanced/          # 增强解析（含专家建议）
│   └── rule_*.md
├── scala_T/                   # 第一次生成的正例
│   └── rule_*_T.scala
├── scala_F/                   # 第一次生成的反例
│   └── rule_*_F.scala
├── verified_pass/             # 第一轮验证通过
│   └── rule_*_T.scala / rule_*_F.scala
├── verified_fail/             # 第一轮验证失败
│   └── rule_*_T.scala / rule_*_F.scala
├── verifiedv2_T/              # 第二轮正例（含复制的通过文件）
│   └── rule_*_T.scala
├── verifiedv2_F/              # 第二轮反例（含复制的通过文件）
│   └── rule_*_F.scala
├── verifiedv2_pass/           # 第二轮验证通过
│   └── rule_*_T.scala / rule_*_F.scala
├── verifiedv2_fail/           # 第二轮验证仍失败
│   └── rule_*_T.scala / rule_*_F.scala
├── verify_report.log          # 第一轮验证日志
└── verifiedv2_report.log      # 第二轮验证日志
```

#### Harness 报告
```
harness_reports/iteration_N_report.md
- 本轮指标（T/F 通过率）
- 失败模式聚类
- 失败详情（含 errorMsg）
- 双专家评审建议摘要
- 原始验证日志
- LLM 生成的迭代建议
```

#### Progress 进度文件
```
progress/iteration_N_progress.md
- 第一阶段：初次生成编译结果（v1_stats）
- 第二阶段：专家增强后编译结果（v2_stats）
- 综合总结：两轮合并最终通过率
- 失败规则列表（T 侧 / F 侧）
```

---

## 常见问题排查

### 1. 模型加载失败

**问题**：`RuntimeError: CUDA out of memory`

**解决**：
```python
# 方案 1：减少 batch size（模型内部已设置 batch_size=1）
# 方案 2：使用模型量化
base = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL,
    torch_dtype=torch.float16,  # 使用 FP16
    device_map="auto",           # 自动分配多卡
    load_in_8bit=True,           # 8-bit 量化
)

# 方案 3：使用更小的模型
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
```

### 2. Scala 编译器 API 超时

**问题**：`TIMEOUT | Scala compiler hung`

**解决**：
```python
# config.py
TIMEOUT = 600  # 增加超时时间（秒）

# 或在 verifier.py 中调整
resp = requests.get(api_url, params=params, timeout=(5, 600))
```

### 3. 生成的代码为空或格式错误

**问题**：`⚠️ 生成失败，规则 MD 文件格式异常或内容缺失`

**排查步骤**：
1. 检查规则解析是否成功：
   ```bash
   cat output_xxx/iteration_0/rule_md_analysis/rule_1.md
   ```
2. 检查模型输出是否被截断：
   ```python
   # 增加 max_new_tokens
   max_new_tokens=3000  # 默认 1200
   ```
3. 检查提示词是否正确：
   ```bash
   # 查看 deep_generate.py 中的 build_deep_generate_prompt
   ```

### 4. 专家评审没有生效

**问题**：第二轮重生成后通过率没有提升

**排查步骤**：
1. 检查专家建议是否生成：
   ```bash
   cat output_xxx/iteration_0/rule_md_enhanced/rule_1.md
   # 应该包含 "## 专家评审建议" 部分
   ```
2. 检查是否正确识别 T/F 侧问题：
   ```python
   # 在 nodes.py 中打印调试信息
   print(f"[DEBUG] syntax_advice: {syntax_advice}")
   print(f"[DEBUG] needs_regen_T: {needs_regen_T}")
   print(f"[DEBUG] needs_regen_F: {needs_regen_F}")
   ```
3. 检查 verifiedv2_T/F/ 是否包含正确的文件：
   ```bash
   ls output_xxx/iteration_0/verifiedv2_T/
   ls output_xxx/iteration_0/verifiedv2_F/
   ```


### 5. 迭代次数不符合预期

**问题**：设置 `MAX_ITERATIONS = 3` 但只跑了 iteration 0

**原因**：
- `fail_count == 0`：所有文件都通过了，提前结束
- `it >= max_it`：判断逻辑是 `>=`，所以 `MAX_ITERATIONS = 3` 只跑 0, 1, 2

**解决**：
```python
# 方案 1：增加迭代次数
MAX_ITERATIONS = 4  # 跑 iteration 0~3

# 方案 2：修改判断逻辑（build_graph.py）
if it > max_it:  # 改为 >
    return "end"

# 方案 3：设置 TARGET_PASS_RATE 为不可达
TARGET_PASS_RATE = 1.1  # 确保不会因为通过率达标而提前退出
```

### 6. 路径错误或文件找不到

**问题**：`FileNotFoundError: [Errno 2] No such file or directory`

**排查步骤**：
1. 检查 `config.py` 中的路径是否正确：
   ```python
   BASE_ROOT = "/root/Test-Agent/output_xxx"  # 确保路径存在
   RULES_FILE = "/root/Test-Agent/rule30.txt" # 确保文件存在
   ```
2. 检查目录权限：
   ```bash
   ls -la /root/Test-Agent/
   chmod -R 755 /root/Test-Agent/
   ```
3. 检查是否有残留的旧目录：
   ```bash
   rm -rf /root/Test-Agent/output_xxx/iteration_0/
   ```

### 7. Harness 报告为空或不完整

**问题**：`harness_reports/iteration_0_report.md` 内容很少

**原因**：
- `verify_report.log` 不存在或为空
- LLM 生成建议时超时或失败

**解决**：
```bash
# 检查日志是否存在
cat output_xxx/iteration_0/verify_report.log

# 检查 Harness Agent 是否正常运行
# 在 harness_agent.py 中添加调试信息
print(f"[DEBUG] log_path: {log_path}")
print(f"[DEBUG] stats: {stats}")
```

### 8. 反例生成质量低

**问题**：反例（F）通过率很低，大多是语法错误而非规则违反

**优化策略**：
1. **增强提示词**：在 `deep_generate.py` 中强调"语法正确但违反规则"
2. **使用 CoT 技能**：
   ```python
   FORCE_SKILL = "cot"  # 使用思维链推理
   ```
3. **启用 RAG**：提供更多正确的 Chisel 代码示例
4. **增加专家评审权重**：
   ```python
   # 在 compliance_reviewer.py 中增加反例检查逻辑
   if neg_code and "// 违反规则" not in neg_code:
       quick_issues.append("[反例] 缺少违规注释")
   ```

### 9. 内存泄漏或显存不释放

**问题**：运行多轮后显存占用越来越高

**解决**：
```python
# 在 nodes.py 的 load_resources_once 后添加
import gc
import torch

# 每轮结束后清理
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# 在 update_iteration_node 中调用
cleanup()
```

### 10. 日志输出过多

**问题**：终端输出太多，难以查看关键信息

**解决**：
```python
# 方案 1：重定向到文件
python run_pipeline.py > pipeline.log 2>&1

# 方案 2：调整日志级别
import logging
logging.basicConfig(level=logging.WARNING)

# 方案 3：只保留关键节点的输出
# 在 nodes.py 中注释掉不必要的 print 语句
```

---

## 性能优化建议

### 1. 并行化处理
当前实现是串行的，可以考虑：
- 规则解析并行化（多进程）
- 代码生成并行化（多 GPU）
- 验证并行化（批量提交到编译器 API）

### 2. 缓存机制
- 规则解析结果缓存（相同规则不重复解析）
- 模型输出缓存（相同输入不重复推理）
- 编译结果缓存（相同代码不重复验证）

### 3. 增量优化
- 只对失败的规则重新解析和生成（当前已部分实现）
- 跨 iteration 复用通过的文件（需要修改 update_iteration_node）

---

## 扩展开发指南

### 添加新的技能

1. 创建新的解析和生成文件：
   ```bash
   touch LangGraph_agents/custom_analysis.py
   touch LangGraph_agents/custom_generate.py
   ```

2. 实现解析和生成函数：
   ```python
   # custom_analysis.py
   def run_custom_rule_analysis(pending_rules, output_dir, ...):
       # 自定义解析逻辑
       pass
   
   # custom_generate.py
   def run_custom_code_generate(rule_md_dir, scala_t_dir, ...):
       # 自定义生成逻辑
       pass
   ```

3. 注册到技能路由：
   ```python
   # skill_router.py
   SKILL_MAP = {
       "base": (run_rule_analysis, run_code_generate),
       "cot": (run_retry_rule_analysis, run_retry_code_generate),
       "deep": (run_deep_rule_analysis, run_deep_code_generate),
       "custom": (run_custom_rule_analysis, run_custom_code_generate),  # 新增
   }
   ```

### 添加新的专家 Agent

1. 创建新的专家文件：
   ```bash
   touch LangGraph_agents/performance_reviewer.py
   ```

2. 实现评审函数：
   ```python
   def run_performance_review(rule_ids, scala_t_dir, scala_f_dir, ...):
       advice = {}
       for rid in rule_ids:
           # 性能分析逻辑
           advice[rid] = "性能建议..."
       return advice
   ```

3. 在 `nodes.py` 中添加节点：
   ```python
   def performance_review_node(state: PipelineState) -> PipelineState:
       # 调用 run_performance_review
       state["performance_advice"] = run_performance_review(...)
       return state
   ```

4. 在 `build_graph.py` 中注册和连接：
   ```python
   workflow.add_node("performance_review", performance_review_node)
   workflow.add_edge("compliance_review", "performance_review")
   workflow.add_edge("performance_review", "reviewed_codegen")
   ```

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发规范
- 代码风格：遵循 PEP 8
- 注释语言：中文
- 提交信息：中文，格式 `[模块] 简短描述`
- 测试：添加单元测试（`tests/` 目录）

### 提交流程
1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/xxx`
3. 提交更改：`git commit -m "[nodes] 添加新的专家节点"`
4. 推送分支：`git push origin feature/xxx`
5. 创建 Pull Request

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 致谢

- **LangGraph**：状态图框架
- **Qwen2.5-Coder**：代码生成基座模型
- **Chisel 3.x**：硬件描述语言
- **Spring Boot Chisel Compiler**：Scala 编译服务

---

## 联系方式

- 项目维护者：[zms]
- 邮箱：[2029747446@qq.com]      看到问题会回复

---

**最后更新**：2026-04-26


              
