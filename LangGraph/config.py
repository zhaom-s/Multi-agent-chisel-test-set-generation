# /root/Test-Agent/LangGraph/config.py
import os

# ===============================
# 📂 路径核心配置
# ===============================
BASE_ROOT = "/root/Test-Agent/output30_deep_v5_LORA"

RULES_FILE = "/root/Test-Agent/rule30.txt"
RULES_PATH = "/root/Test-Agent/rule30.txt"

LOG_FILE = "/root/Test-Agent/LangGraph/pipeline.log"

# ===============================
# 🤖 模型与推理配置
# ===============================
BASE_MODEL = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Coder-32B-Instruct"
LORA_PATH = "/root/Qwen-2.5-coder/finetune/lora_ckpt1"

USE_LORA = True

RAG_START_ITERATION = 100
RAG_END_ITERATION = 100

API_URL = "http://localhost:8080/api/internal/parse-stream"
TIMEOUT = 300

# ===============================
# 🔁 迭代控制参数
# ===============================
MAX_ITERATIONS = 3
TARGET_PASS_RATE = 1.1  # 设为不可达，确保不会提前退出
MIN_IMPROVEMENT = 3

# ===============================
# 🎯 强制 Skill 选择（调试用）
# 设为非空字符串时，Planner 直接使用该 skill，跳过 LLM 决策
# 可选值: "base" | "cot" | "deep" | "" (空=由 Planner 自动决策)
# ===============================
FORCE_SKILL = ""

# ===============================
# 🛠️ 初始化环境
# ===============================
if not os.path.exists(BASE_ROOT):
    os.makedirs(BASE_ROOT, exist_ok=True)

print(f"⚙️  [Config] 迭代上限: {MAX_ITERATIONS}")
print(f"⚙️  [Config] 强制 Skill: {FORCE_SKILL or '自动决策'}")
print(f"⚙️  [Config] 模型路径: {os.path.basename(BASE_MODEL)}")
