# improvement = (v2_T + v2_F) - (v1_T + v1_F)
# 


import os
import re
import matplotlib.pyplot as plt


# =========================
# 1. 解析单个目录
# =========================
def parse_folder(folder, max_rule=30):
    stats = {i: {"T": 0, "F": 0} for i in range(1, max_rule + 1)}

    if not os.path.exists(folder):
        return stats

    for fname in os.listdir(folder):
        match = re.match(r"rule_(\d+)_(T|F)\.scala", fname)
        if not match:
            continue

        rid = int(match.group(1))
        suffix = match.group(2)

        if rid <= max_rule:
            stats[rid][suffix] = 1

    return stats


# =========================
# 2. 计算总分
# =========================
def compute_score(stats):
    return sum(stats[r]["T"] + stats[r]["F"] for r in stats)


# =========================
# 3. 画“提升折线图”（核心）
# =========================
def plot_improvement(iter_id, stats_v1, stats_v2, output_dir):
    ids = list(stats_v1.keys())

    improvements = []
    for i in ids:
        v1 = stats_v1[i]["T"] + stats_v1[i]["F"]
        v2 = stats_v2[i]["T"] + stats_v2[i]["F"]
        improvements.append(v2 - v1)

    plt.figure(figsize=(16, 6))

    plt.plot(ids, improvements, marker="o")

    # 0 基准线（非常关键）
    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Rule ID")
    plt.ylabel("Improvement (v2 - v1)")
    plt.title(f"Iteration {iter_id}: Improvement per Rule")

    plt.xticks(ids)
    plt.grid(alpha=0.3)

    save_path = os.path.join(output_dir, f"iter_{iter_id}_improvement.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


# =========================
# 4. 画整体趋势图
# =========================
def plot_overall_trend(scores_v1, scores_v2, output_dir):
    iters = list(range(len(scores_v1)))

    plt.figure(figsize=(10, 5))

    plt.plot(iters, scores_v1, marker="o", label="v1")
    plt.plot(iters, scores_v2, marker="o", label="v2")

    plt.xlabel("Iteration")
    plt.ylabel("Total Score")
    plt.title("Overall Performance Trend")

    plt.legend()
    plt.grid(alpha=0.3)

    save_path = os.path.join(output_dir, "overall_trend.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


# =========================
# 5. 主流程
# =========================
def analyze(base_dir, output_dir, max_rule=30, num_iters=4):
    os.makedirs(output_dir, exist_ok=True)

    md_lines = []
    md_lines.append("# 📊 v2 相比 v1 提升分析\n")

    scores_v1 = []
    scores_v2 = []

    for i in range(num_iters):
        v1_path = f"{base_dir}/iteration_{i}/verified_pass"
        v2_path = f"{base_dir}/iteration_{i}/verifiedv2_pass"

        stats_v1 = parse_folder(v1_path, max_rule)
        stats_v2 = parse_folder(v2_path, max_rule)

        score_v1 = compute_score(stats_v1)
        score_v2 = compute_score(stats_v2)

        scores_v1.append(score_v1)
        scores_v2.append(score_v2)

        improvement = score_v2 - score_v1
        ratio = (improvement / score_v1 * 100) if score_v1 > 0 else 0

        # 画“提升图”
        img_path = plot_improvement(i, stats_v1, stats_v2, output_dir)

        # Markdown
        md_lines.append(f"## Iteration {i}\n")
        md_lines.append(f"- v1 成功数: **{score_v1}** / {max_rule*2}")
        md_lines.append(f"- v2 成功数: **{score_v2}** / {max_rule*2}")
        md_lines.append(f"- 提升: **{improvement}**")
        md_lines.append(f"- 提升比例: **{ratio:.2f}%**\n")
        md_lines.append(f"![iter{i}]({os.path.basename(img_path)})\n")

    # 画整体趋势
    trend_img = plot_overall_trend(scores_v1, scores_v2, output_dir)

    md_lines.append("## 📈 Overall Trend\n")
    md_lines.append(f"![trend]({os.path.basename(trend_img)})\n")

    # 保存 Markdown
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✅ 分析完成: {md_path}")


# =========================
# 6. 入口
# =========================
if __name__ == "__main__":
    BASE_DIR = "/root/Test-Agent/output30_deep_v5"
    OUTPUT_DIR = f"{BASE_DIR}/analysis_result"

    analyze(BASE_DIR, OUTPUT_DIR, max_rule=30, num_iters=4)
