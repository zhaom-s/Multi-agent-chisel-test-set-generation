import os
import shutil
import re
import logging
import matplotlib.pyplot as plt
from collections import defaultdict

# --- 1. 定义绘图与日志辅助函数 ---

def setup_logging(output_dir):
    """设置日志配置：同时输出到控制台和文件"""
    log_file = os.path.join(output_dir, "collection_summary.log")
    
    # 清理已有的 logger 处理器，防止重复
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),
            logging.StreamHandler()
        ]
    )
    return log_file

def generate_chart(stats, max_id, output_dir):
    """生成总览堆叠柱状图"""
    ids = list(range(1, max_id + 1))
    t_counts = [stats[i]["T"] for i in ids]
    f_counts = [stats[i]["F"] for i in ids]

    plt.figure(figsize=(22, 8)) # 进一步加宽以适应 240 条数据
    plt.bar(ids, t_counts, label='T (Positive)', color='#3498db', alpha=0.8)
    plt.bar(ids, f_counts, label='F (Negative)', color='#e74c3c', alpha=0.6, bottom=t_counts)
    plt.xlabel('Rule ID')
    plt.ylabel('Samples Count')
    plt.title(f'Total Data Collection Coverage (Rules 1-{max_id})')
    
    # 240 条规则下，每 10 条显示一个刻度
    plt.xticks(range(0, max_id + 1, 10))
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "coverage_total_stacked.png"), dpi=300)
    plt.close()

def generate_detailed_chart(stats, max_id, suffix, output_dir):
    """生成 T 或 F 的专项分析图"""
    ids = list(range(1, max_id + 1))
    counts = [stats[i][suffix] for i in ids]
    
    covered_count = sum(1 for c in counts if c > 0)
    coverage_pct = (covered_count / max_id) * 100
    avg_count = sum(counts) / max_id if max_id > 0 else 0
    
    plt.figure(figsize=(22, 8))
    color = '#3498db' if suffix == 'T' else '#e74c3c'
    label_name = "Positive (T)" if suffix == 'T' else "Negative (F)"
    
    plt.bar(ids, counts, color=color, alpha=0.7, label=f'{label_name} Counts')
    plt.axhline(y=avg_count, color='green', linestyle='--', linewidth=1.5, label=f'Average: {avg_count:.2f}')
    
    stats_text = f"Coverage: {covered_count}/{max_id} ({coverage_pct:.1f}%)\nAvg Samples/Rule: {avg_count:.2f}"
    plt.text(0.01, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('Rule ID')
    plt.ylabel('Number of Files')
    plt.title(f'Rule Coverage Analysis - Type: {label_name} (Total {max_id} Rules)')
    
    plt.xticks(range(0, max_id + 1, 10))
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.savefig(os.path.join(output_dir, f"coverage_{suffix}_detail.png"), dpi=300)
    plt.close()

# --- 2. 核心收集与分析逻辑 ---

def collect_scala_files(src_dirs, output_dir, max_rule_id=240):
    # 1. 预清理
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"🧹 正在预清理目录: {output_dir}")
    for f in os.listdir(output_dir):
        if f.endswith((".scala", ".png", ".log")):
            try: os.remove(os.path.join(output_dir, f))
            except: pass

    # 2. 初始化日志
    log_path = setup_logging(output_dir)
    logging.info(f"🚀 开始执行 Scala 数据汇总 (目标规则数: {max_rule_id})...")

    name_counter = defaultdict(int)
    coverage_stats = defaultdict(lambda: {"T": 0, "F": 0})
    total_files_count = 0

    # 3. 收集文件
    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            logging.warning(f"⚠️ 跳过不存在的目录: {src_dir}")
            continue

        files = [f for f in os.listdir(src_dir) if f.endswith(".scala")]
        new_contributions = 0
        for fname in files:
            match = re.match(r"rule_(\d+)_(T|F)\.scala", fname)
            if not match: continue
            
            rid, suffix = int(match.group(1)), match.group(2)
            if rid > max_rule_id: continue

            base_key = f"rule_{rid}_{suffix}"
            count = name_counter[base_key]
            # 自动处理同名文件覆盖，重命名为 rule_X.1_T.scala 等
            new_name = f"rule_{rid}_{suffix}.scala" if count == 0 else f"rule_{rid}.{count}_{suffix}.scala"
            
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(output_dir, new_name))
            
            name_counter[base_key] += 1
            coverage_stats[rid][suffix] += 1
            new_contributions += 1
            total_files_count += 1
        
        logging.info(f"📂 扫描目录: {os.path.basename(os.path.dirname(src_dir)):<15} | 贡献新增: {new_contributions}")

    # 4. 统计
    max_id = max_rule_id
    both, only_t, only_f, none = [], [], [], []
    for i in range(1, max_id + 1):
        has_t, has_f = coverage_stats[i]["T"] > 0, coverage_stats[i]["F"] > 0
        if has_t and has_f: both.append(i)
        elif has_t: only_t.append(i)
        elif has_f: only_f.append(i)
        else: none.append(i)

    # 5. 打印完整报告 (关键点：每一行都要用 logging.info)
    logging.info("\n" + "="*65)
    logging.info(f"📊 数据完整性汇总报告 (Rule ID 1-{max_id})")
    logging.info("="*65)
    logging.info(f"✅ 最终收集文件总数: {total_files_count}")
    logging.info(f"🔹 双向生成 (T&F): {len(both):>3} | 占比: {len(both)/max_id:.1%}")
    logging.info(f"🔹 缺失 F 样本:   {len(only_t):>3} | 占比: {len(only_t)/max_id:.1%}")
    logging.info(f"🔹 缺失 T 样本:   {len(only_f):>3} | 占比: {len(only_f)/max_id:.1%}")
    logging.info(f"🔹 完全缺失样本:  {len(none):>3} | 占比: {len(none)/max_id:.1%}")
    logging.info("-" * 65)
    if none:   logging.info(f"❌ 缺失 Rule IDs (T和F均无): {none}")
    if only_t: logging.info(f"⚠️ 缺少 F 样本的 IDs: {only_t}")
    if only_f: logging.info(f"⚠️ 缺少 T 样本的 IDs: {only_f}")
    logging.info("="*65)

    # 6. 图表
    generate_chart(coverage_stats, max_id, output_dir)
    generate_detailed_chart(coverage_stats, max_id, 'T', output_dir)
    generate_detailed_chart(coverage_stats, max_id, 'F', output_dir)
    
    logging.info(f"📈 统计图表已保存至: {output_dir}")
    logging.info(f"📑 详细日志已写入: {log_path}")

    # 7. 强制刷新与关闭（解决日志内容不全的关键）
    for handler in logging.root.handlers[:]:
        handler.flush()
        handler.close()
        logging.root.removeHandler(handler)

# --- 3. 程序入口 ---

if __name__ == "__main__":
    # 配置区
    MAX_RULES = 240
    BASE_DIR = "/root/Test-Agent/output_results"
    
    # 输入：0-5 次迭代的 verified_pass 目录
    src_paths = [f"{BASE_DIR}/iteration_{i}/verified_pass" for i in range(6)]
    # 输出：汇总目录
    out_path = f"{BASE_DIR}/scala_files240"

    collect_scala_files(src_paths, out_path, max_rule_id=MAX_RULES)