"""plot_and_report.py — 结果可视化与报告生成。

职责：
  - 读取 results/train_log_*.csv 绘制 loss 曲线
  - 读取 results/pe_comparison.csv 绘制 BLEU / token accuracy 对比图
  - 句长分桶对比分析
  - 翻译样例整理
  - 生成 report/report_material.md
"""

import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Dict, Optional


# ── 全局中文字体设置 ──
def _setup_cjk_font():
    """设置 matplotlib 中文字体，优先 Microsoft YaHei。"""
    cjk_candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC',
                      'Noto Sans CJK SC', 'Arial Unicode MS']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in cjk_candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False


_setup_cjk_font()


# ═══════════════════════════════════════════════════════════
# 训练曲线
# ═══════════════════════════════════════════════════════════

def read_train_logs(log_dir: str = 'results') -> Dict[str, Dict[str, List[float]]]:
    """读取 results/train_log_{pe_type}.csv，返回 {pe_type: {'train_loss': [...], 'valid_loss': [...]}}。"""
    raise NotImplementedError


def plot_loss_curves(all_logs: Dict[str, Dict[str, List[float]]],
                     out_path: str = 'figures/loss_curve_comparison.png'):
    """绘制五组位置编码的训练/验证 loss 曲线对比图。"""
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════
# 评价指标对比
# ═══════════════════════════════════════════════════════════

def plot_bleu_comparison(comparison_csv: str = 'results/pe_comparison.csv',
                         out_path: str = 'figures/bleu_comparison.png'):
    """读取 pe_comparison.csv 绘制 BLEU / token accuracy 柱状对比图。"""
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════
# 句长分桶分析
# ═══════════════════════════════════════════════════════════

def plot_length_bucket_comparison(comparison_csv: str = 'results/pe_comparison.csv',
                                  out_path: str = 'figures/length_bucket_comparison.png'):
    """绘制不同句长分桶下的 BLEU 对比图。"""
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════
# 翻译样例整理
# ═══════════════════════════════════════════════════════════

def collect_translation_examples(
    examples_dir: str = 'results',
    pe_types: List[str] = None,
) -> Dict[str, List[Dict]]:
    """读取各 pe_type 的 translation_examples_*.txt 整理为字典。"""
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════
# 报告材料生成
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# CPU Baseline 结果合并
# ═══════════════════════════════════════════════════════════

import json
import re

PE_TYPES = ['none', 'absolute', 'sinusoidal', 'learned', 'gated_mix']


def generate_cpu_baseline_config(out_path: str = 'results/cpu_baseline_config.json'):
    """生成 CPU 轻量实验的配置 JSON。"""
    config = {
        "device": "cpu",
        "train_samples": 8000,
        "epochs": 5,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "batch_size": 32,
        "max_len": 20,
        "learning_rate": 0.0005,
        "dropout": 0.1,
        "label_smoothing": 0.0,
        "seed": 42,
        "src_max_vocab": 15000,
        "tgt_max_vocab": 15000,
        "src_vocab_actual": 8946,
        "tgt_vocab_actual": 6334,
        "train_size": 6381,
        "valid_size": 797,
        "test_size": 799,
        "total_params_none": 3698622,
        "total_params_absolute": 3698622,
        "total_params_sinusoidal": 3698622,
        "total_params_learned": 3703742,
        "total_params_gated_mix": 3698626,
        "optimizer": "Adam",
        "adam_betas": [0.9, 0.98],
        "adam_eps": 1e-9,
        "gradient_clip": 1.0,
        "loss_function": "CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)",
        "src_max_len_actual": 20,
        "tgt_max_len_actual": 20,
        "data_file": "data/deu.txt",
        "data_format": "EN\\tDE\\tATTRIBUTION",
        "task_direction": "de->en",
        "tokenizer": "lowercase + whitespace split",
        "special_tokens": {"pad": "<pad>=0", "bos": "<bos>=1", "eos": "<eos>=2", "unk": "<unk>=3"},
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'  Saved: {out_path}')
    return config


def generate_cpu_baseline_train_logs_csv(out_path: str = 'results/cpu_baseline_train_logs.csv'):
    """合并五组 train_log CSV，加入 pe_type 和 epoch-level 指标。

    Note: 原始 train_log_*.csv 仅有 epoch/train_loss/valid_loss，
    token_acc/short_acc/medium_acc/long_acc 在 epoch 级别不可用（只有最终汇总），
    故这些字段填入空字符串。
    """
    rows = []
    for pe_type in PE_TYPES:
        src = f'results/train_log_{pe_type}.csv'
        with open(src, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    'pe_type': pe_type,
                    'epoch': r['epoch'],
                    'train_loss': r['train_loss'],
                    'valid_loss': r['valid_loss'],
                    'token_acc': '',
                    'short_acc': '',
                    'medium_acc': '',
                    'long_acc': '',
                })

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pe_type', 'epoch', 'train_loss', 'valid_loss',
            'token_acc', 'short_acc', 'medium_acc', 'long_acc'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved: {out_path}  ({len(rows)} rows)')


def generate_cpu_baseline_pe_comparison(out_path: str = 'results/cpu_baseline_pe_comparison.csv'):
    """由现有 pe_comparison.csv 规范化生成 CPU baseline 对比表。"""
    src = 'results/pe_comparison.csv'
    rows = []
    with open(src, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'pe_type': r['pe_type'],
                'train_loss': r['train_loss_final'],
                'valid_loss': r['valid_loss_final'],
                'token_acc': r['token_acc'],
                'short_acc': r['short_acc'],
                'medium_acc': r['medium_acc'],
                'long_acc': r['long_acc'],
                'gate_sin': r.get('gate_sin', ''),
                'gate_abs': r.get('gate_abs', ''),
            })

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pe_type', 'train_loss', 'valid_loss', 'token_acc',
            'short_acc', 'medium_acc', 'long_acc', 'gate_sin', 'gate_abs'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved: {out_path}  ({len(rows)} rows)')


def parse_translation_examples(filepath: str) -> List[Dict[str, str]]:
    """解析单个 translation_examples_*.txt 文件。"""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    blocks = re.split(r'--- Example \d+ ---', text)[1:]  # 跳过文件头
    for block in blocks:
        src_match = re.search(r'Source \(DE\):\s+(.*)', block)
        ref_match = re.search(r'Reference \(EN\):\s+(.*)', block)
        pred_match = re.search(r'Prediction \(EN\):\s+(.*)', block)
        examples.append({
            'source': src_match.group(1).strip() if src_match else '',
            'reference': ref_match.group(1).strip() if ref_match else '',
            'prediction': pred_match.group(1).strip() if pred_match else '',
        })
    return examples


def generate_cpu_baseline_translation_examples_csv(
        out_path: str = 'results/cpu_baseline_translation_examples.csv'):
    """合并五组翻译样例为横向对比 CSV。

    对齐方式：按出现顺序对齐（所有 PE 类型使用相同的 test set，
    固定 seed=42，因此 same example_id 对应 same source/reference）。
    如后续发现不对齐，note 字段已注明对齐假设。
    """
    all_preds = {}
    sources = None
    references = None

    for pe_type in PE_TYPES:
        fp = f'results/translation_examples_{pe_type}.txt'
        examples = parse_translation_examples(fp)
        all_preds[pe_type] = [ex['prediction'] for ex in examples]
        if sources is None:
            sources = [ex['source'] for ex in examples]
            references = [ex['reference'] for ex in examples]

    n = len(sources)
    rows = []
    for i in range(n):
        src_tokens = sources[i].split()
        ref_len = len(references[i].split())
        if ref_len <= 6:
            bucket = 'short'
        elif ref_len <= 12:
            bucket = 'medium'
        else:
            bucket = 'long'

        row = {
            'example_id': i + 1,
            'length_bucket': bucket,
            'source_de': sources[i],
            'reference_en': references[i],
            'pred_none': all_preds['none'][i],
            'pred_absolute': all_preds['absolute'][i],
            'pred_sinusoidal': all_preds['sinusoidal'][i],
            'pred_learned': all_preds['learned'][i],
            'pred_gated_mix': all_preds['gated_mix'][i],
            'note': 'Aligned by index; same test set & seed=42 across PE types',
        }
        rows.append(row)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'example_id', 'length_bucket', 'source_de', 'reference_en',
            'pred_none', 'pred_absolute', 'pred_sinusoidal',
            'pred_learned', 'pred_gated_mix', 'note'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved: {out_path}  ({len(rows)} rows, {n} examples x 5 PE types)')


# ═══════════════════════════════════════════════════════════
# CPU Baseline 对比图（英文标题）
# ═══════════════════════════════════════════════════════════

def plot_cpu_baseline_token_accuracy(
        csv_path: str = 'results/cpu_baseline_pe_comparison.csv',
        out_path: str = 'figures/cpu_baseline_token_accuracy_comparison.png'):
    """绘制 CPU baseline token accuracy 柱状对比图。"""
    pe_types = []
    accs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pe_types.append(r['pe_type'])
            accs.append(float(r['token_acc']))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#888888', '#E69F00', '#56B4E9', '#009E73', '#CC79A7']
    bars = ax.bar(pe_types, accs, color=colors, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{v:.4f}', ha='center', fontsize=9)

    ax.set_ylabel('Token Accuracy')
    ax.set_title('CPU Baseline: Token Accuracy by Position Encoding Type')
    ax.set_ylim(0, max(accs) * 1.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_cpu_baseline_length_bucket(
        csv_path: str = 'results/cpu_baseline_pe_comparison.csv',
        out_path: str = 'figures/cpu_baseline_length_bucket_comparison.png'):
    """绘制 CPU baseline 句长分桶 token accuracy 对比图。"""
    data = {'pe_type': [], 'short': [], 'medium': [], 'long': []}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            data['pe_type'].append(r['pe_type'])
            data['short'].append(float(r['short_acc']))
            data['medium'].append(float(r['medium_acc']))
            data['long'].append(float(r['long_acc']))

    x = range(len(data['pe_type']))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], data['short'], width, label='Short (<=6)',
           color='#56B4E9', edgecolor='white')
    ax.bar(x, data['medium'], width, label='Medium (7-12)',
           color='#E69F00', edgecolor='white')
    ax.bar([i + width for i in x], data['long'], width, label='Long (>12)',
           color='#D55E00', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(data['pe_type'])
    ax.set_ylabel('Token Accuracy')
    ax.set_title('CPU Baseline: Token Accuracy by Sentence Length Bucket')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_cpu_baseline_valid_loss(
        csv_path: str = 'results/cpu_baseline_pe_comparison.csv',
        out_path: str = 'figures/cpu_baseline_valid_loss_comparison.png'):
    """绘制 CPU baseline valid loss 柱状对比图。"""
    pe_types = []
    losses = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pe_types.append(r['pe_type'])
            losses.append(float(r['valid_loss']))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#888888', '#E69F00', '#56B4E9', '#009E73', '#CC79A7']
    bars = ax.bar(pe_types, losses, color=colors, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', fontsize=9)

    ax.set_ylabel('Valid Loss')
    ax.set_title('CPU Baseline: Validation Loss by Position Encoding Type')
    min_loss = min(losses)
    ax.set_ylim(min_loss * 0.995, max(losses) * 1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def generate_report_material(out_path: str = 'report/report_material.md'):
    """读取 results/ 和 figures/ 中的所有真实结果，生成报告材料 Markdown。

    报告结构：
    标题、作者、摘要、关键词
    一、引言
    二、数据集说明、可视化与预处理
    三、模型方法
    四、实验设计与实验步骤
    五、实验结果与分析
    六、位置编码机制讨论与创新实验分析
    七、结论
    八、参考文献

    必须基于真实结果文件生成，不编造数值。
    """
    raise NotImplementedError
