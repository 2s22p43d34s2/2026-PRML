# PRML作业四：基于德英翻译任务的Transformer复现及位置编码机制对比研究

## 项目简介

本项目是模式识别与机器学习课程作业四。任务为德语到英语机器翻译，模型为纯PyTorch手写Transformer Encoder-Decoder，不使用HuggingFace、transformers库或预训练模型。实验围绕位置编码（Positional Encoding）机制展开，对比五种不同位置编码方式对翻译性能的影响。

## 任务定义

- 输入：德语句子
- 输出：英语句子
- 数据：data/deu.txt（德英平行句对，约32万对）
- 评价指标：token accuracy（非BLEU）

## 五种位置编码

| pe_type | 含义 |
|---|---|
| none | 无位置编码 |
| absolute | 简单绝对位置编码：P(pos,j) = (2pos/(L-1)-1) × (2j/(D-1)-1) |
| sinusoidal | 原论文正弦位置编码（Vaswani et al., 2017） |
| learned | 可学习绝对位置编码（nn.Embedding） |
| gated_mix | 门控混合位置编码：sigmoid(α)×P_sin + sigmoid(β)×P_abs |

## 实验版本

| 版本 | 样本 | epoch | d_model | 用途 |
|---|---|---|---|---|
| CPU基线版 | 8,000 | 5 | 128 | 轻量验证 |
| GPU 2h版 | 50,000 | 12 | 256 | 快速GPU验证 |
| GPU Big版（**主结果**） | 100,000 | 20 | 256 | 最终报告版 |

最终主结果来自GPU Big版实验（NVIDIA RTX 4060 Laptop 8GB，约2.2小时训练）。

## 主要实验结果（GPU Big）

| pe_type | valid_loss(验证集) | token_acc(测试集) | short_acc(测试集) | medium_acc(测试集) | long_acc(测试集) |
|---|---|---|---|---|---|
| none | 3.0440 | 0.4684 | 0.5429 | 0.4265 | 0.1893 |
| absolute | 3.0421 | 0.4649 | 0.5460 | 0.4186 | 0.1795 |
| sinusoidal | 3.0336 | 0.4679 | 0.5416 | 0.4268 | 0.1959 |
| learned | 3.0002 | 0.4979 | 0.5715 | 0.4586 | 0.2208 |
| gated_mix | 3.0372 | 0.4680 | 0.5462 | 0.4235 | 0.1905 |

详细结果见 `results/gpu_big_pe_comparison.csv`、`results/gpu_big_train_logs.csv`。

## 项目结构

```
├── README.md
├── .gitignore
├── data/
│   ├── _about.txt
│   └── deu.txt
├── src/
│   ├── data_utils.py         # 数据管线（读取/清洗/分词/词表/Dataset/DataLoader）
│   ├── transformer_model.py  # Transformer模型（5种PE/多头注意力/Encoder/Decoder）
│   ├── experiment.py         # 训练/验证/测试/贪心解码/实验调度
│   └── plot_and_report.py    # 结果可视化与报告生成
├── results/
│   ├── gpu_big_pe_comparison.csv
│   ├── gpu_big_train_logs.csv
│   ├── gpu_big_translation_examples.csv
│   ├── gpu_big_config.json
│   └── ...（其他版本结果）
└── figures/
```

## 环境要求

- Python 3.11+
- PyTorch 2.6+ (CUDA版本，需NVIDIA GPU)
- matplotlib

安装依赖：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install matplotlib
```

## 数据准备

本项目使用的德英平行语料来自Tatoeba项目（CC-BY 2.0许可）。

`data/deu.txt` 已包含在仓库中。文件格式为：

```
英语句子\t德语句子\t版权信息
```

每行3列，Tab分隔。`data/_about.txt` 中记录了数据来源。

## 模型checkpoint

`checkpoints/` 目录中的 `.pt` 文件未上传。可通过以下命令重新训练生成：

```bash
python -c "from src.experiment import run_single_experiment; ..."
```

各组实验的完整配置见 `results/gpu_big_config.json`。

## 复现实验

1. 安装依赖：`pip install torch matplotlib`
2. 运行完整五组实验：

```python
from src.experiment import run_single_experiment
import torch

for pe_type in ['none', 'absolute', 'sinusoidal', 'learned', 'gated_mix']:
    result = run_single_experiment(
        pe_type=pe_type,
        data_config={'data_path': 'data/deu.txt', 'max_samples': 100000,
                     'src_max_len': 30, 'tgt_max_len': 30,
                     'src_max_vocab': 30000, 'tgt_max_vocab': 30000,
                     'batch_size': 128},
        model_config={'d_model': 256, 'n_heads': 8, 'num_layers': 3,
                      'dim_feedforward': 1024, 'max_len': 30, 'dropout': 0.1},
        train_config={'lr': 0.0003, 'n_epochs': 20, 'clip': 1.0, 'seed': 42,
                      'label_smoothing': 0.1, 'ckpt_prefix': 'gpu_big'},
        device=torch.device('cuda'),
    )
```

4. 使用 `src/plot_and_report.py` 生成对比图和报告材料

## 注意事项

- 本实验为课程作业，模型规模有限（d_model=256, 3层），翻译质量未达到工业级水平
- 当前评价指标为token accuracy，非标准BLEU分数
- 训练需要NVIDIA GPU，CPU训练速度较慢
- 完整五组实验约需2-2.5小时（RTX 4060 Laptop）
- 通过 `max_samples` 参数可减少样本量进行快速测试
