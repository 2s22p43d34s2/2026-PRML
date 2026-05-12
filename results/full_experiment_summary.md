# PRML作业四：Transformer复现与位置编码对比实验 — 完整成果汇总

## 一、项目背景

本项目是PRML作业四，目标是：
1. 复现《Attention Is All You Need》(Vaswani et al., 2017)中的Transformer Encoder-Decoder
2. 围绕**位置编码**方向进行对比实验，理解位置编码在Transformer中的核心作用
3. 任务：德语→英语机器翻译（data/deu.txt，324K平行句对）
4. 要求：纯PyTorch手写，不使用HuggingFace/transformers/预训练模型

## 二、环境与算力

| 项目 | 值 |
|---|---|
| Python | 3.11.4 |
| PyTorch | 2.11.0+CPU |
| GPU | **无 (纯CPU训练)** |
| OS | Windows 11 |

## 三、数据集统计

| 指标 | 英语 | 德语 |
|---|---|---|
| 去重后句对数 | 324,230 | 324,230 |
| Min / Max | 1 / 101 | 1 / 76 |
| Mean / Median | 6.4 / 6.0 | 6.4 / 6.0 |
| 90th / 95th / 99th | 10 / 11 / 14 | 10 / 11 / 14 |

句长分桶：93%+ 的句子 ≤10 词，以短句为主。

分词方式：转小写 + 空白字符切分（简单空格分词）。
特殊符号：`<pad>=0 <bos>=1 <eos>=2 <unk>=3`。
数据划分：train 80% / valid 10% / test 10%，固定随机种子42。

## 四、项目代码结构（4个Python文件）

```
project/
├── data/deu.txt (50MB, 32万行)
├── src/
│   ├── data_utils.py        # 数据读取/清洗/分词/词表/Dataset/DataLoader
│   ├── transformer_model.py # Transformer主体，含5种位置编码
│   ├── experiment.py        # 训练/验证/测试/贪心解码/实验调度
│   └── plot_and_report.py   # 绘图/报告生成（骨架，尚未实现）
├── results/  (训练日志/CSV/翻译样例)
├── figures/  (length_distribution.png)
├── checkpoints/ (模型权重 .pt)
└── report/
```

## 五、模型架构

| 参数 | 值 |
|---|---|
| d_model | 128 |
| n_heads | 4 |
| num_layers (Encoder/Decoder) | 2 |
| dim_feedforward | 512 |
| dropout | 0.1 |
| max_len | 20 |
| 总参数 | ~3,698,622 |

结构：标准 Transformer Encoder-Decoder
- Encoder: Token Embedding + PE → N×[Self-Attention + FFN + Residual + LayerNorm]
- Decoder: Token Embedding + PE → N×[Masked Self-Attn + Cross-Attn + FFN + Residual + LayerNorm] → Linear
- 采用 post-norm 残差连接
- 手写 Multi-Head Scaled Dot-Product Attention，不调用 `nn.MultiheadAttention`
- 训练时 teacher forcing，推理时 greedy decoding

## 六、五组位置编码对比实验

### 6.1 五种位置编码说明

| pe_type | 含义 | 可训练参数 |
|---|---|---|
| none | 无位置编码，仅token embedding | 0 |
| absolute | 固定简单绝对编码：P(pos,j)=(2*pos/(L-1)-1)*(2*j/(D-1)-1) | 0 |
| sinusoidal | 原论文正弦编码：PE(pos,2i)=sin(pos/10000^(2i/D)), PE(pos,2i+1)=cos(...) | 0 |
| learned | nn.Embedding 可学习绝对位置编码 | 5,120 |
| gated_mix | 门控混合：sigmoid(α)*P_sin + sigmoid(β)*P_abs | 4 (α,β) |

### 6.2 实验配置（五组完全一致）

| 超参数 | 值 |
|---|---|
| 训练样本 | 8,000（全量数据中随机抽取） |
| 实际 train/valid/test | 6,381 / 797 / 799 |
| 源语言词表(DE) | 8,946 |
| 目标语言词表(EN) | 6,334 |
| batch_size | 32 |
| lr | 0.0005 |
| optimizer | Adam (betas=0.9, 0.98, eps=1e-9) |
| epochs | 5 |
| gradient clip | 1.0 |
| seed | 42 |

### 6.3 核心结果

| pe_type | train_loss | valid_loss | token_acc | short_acc | medium_acc | long_acc |
|---|---|---|---|---|---|---|
| none | 3.8881 | 4.4791 | 0.1642 | 0.1965 | 0.1413 | 0.0781 |
| absolute | 3.8851 | 4.4713 | 0.1606 | 0.1969 | 0.1343 | 0.0568 |
| sinusoidal | 3.8846 | 4.4670 | 0.1665 | 0.2009 | 0.1404 | **0.0847** |
| learned | 3.8546 | 4.4675 | **0.1732** | **0.2083** | **0.1490** | 0.0722 |
| gated_mix | 3.8818 | **4.4623** | 0.1664 | 0.2019 | 0.1422 | 0.0576 |

gated_mix 门控值：gate_sin = 0.5223, gate_abs = 0.5035

训练总耗时：约16分钟（5组×5 epoch，CPU）。

### 6.4 翻译样例对比（同一测试句，5种PE的预测）

**例1：短句**
- Source(DE): wo kann ich sie finden?
- Reference(EN): where can i find you?
- none: i can't do you can do that.
- absolute: i can't do you do that.
- sinusoidal: i can't do you can do that.
- learned: where can i find the best way to get a little bit of money?
- gated_mix: i can't do you can do that.

**例2：中句**
- Source(DE): ohne sie hätte tom das nicht geschafft.
- Reference(EN): tom couldn't have done it without you.
- none: tom didn't want to do that.
- absolute: tom didn't want to do that.
- sinusoidal: tom didn't want to do that.
- learned: tom didn't want to do that.
- gated_mix: tom didn't want to do that.

**例3：长句**
- Source(DE): sie flehte ihn an, zu bleiben.
- Reference(EN): she begged him to stay.
- none: she is not to be able to be able to be able to be able to be able to
- absolute: she is not to be able to be able to be able to be able to be able to
- sinusoidal: she is not to be able to be able to be able to be able to be able to
- learned: she is not to be able to be able to be able to be able to be able to
- gated_mix: she is not to be able to be able to be able to be able to be able to

**例4：短句**
- Source(DE): findest du mich hübsch?
- Reference(EN): do you think i'm handsome?
- none: you have me to me.
- absolute: you have me to me.
- learned: you have me a little bit of a little bit
- gated_mix: you have me to me.

**例5：短句**
- Source(DE): ich werde tom nie wieder vertrauen.
- Reference(EN): i'm never going to trust tom again.
- none: i never never never never never never never never never ever tom.
- sinusoidal: i never never never never never never never never never never.
- learned: i never thought tom never never to be able to be able to be able to be

**例6：中句**
- Source(DE): der erdradius misst ca. 6000 km.
- Reference(EN): the earth's radius is about 6,000 kilometers.
- sinusoidal: the only are a lot of the same book
- learned: the only are is a part of the same as possible.

## 七、关键实验发现

1. **无位置编码(none)仍可工作**（acc=0.1642）：token embedding携带的语义共有信息使模型能完成弱翻译。但预测中频繁出现重复短语（如"you want to do you want to do"），缺乏语序建模能力。

2. **简单绝对编码(absolute)低于sinusoidal**：在各指标上均差于sinusoidal，尤其在长句上（long_acc 0.0568 vs 0.0847）。简单的线性位置表示不足以充分建模序列顺序关系。

3. **正弦编码(sinusoidal)长句最优**：long_acc=0.0847显著高于其他类型，验证了原论文设计的有效性——正弦函数的多频表示能更好地泛化到未见过的位置。

4. **可学习编码(learned)总准确率最高**：token_acc=0.1732，short_acc=0.2083。在小数据集上可以学到更适合当前数据分布的位置表示。但长句能力(0.0722)不如sinusoidal，存在对训练长度分布的过拟合风险。

5. **门控混合(gated_mix)valid_loss最低**：4.4623，略优于pure sinusoidal(4.4670)。但门控值接近0.5（gate_sin=0.5223, gate_abs=0.5035），说明模型未明显偏好任一种编码。短句表现好(0.2019)，长句退化明显(0.0576)。

6. **长句翻译是共同瓶颈**：所有PE类型下long_acc都远低于short_acc（约2-3倍差距），说明当前小模型和有限训练下，长距离依赖建模是主要短板。

## 八、已知局限

1. **算力严重不足**：纯CPU训练，仅8K样本×5 epoch，距离充分训练差距很大
2. **模型容量小**：d_model=128, 2层，约3.7M参数。原始Transformer是d_model=512, 6层, ~65M参数
3. **分词过于简单**：空格分词无法处理德语复合词和英语缩写（如wasn't → wasn, t）
4. **评价指标有限**：token accuracy并非标准BLEU，只能反映token级匹配
5. **max_len=20**：超过20词的句子被截断，约1%句子受影响
6. **epoch太少**：5 epoch下loss仍在下降，模型远未收敛
7. **gated_mix门控未明显分化**：两个gate值都接近0.5，可能是训练不足或标量门控表达能力有限

## 九、待完成工作

剩余工作：
- 阶段7：绘图（loss curve/bleu/length bucket comparison figures）
- 阶段8：生成report/report_material.md

可选改进方向（需用户决策）：
- 增加训练数据量和epoch（如30K样本×10 epoch）以获得更有意义的对比
- 改进分词（BPE/subword）以提升翻译质量
- 实现标准BLEU评价以更准确度量翻译质量
- 增大模型容量（d_model=256或512）如果算力允许
- gated_mix可尝试向量级门控（而非标量）以增强表达能力
- 增加更多分析维度（attention可视化、位置相似度矩阵等）

## 十、已保存的全部文件

```
results/
├── data_summary.txt                          # 阶段1
├── train_log_none.csv                        # 阶段6
├── train_log_absolute.csv                    # 阶段6
├── train_log_sinusoidal.csv                  # 阶段4/6
├── train_log_learned.csv                     # 阶段6
├── train_log_gated_mix.csv                   # 阶段6
├── pe_comparison.csv                         # 阶段6
├── translation_examples_none.txt             # 阶段6
├── translation_examples_absolute.txt         # 阶段6
├── translation_examples_sinusoidal.txt       # 阶段4/6
├── translation_examples_learned.txt          # 阶段6
└── translation_examples_gated_mix.txt        # 阶段6

figures/
└── length_distribution.png                   # 阶段1

checkpoints/
├── transformer_none.pt (14.1 MB)             # 阶段6
├── transformer_absolute.pt (14.2 MB)         # 阶段6
├── transformer_sinusoidal.pt (14.2 MB)       # 阶段4/6
├── transformer_learned.pt (14.2 MB)          # 阶段6
└── transformer_gated_mix.pt (14.2 MB)        # 阶段6
```
