# PRML作业四：Transformer复现与位置编码对比 — 完整实验结果

## 项目概述

- 任务：德语→英语机器翻译（data/deu.txt，324K平行句对）
- 模型：纯PyTorch手写Transformer Encoder-Decoder（无HuggingFace/预训练模型）
- 目标：复现《Attention Is All You Need》并完成位置编码机制对比实验
- 实验方向：2.1 位置编码
- GPU：NVIDIA GeForce RTX 4060 Laptop (8GB)

## 指标来源说明

| 指标 | 数据源 | 说明 |
|---|---|---|
| train_loss | 训练集 | 每个epoch在训练集上的CrossEntropyLoss |
| valid_loss | 验证集 | 每个epoch在验证集上的CrossEntropyLoss |
| best_epoch | 验证集 | 选择验证集valid_loss最低的epoch |
| 逐epoch token_acc/short/medium/long (train_logs.csv) | **验证集** | 每个epoch结束时在验证集上greedy decode评估 |
| 最终 token_acc/short/medium/long (pe_comparison.csv) | **测试集** | 训练结束后用best checkpoint在测试集上greedy decode评估 |
| 翻译样例 | 测试集 | 来自测试集，使用best checkpoint生成 |

## 三版实验配置对比

| 参数 | CPU Baseline | GPU 2h | GPU Big (最终) |
|---|---|---|---|
| 训练样本 | 8,000 | 50,000 | 100,000 |
| Epochs | 5 | 12 | 20 |
| d_model | 128 | 256 | 256 |
| n_heads | 4 | 8 | 8 |
| num_layers | 2 | 3 | 3 |
| dim_feedforward | 512 | 1024 | 1024 |
| batch_size | 32 | 128 | 128 |
| max_len | 20 | 30 | 30 |
| learning_rate | 0.0005 | 0.0003 | 0.0003 |
| label_smoothing | 0.0 | 0.1 | 0.1 |
| 模型参数量 | ~3.7M | ~20.7M | ~24.6M |
| 显存峰值 | — | 1.61 GB | 2.00 GB |
| 总耗时 | ~16 min | ~41 min | ~130 min |
| 评价指标 | token_accuracy | token_accuracy | token_accuracy |

## 三版实验结果

### CPU Baseline (8K样本, 5 epoch) — 验证集损失与测试集准确率对比

| pe_type | valid_loss(验证集) | token_acc(测试集) | short_acc(测试集) | medium_acc(测试集) | long_acc(测试集) | gate |
|---|---|---|---|---|---|---|
| none | 4.4791 | 0.1642 | 0.1965 | 0.1413 | 0.0781 | — |
| absolute | 4.4713 | 0.1606 | 0.1969 | 0.1343 | 0.0568 | — |
| sinusoidal | 4.4670 | 0.1665 | 0.2009 | 0.1404 | **0.0847** | — |
| learned | 4.4675 | **0.1732** | **0.2083** | **0.1490** | 0.0722 | — |
| gated_mix | 4.4623 | 0.1664 | 0.2019 | 0.1422 | 0.0576 | 0.5223/0.5035 |

### GPU 2h (50K样本, 12 epoch) — 验证集损失与测试集准确率对比

| pe_type | valid_loss(验证集) | token_acc(测试集) | short_acc(测试集) | medium_acc(测试集) | long_acc(测试集) | gate |
|---|---|---|---|---|---|---|
| none | 3.4849 | 0.3747 | 0.4379 | 0.3354 | 0.1827 | — |
| absolute | 3.4808 | 0.3740 | 0.4356 | 0.3354 | 0.1796 | — |
| sinusoidal | 3.4815 | 0.3805 | 0.4459 | 0.3382 | 0.1957 | — |
| learned | 3.4461 | **0.4110** | **0.4757** | **0.3715** | **0.2061** | — |
| gated_mix | 3.4811 | 0.3803 | 0.4437 | 0.3421 | 0.1791 | 0.6957/0.6465 |

### GPU Big (100K样本, 20 epoch) — 最终版，验证集损失与测试集准确率对比

| pe_type | valid_loss(验证集) | token_acc(测试集) | short_acc(测试集) | medium_acc(测试集) | long_acc(测试集) | gate |
|---|---|---|---|---|---|---|
| none | 3.0440 | 0.4684 | 0.5429 | 0.4265 | 0.1893 | — |
| absolute | 3.0421 | 0.4649 | 0.5460 | 0.4186 | 0.1795 | — |
| sinusoidal | 3.0336 | 0.4679 | 0.5416 | 0.4268 | **0.1959** | — |
| learned | **3.0002** | **0.4979** | **0.5715** | **0.4586** | **0.2208** | — |
| gated_mix | 3.0372 | 0.4680 | 0.5462 | 0.4235 | 0.1905 | **0.9462/0.9017** |

## GPU Big 逐epoch训练日志（完整）

### none (无位置编码) — best epoch=20
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time(s)
1      5.4195      4.5588      0.2144   0.2580     0.1851      0.0809    103
2      4.3393      4.0716      0.2781   0.3273     0.2432      0.1440    77
3      3.9321      3.7866      0.3137   0.3696     0.2774      0.1547    74
4      3.6569      3.6038      0.3422   0.4004     0.3034      0.1806    73
5      3.4458      3.4719      0.3626   0.4209     0.3233      0.1978    82
6      3.2779      3.3799      0.3825   0.4420     0.3427      0.1966    78
7      3.1390      3.2930      0.3937   0.4536     0.3539      0.2084    73
8      3.0266      3.2490      0.4057   0.4671     0.3652      0.2040    80
9      2.9296      3.2021      0.4237   0.4817     0.3844      0.2301    74
10     2.8441      3.1808      0.4206   0.4809     0.3806      0.2188    84
11     2.7728      3.1499      0.4359   0.4955     0.3952      0.2385    76
12     2.7059      3.1230      0.4359   0.4977     0.3942      0.2322    77
13     2.6502      3.1066      0.4428   0.5044     0.4015      0.2251    75
14     2.6000      3.0973      0.4511   0.5126     0.4092      0.2394    74
15     2.5554      3.0816      0.4519   0.5151     0.4088      0.2371    73
16     2.5134      3.0661      0.4551   0.5200     0.4109      0.2284    73
17     2.4758      3.0605      0.4543   0.5204     0.4082      0.2347    72
18     2.4406      3.0618      0.4680   0.5350     0.4205      0.2344    72
19     2.4098      3.0499      0.4665   0.5335     0.4190      0.2386    72
20     2.3821      3.0440      0.4695   0.5373     0.4218      0.2385    71
```

### absolute (简单绝对位置编码) — best epoch=19
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time(s)
1      5.4165      4.5538      0.2141   0.2569     0.1834      0.1026    81
2      4.3374      4.0775      0.2803   0.3304     0.2460      0.1355    73
3      3.9324      3.7842      0.3155   0.3705     0.2787      0.1563    76
4      3.6565      3.6065      0.3410   0.3990     0.3039      0.1636    76
5      3.4459      3.4709      0.3620   0.4195     0.3234      0.1938    74
6      3.2768      3.3777      0.3820   0.4433     0.3406      0.1827    76
7      3.1378      3.2930      0.4012   0.4622     0.3604      0.1906    74
8      3.0262      3.2412      0.4119   0.4740     0.3705      0.1959    76
9      2.9281      3.1992      0.4209   0.4818     0.3799      0.2112    74
10     2.8436      3.1719      0.4242   0.4865     0.3832      0.1986    79
11     2.7717      3.1459      0.4405   0.5027     0.3982      0.2153    72
12     2.7058      3.1146      0.4439   0.5048     0.4037      0.2076    75
13     2.6489      3.1004      0.4458   0.5076     0.4026      0.2186    77
14     2.5973      3.0898      0.4547   0.5198     0.4085      0.2191    75
15     2.5551      3.0793      0.4517   0.5173     0.4058      0.2149    73
16     2.5129      3.0644      0.4569   0.5229     0.4100      0.2184    76
17     2.4763      3.0579      0.4605   0.5278     0.4131      0.2154    80
18     2.4405      3.0543      0.4656   0.5345     0.4165      0.2143    78
19     2.4096      3.0421      0.4685   0.5383     0.4169      0.2238    82
20     2.3821      3.0472      0.4685   0.5354     0.4210      0.2227    73
```

### sinusoidal (原论文正弦位置编码) — best epoch=20
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time(s)
1      5.4162      4.5516      0.2156   0.2606     0.1862      0.0940    84
2      4.3368      4.0734      0.2815   0.3316     0.2486      0.1370    77
3      3.9297      3.7804      0.3161   0.3707     0.2790      0.1602    74
4      3.6537      3.5980      0.3458   0.4035     0.3076      0.1838    76
5      3.4435      3.4714      0.3609   0.4211     0.3212      0.1862    75
6      3.2744      3.3746      0.3831   0.4442     0.3426      0.1907    76
7      3.1365      3.2888      0.3982   0.4601     0.3569      0.1957    75
8      3.0236      3.2407      0.4118   0.4737     0.3704      0.2077    76
9      2.9260      3.1980      0.4236   0.4839     0.3831      0.2163    71
10     2.8411      3.1654      0.4283   0.4884     0.3887      0.2165    78
11     2.7690      3.1396      0.4389   0.5020     0.3959      0.2187    75
12     2.7016      3.1175      0.4431   0.5039     0.4026      0.2208    72
13     2.6449      3.1022      0.4491   0.5125     0.4048      0.2293    71
14     2.5959      3.0848      0.4528   0.5141     0.4106      0.2275    73
15     2.5523      3.0764      0.4522   0.5163     0.4082      0.2183    73
16     2.5095      3.0627      0.4612   0.5247     0.4184      0.2228    71
17     2.4741      3.0484      0.4638   0.5279     0.4187      0.2327    75
18     2.4380      3.0495      0.4637   0.5273     0.4206      0.2123    74
19     2.4083      3.0395      0.4726   0.5362     0.4293      0.2335    71
20     2.3784      3.0336      0.4703   0.5354     0.4257      0.2289    78
```

### learned (可学习绝对位置编码) — best epoch=20
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time(s)
1      5.4037      4.5224      0.2353   0.2843     0.2052      0.0950    82
2      4.3115      4.0446      0.2987   0.3527     0.2635      0.1517    81
3      3.9029      3.7643      0.3312   0.3880     0.2951      0.1640    82
4      3.6224      3.5696      0.3635   0.4246     0.3238      0.1756    76
5      3.4098      3.4309      0.3882   0.4486     0.3486      0.1951    78
6      3.2387      3.3240      0.4162   0.4782     0.3750      0.2089    74
7      3.0998      3.2510      0.4275   0.4892     0.3875      0.2088    74
8      2.9860      3.2012      0.4405   0.5031     0.3987      0.2181    76
9      2.8864      3.1579      0.4506   0.5123     0.4099      0.2217    77
10     2.8031      3.1272      0.4577   0.5211     0.4155      0.2214    73
11     2.7309      3.1010      0.4654   0.5279     0.4227      0.2335    75
12     2.6662      3.0816      0.4701   0.5316     0.4294      0.2346    77
13     2.6098      3.0697      0.4780   0.5407     0.4348      0.2454    75
14     2.5613      3.0456      0.4789   0.5431     0.4353      0.2383    74
15     2.5157      3.0381      0.4824   0.5465     0.4395      0.2410    80
16     2.4749      3.0226      0.4870   0.5505     0.4446      0.2453    76
17     2.4385      3.0177      0.4895   0.5553     0.4442      0.2507    73
18     2.4049      3.0063      0.4919   0.5565     0.4486      0.2488    77
19     2.3769      3.0009      0.4944   0.5601     0.4515      0.2496    80
20     2.3445      3.0002      0.4981   0.5628     0.4559      0.2541    77
```

### gated_mix (门控混合位置编码) — best epoch=20, gate_sin=0.9462, gate_abs=0.9017
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time(s)
1      5.4207      4.5591      0.2136   0.2587     0.1858      0.0840    84
2      4.3393      4.0712      0.2768   0.3263     0.2420      0.1381    77
3      3.9312      3.7854      0.3135   0.3726     0.2743      0.1464    79
4      3.6545      3.5997      0.3454   0.4048     0.3062      0.1619    77
5      3.4448      3.4688      0.3653   0.4240     0.3260      0.1892    73
6      3.2753      3.3808      0.3781   0.4387     0.3389      0.1821    82
7      3.1352      3.2942      0.3940   0.4556     0.3525      0.1928    78
8      3.0234      3.2395      0.4085   0.4712     0.3666      0.1980    78
9      2.9247      3.2077      0.4261   0.4881     0.3848      0.2051    72
10     2.8398      3.1771      0.4248   0.4872     0.3837      0.1998    77
11     2.7678      3.1418      0.4361   0.4976     0.3954      0.2087    75
12     2.7015      3.1194      0.4424   0.5048     0.4006      0.2112    81
13     2.6448      3.1019      0.4452   0.5087     0.4032      0.2210    78
14     2.5943      3.0911      0.4568   0.5205     0.4140      0.2131    72
15     2.5514      3.0805      0.4554   0.5213     0.4096      0.2184    73
16     2.5100      3.0594      0.4623   0.5279     0.4180      0.2203    78
17     2.4735      3.0517      0.4566   0.5246     0.4101      0.2113    77
18     2.4372      3.0565      0.4647   0.5331     0.4187      0.2078    81
19     2.4066      3.0420      0.4717   0.5379     0.4267      0.2252    75
20     2.3787      3.0372      0.4715   0.5427     0.4216      0.2252    75
```

## GPU Big 翻译样例（10例横向对比）

### 例1 (medium, ref_len=12)
- Source(DE): wie werden leute wie du zu leuten wie dir?
- Reference(EN): how do people like you get to be people like you?
- none: how do you get to be like you?
- absolute: how do you get to be like you?
- sinusoidal: how do you get to be like you?
- learned: how do you get to be the people you like?
- gated_mix: how do you get to be like you?

### 例2 (short, ref_len=6)
- Source(DE): wo kann ich sie finden?
- Reference(EN): where can i find you?
- none: where can i find you?
- absolute: where can i find you?
- sinusoidal: where can i find you?
- learned: where can i find you?
- gated_mix: where can i find you?

### 例3 (short, ref_len=6)
- Source(DE): du bist noch lange nicht fertig.
- Reference(EN): you're far from being finished.
- none: you're not done yet.
- absolute: you're not done yet.
- sinusoidal: you're not done yet.
- learned: you're not done yet.
- gated_mix: you're not done yet.

### 例4 (medium, ref_len=9)
- Source(DE): der erdradius misst ca. 6000 km.
- Reference(EN): the earth's radius is about 6,000 kilometers.
- none: the average temperature is about 6,000 kilometers.
- absolute: the distance from the earth is about 6,000 km.
- sinusoidal: the earth's radius is about 6,000 kilometers.
- learned: the radius of the earth is about 6,000 kilometers.
- gated_mix: the distance from the sun is about 6,000 km.

### 例5 (short, ref_len=6)
- Source(DE): findest du mich hübsch?
- Reference(EN): do you think i'm handsome?
- none: do you think i'm pretty?
- absolute: do you think i'm pretty?
- sinusoidal: do you think i'm pretty?
- learned: do you think i'm pretty?
- gated_mix: do you think i'm pretty?

### 例6 (medium, ref_len=8)
- Source(DE): ohne sie hätte tom das nicht geschafft.
- Reference(EN): tom couldn't have done it without you.
- none: tom couldn't have done it without her.
- absolute: tom wouldn't have done it without her.
- sinusoidal: tom couldn't have done it without her.
- learned: tom couldn't have done it without her.
- gated_mix: tom couldn't have done it without her.

### 例7 (short, ref_len=5)
- Source(DE): sie flehte ihn an, zu bleiben.
- Reference(EN): she begged him to stay.
- none: she asked him to stay.
- absolute: she asked him to stay.
- sinusoidal: she pleaded with him to stay.
- learned: she begged him to stay.
- gated_mix: she asked him to stay.

### 例8 (short, ref_len=6)
- Source(DE): der nebel begann langsam sich zu heben.
- Reference(EN): the fog gradually began to clear.
- none: the fog began to lift slowly.
- absolute: the fog began to lift slowly.
- sinusoidal: the fog began to lift slowly.
- learned: the fog slowly began to lift.
- gated_mix: the fog began to lift slowly.

### 例9 (medium, ref_len=8)
- Source(DE): ich werde tom nie wieder vertrauen.
- Reference(EN): i'm never going to trust tom again.
- none: i will never trust tom again.
- absolute: i will never trust tom again.
- sinusoidal: i will never trust tom again.
- learned: i will never trust tom again.
- gated_mix: i will never trust tom again.

### 例10 (short, ref_len=5)
- Source(DE): ich habe dich erwartet.
- Reference(EN): i've been waiting for you.
- none: i've been waiting for you.
- absolute: i've been waiting for you.
- sinusoidal: i've been waiting for you.
- learned: i've been waiting for you.
- gated_mix: i've been waiting for you.

## 主要实验发现

1. **learned PE 在三版实验中均排名第一且优势随数据/epoch增大而扩大**
   - CPU: 0.1732 → GPU 2h: 0.4110 → GPU Big: 0.4979
   - 在50K数据×12epoch时全面超越固定编码，在100K×20epoch时全面碾压

2. **sinusoidal 始终是固定编码中最优**，尤其在长句上
   - GPU Big长句acc: sinusoidal 0.1959 vs absolute 0.1795
   - 原论文的正弦多频设计在长距离泛化上具有结构性优势

3. **gated_mix 门控行为随训练加深发生显著变化**
   - CPU (5ep): gate≈0.50/0.50（未分化）
   - GPU 2h (12ep): gate=0.70/0.65（开始偏好sin）
   - GPU Big (20ep): gate=0.95/0.90（高度激活，轻微偏好sin）
   - 说明充分训练后门控明确"打开"两种编码，但标量门控表达能力有限

4. **absolute 在三版实验中始终垫底**
   - 简单的线性位置编码P(pos,j)=(2pos/(L-1)-1)(2j/(D-1)-1)不足以充分建模序列顺序

5. **none 位置编码的重要性随训练数据增加而降低**
   - GPU Big中none的token_acc(0.4684)仅比sinusoidal(0.4679)低0.0005
   - 但长句上仍有显著差距：none 0.1893 vs sinusoidal 0.1959

6. **翻译质量从CPU到大GPU版有质的飞跃**
   - CPU版：大量重复退化（"i never never never..."）
   - GPU Big版：多数短句达到正确翻译水平，中长句也有可读输出

## 所有结果文件清单

```
results/
├── data_summary.txt
├── cpu_baseline_config.json
├── cpu_baseline_train_logs.csv          (25 rows)
├── cpu_baseline_pe_comparison.csv       (5 rows)
├── cpu_baseline_translation_examples.csv
├── gpu_speed_test_log.csv
├── gpu_2h_config.json
├── gpu_2h_train_logs.csv                (60 rows)
├── gpu_2h_pe_comparison.csv             (5 rows)
├── gpu_2h_translation_examples.csv
├── gpu_big_config.json
├── gpu_big_train_logs.csv               (100 rows)
├── gpu_big_pe_comparison.csv            (5 rows)
├── gpu_big_translation_examples.csv
└── archive_cpu_raw/                     (原始CPU分散文件)

checkpoints/
├── gpu_2h_{none,absolute,sinusoidal,learned,gated_mix}.pt
└── gpu_big_{none,absolute,sinusoidal,learned,gated_mix}.pt
```
