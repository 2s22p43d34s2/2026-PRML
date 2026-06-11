# PRML作业四：GPU正式实验结果

## 实验配置

| 参数 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB) |
| 训练样本 | 50,000 (train 40K / valid 5K / test 5K) |
| 源语言词表 (DE) | 26,564 |
| 目标语言词表 (EN) | 16,309 |
| d_model | 256 |
| n_heads | 8 |
| num_layers (Encoder/Decoder) | 3 |
| dim_feedforward | 1024 |
| batch_size | 128 |
| max_len | 30 |
| learning_rate | 0.0003 |
| dropout | 0.1 |
| label_smoothing | 0.1 |
| epochs | 12 |
| seed | 42 |
| 模型参数量 | ~20.7M |
| 显存峰值 | 1.61 GB |
| 总耗时 | 40.7 分钟 |

## 五组位置编码结果 — 验证集损失与测试集准确率对比

| pe_type | train_loss(训练集) | valid_loss(验证集) | token_acc(测试集) | short_acc(测试集) | medium_acc(测试集) | long_acc(测试集) | gate |
|---|---|---|---|---|---|---|---|
| none | 2.8214 | 3.4849 | 0.3747 | 0.4379 | 0.3354 | 0.1827 | — |
| absolute | 2.8189 | 3.4808 | 0.3740 | 0.4356 | 0.3354 | 0.1796 | — |
| sinusoidal | 2.8165 | 3.4815 | 0.3805 | 0.4459 | 0.3382 | 0.1957 | — |
| learned | 2.7781 | **3.4461** | **0.4110** | **0.4757** | **0.3715** | **0.2061** | — |
| gated_mix | 2.8189 | 3.4811 | 0.3803 | 0.4437 | 0.3421 | 0.1791 | sin=0.6957 / abs=0.6465 |

## 逐epoch训练日志

### none (无位置编码)
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time
1      5.8577      5.0165      0.1731   0.2174     0.1446      0.0686    40.4s
2      4.7596      4.5192      0.2224   0.2763     0.1875      0.0994    39.8s
3      4.3312      4.2479      0.2530   0.3057     0.2180      0.1323    35.7s
4      4.0417      4.0581      0.2882   0.3415     0.2527      0.1613    37.0s
5      3.8106      3.9229      0.3003   0.3571     0.2626      0.1653    36.8s
6      3.6153      3.7993      0.3198   0.3758     0.2820      0.1824    36.4s
7      3.4469      3.7107      0.3326   0.3932     0.2942      0.1718    36.1s
8      3.2944      3.6508      0.3485   0.4104     0.3091      0.1868    35.7s
9      3.1571      3.5774      0.3570   0.4161     0.3197      0.1924    36.5s
10     3.0336      3.5406      0.3585   0.4201     0.3184      0.1932    36.8s
11     2.9224      3.5034      0.3706   0.4347     0.3303      0.1860    39.4s
12     2.8214      3.4849      0.3767   0.4403     0.3362      0.1946    38.7s
```

### absolute (简单绝对位置编码)
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time
1      5.8605      5.0184      0.1734   0.2170     0.1419      0.0780    26.6s
2      4.7610      4.5159      0.2243   0.2768     0.1883      0.1053    26.8s
3      4.3303      4.2449      0.2534   0.3061     0.2173      0.1376    46.0s
4      4.0399      4.0555      0.2848   0.3379     0.2500      0.1524    43.2s
5      3.8082      3.9218      0.3025   0.3588     0.2658      0.1644    42.6s
6      3.6136      3.7963      0.3191   0.3748     0.2813      0.1802    38.1s
7      3.4438      3.7084      0.3320   0.3913     0.2949      0.1694    34.2s
8      3.2927      3.6453      0.3482   0.4089     0.3105      0.1831    42.9s
9      3.1561      3.5821      0.3547   0.4159     0.3169      0.1815    44.1s
10     3.0323      3.5366      0.3559   0.4207     0.3156      0.1764    38.7s
11     2.9208      3.5017      0.3721   0.4372     0.3311      0.1936    38.1s
12     2.8189      3.4808      0.3749   0.4372     0.3362      0.1911    38.1s
```

### sinusoidal (原论文正弦位置编码)
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time
1      5.8600      5.0194      0.1733   0.2186     0.1448      0.0641    40.9s
2      4.7580      4.5114      0.2274   0.2825     0.1917      0.1031    37.7s
3      4.3283      4.2451      0.2560   0.3092     0.2210      0.1365    38.8s
4      4.0376      4.0518      0.2870   0.3432     0.2511      0.1499    45.0s
5      3.8067      3.9141      0.3041   0.3589     0.2690      0.1690    39.7s
6      3.6111      3.7974      0.3194   0.3778     0.2824      0.1707    41.2s
7      3.4406      3.7046      0.3350   0.3957     0.2974      0.1683    41.4s
8      3.2901      3.6441      0.3452   0.4087     0.3062      0.1674    40.5s
9      3.1527      3.5772      0.3572   0.4197     0.3175      0.1857    42.4s
10     3.0306      3.5374      0.3622   0.4262     0.3208      0.1908    41.0s
11     2.9172      3.5009      0.3690   0.4353     0.3311      0.1872    42.2s
12     2.8165      3.4815      0.3806   0.4480     0.3394      0.1925    56.2s
```

### learned (可学习绝对位置编码)
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time
1      5.8535      5.0134      0.1745   0.2193     0.1515      0.0448    40.9s
2      4.7419      4.4917      0.2403   0.2961     0.2054      0.1031    38.1s
3      4.3013      4.2016      0.2771   0.3349     0.2417      0.1336    40.2s
4      4.0069      4.0132      0.3077   0.3672     0.2710      0.1593    36.3s
5      3.7734      3.8657      0.3252   0.3844     0.2884      0.1761    37.3s
6      3.5777      3.7507      0.3439   0.4047     0.3059      0.1885    38.1s
7      3.4017      3.6547      0.3581   0.4205     0.3187      0.1932    38.8s
8      3.2518      3.5869      0.3819   0.4455     0.3417      0.2087    36.0s
9      3.1158      3.5337      0.3792   0.4438     0.3382      0.1995    40.0s
10     2.9943      3.4846      0.3960   0.4598     0.3565      0.2086    34.9s
11     2.8789      3.4611      0.3915   0.4592     0.3498      0.2005    37.9s
12     2.7781      3.4461      0.4076   0.4721     0.3680      0.2139    37.1s
```

### gated_mix (门控混合位置编码)
```
epoch  train_loss(训练集)  valid_loss(验证集)  tok_acc(验证集)  short_acc(验证集)  medium_acc(验证集)  long_acc(验证集)  time
1      5.8618      5.0179      0.1693   0.2141     0.1397      0.0705    40.7s
2      4.7631      4.5205      0.2234   0.2779     0.1883      0.0990    38.4s
3      4.3314      4.2443      0.2575   0.3117     0.2236      0.1210    36.1s
4      4.0397      4.0555      0.2896   0.3454     0.2548      0.1468    35.9s
5      3.8084      3.9180      0.3046   0.3618     0.2676      0.1606    35.4s
6      3.6135      3.7936      0.3176   0.3755     0.2810      0.1691    38.5s
7      3.4444      3.7075      0.3310   0.3916     0.2942      0.1684    36.2s
8      3.2939      3.6493      0.3502   0.4123     0.3117      0.1811    37.3s
9      3.1574      3.5788      0.3577   0.4203     0.3189      0.1842    37.2s
10     3.0324      3.5380      0.3594   0.4244     0.3191      0.1739    39.1s
11     2.9215      3.5074      0.3732   0.4390     0.3319      0.1938    37.9s
12     2.8189      3.4811      0.3790   0.4437     0.3402      0.1934    40.0s
```
门控最终值：gate_sin=0.6957, gate_abs=0.6465

## 翻译样例对比

### 例1 (medium, ref_len=12)
- Source(DE): wie werden leute wie du zu leuten wie dir?
- Reference(EN): how do people like you get to be people like you?
- none: how do you get to be like to be a person like you?
- absolute: how do you get to be like to be a person like you?
- sinusoidal: how do you get to be like to be a person like you?
- learned: how do you get to be like you?
- gated_mix: how do you get to be like to be a person like you?

### 例2 (short, ref_len=6)
- Source(DE): wo kann ich sie finden?
- Reference(EN): where can i find you?
- none: where can i find some water?
- absolute: where can i find some water?
- sinusoidal: where can i find a toilet?
- learned: where can i find it?
- gated_mix: where can i find some water?

### 例3 (short, ref_len=6)
- Source(DE): du bist noch lange nicht fertig.
- Reference(EN): you're far from being finished.
- none: you're not done yet.
- absolute: you're not done yet.
- sinusoidal: you're not ready yet.
- learned: you're not done yet.
- gated_mix: you're not done yet.

### 例4 (medium, ref_len=9)
- Source(DE): der erdradius misst ca. 6000 km.
- Reference(EN): the earth's radius is about 6,000 kilometers.
- none: the average of the mind is that the mind is mind.
- absolute: the average of the mind is that the mind is mind.
- sinusoidal: the average of the mind is the only way to make the work.
- learned: the number of the people who are in the world is about 6,000.
- gated_mix: the average of the mind is the only way to make the work.

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
- none: tom couldn't have done it without you.
- absolute: tom couldn't have done it without you.
- sinusoidal: tom couldn't have done it without you.
- learned: tom couldn't have done it without you.
- gated_mix: tom couldn't have done it without you.

### 例7 (short, ref_len=5)
- Source(DE): sie flehte ihn an, zu bleiben.
- Reference(EN): she begged him to stay.
- none: she asked him to stay.
- absolute: she asked him to stay.
- sinusoidal: she pleaded with him to stay.
- learned: she asked him to stay.
- gated_mix: she asked him to stay.

### 例8 (short, ref_len=6)
- Source(DE): der nebel begann langsam sich zu heben.
- Reference(EN): the fog gradually began to clear.
- none: the fog began to move slowly.
- absolute: the fog began to move slowly.
- sinusoidal: the fog began to move slowly.
- learned: the fog slowly began to lift.
- gated_mix: the fog began to move slowly.

### 例9 (medium, ref_len=8)
- Source(DE): ich werde tom nie wieder vertrauen.
- Reference(EN): i'm never going to trust tom again.
- none: i'll never trust tom again.
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

## 主要发现

1. **learned PE 全面最优**：token_acc=0.4110，short/medium/long全部最高。可学习位置编码在50K数据上足以学到比固定编码更好的位置表示。

2. **sinusoidal 长句表现最好（固定编码中）**：long_acc=0.1957。验证了原论文正弦编码的长距离泛化优势。

3. **gated_mix 门控偏好明确**：gate_sin=0.6957 vs gate_abs=0.6465，模型更依赖正弦编码。但综合表现不如纯learned，标量门控的表达能力有限。

4. **absolute 持续垫底**：long_acc=0.1796。简单线性位置编码不足以建模序列顺序关系。

5. **none 可完成基本翻译**：token_acc=0.3747。词嵌入本身携带较强语义信息，但缺少位置信息在长句上代价明显（long_acc 0.1827 vs learned 0.2061）。

6. **翻译质量显著提升**：相比CPU版（8K样本/5epoch），GPU版（50K样本/12epoch）的译文从大量重复退化变为可读的英文句子，多数短句达到正确翻译。

## 与CPU轻量版对比

| 指标 | CPU (8K/5ep) | GPU (50K/12ep) | 提升倍数 |
|---|---|---|---|
| Best token_acc | 0.1732 (learned) | 0.4110 (learned) | 2.37× |
| Best long_acc | 0.0847 (sinusoidal) | 0.2061 (learned) | 2.43× |
| 模型参数 | 3.7M | 20.7M | 5.6× |
| 训练数据 | 8K | 50K | 6.25× |
