"""data_utils.py — 德英平行语料数据管线。

职责：读取 data/deu.txt、解析列、清洗、分词、构建词表、Dataset/DataLoader。
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from functools import partial
from typing import Tuple, List, Optional


# ── 特殊符号 ──
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ═══════════════════════════════════════════════════════════
# 读取 & 清洗
# ═══════════════════════════════════════════════════════════

def read_parallel_data(filepath: str = 'data/deu.txt',
                       max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
    """读取 data/deu.txt，返回去重后的 (英语, 德语) 句对列表。

    文件格式: 英语 \t 德语 \t 版权信息
    任务方向: 德语 → 英语（源语言=de, 目标语言=en）
    """
    pairs = []
    seen = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            en = parts[0].strip()
            de = parts[1].strip()
            if not en or not de:
                continue
            key = (en.lower(), de.lower())
            if key in seen:
                continue
            seen.add(key)
            pairs.append((en, de))

    # 固定种子洗牌，保证多次运行可复现
    random.seed(42)
    random.shuffle(pairs)

    if max_samples is not None and max_samples < len(pairs):
        pairs = pairs[:max_samples]

    return pairs


def tokenize(text: str) -> List[str]:
    """简单空格分词，按空白字符切分并转小写。"""
    return text.lower().split()


# ═══════════════════════════════════════════════════════════
# 词表构建 & 编码
# ═══════════════════════════════════════════════════════════

def build_vocab(sentences: List[List[str]], max_size: int = 30000
                ) -> Tuple[dict, dict]:
    """构建词表，返回 (token2idx, idx2token)。

    token2idx 中特殊符号索引固定: <pad>=0 <bos>=1 <eos>=2 <unk>=3
    """
    counter = Counter()
    for tokens in sentences:
        counter.update(tokens)

    token2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    idx2token = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}

    # 按频率降序取 most_common，为特殊符号留空间
    vocab_limit = max_size - len(SPECIAL_TOKENS)
    for token, _ in counter.most_common(vocab_limit):
        idx = len(token2idx)
        token2idx[token] = idx
        idx2token[idx] = token

    return token2idx, idx2token


def encode(tokens: List[str], token2idx: dict, max_len: int,
           add_bos: bool = True, add_eos: bool = True) -> List[int]:
    """将 token 列表转为 idx 序列。

    - add_bos=True: 开头加 <bos>
    - add_eos=True: 末尾加 <eos>
    - 截断到 max_len
    - 未知词替换为 <unk>
    """
    ids = [token2idx.get(t, UNK_IDX) for t in tokens]

    # 计算可用空间: max_len - (bos? 1:0) - (eos? 1:0)
    avail = max_len
    if add_bos:
        avail -= 1
    if add_eos:
        avail -= 1
    ids = ids[:avail]

    if add_bos:
        ids = [BOS_IDX] + ids
    if add_eos:
        ids = ids + [EOS_IDX]
    return ids


# ═══════════════════════════════════════════════════════════
# PyTorch Dataset & Collate
# ═══════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """德英翻译 PyTorch Dataset。

    每个样本是 (src_ids, tgt_ids) 的 LongTensor。
    src 格式: [<bos>] + tokens + [<eos>]
    tgt 格式: [<bos>] + tokens + [<eos>]
    """

    def __init__(self, src_ids: List[List[int]], tgt_ids: List[List[int]]):
        assert len(src_ids) == len(tgt_ids)
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.src_ids[idx], dtype=torch.long),
                torch.tensor(self.tgt_ids[idx], dtype=torch.long))


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]],
               pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """将变长序列 batch 填充到相同长度并堆叠。

    返回:
        src_batch: (batch, src_max_len)
        tgt_batch: (batch, tgt_max_len)
    """
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True,
                                                  padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True,
                                                  padding_value=pad_idx)
    return src_padded, tgt_padded


# ═══════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════

def create_dataloaders(
    pairs: Optional[List[Tuple[str, str]]] = None,
    data_path: str = 'data/deu.txt',
    max_samples: Optional[int] = None,
    src_max_len: int = 30,
    tgt_max_len: int = 30,
    src_max_vocab: int = 30000,
    tgt_max_vocab: int = 30000,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[dict, dict, dict, DataLoader, DataLoader, DataLoader]:
    """完整的 DataLoader 构建管线。

    流程: 读取 → 分词 → 过滤超长句 → 建词表 → 编码 → 划分 → DataLoader

    Args:
        pairs: 已读取的句对列表，为 None 则从 data_path 读取
        data_path: 数据文件路径
        max_samples: 限制使用的样本数，None 表示全用
        src_max_len: 源语言(德语)最大序列长度
        tgt_max_len: 目标语言(英语)最大序列长度
        src_max_vocab: 源语言词表最大大小
        tgt_max_vocab: 目标语言词表最大大小
        batch_size: batch 大小
        train_ratio: 训练集比例
        valid_ratio: 验证集比例（测试集 = 1 - train - valid）
        num_workers: DataLoader 工作进程数
        seed: 随机种子

    Returns:
        src_token2idx, tgt_token2idx, tgt_idx2token,
        train_loader, valid_loader, test_loader
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. 读取
    if pairs is None:
        pairs = read_parallel_data(data_path, max_samples=max_samples)
    elif max_samples is not None and max_samples < len(pairs):
        pairs = pairs[:max_samples]

    # 2. 分词
    en_sents = [tokenize(en) for en, _ in pairs]
    de_sents = [tokenize(de) for _, de in pairs]

    # 3. 过滤超长句子（在没有 <bos>/<eos> 之前先按 token 数量过滤）
    keep = [i for i in range(len(pairs))
            if len(de_sents[i]) <= src_max_len - 2
            and len(en_sents[i]) <= tgt_max_len - 2]

    de_sents = [de_sents[i] for i in keep]
    en_sents = [en_sents[i] for i in keep]
    pairs = [pairs[i] for i in keep]

    # 4. 建词表
    src_token2idx, src_idx2token = build_vocab(de_sents, src_max_vocab)
    tgt_token2idx, tgt_idx2token = build_vocab(en_sents, tgt_max_vocab)

    # 5. 编码
    src_ids = [encode(s, src_token2idx, src_max_len, add_bos=True, add_eos=True)
               for s in de_sents]
    tgt_ids = [encode(s, tgt_token2idx, tgt_max_len, add_bos=True, add_eos=True)
               for s in en_sents]

    # 6. 划分 train / valid / test
    n = len(src_ids)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    indices = list(range(n))
    # pairs 已经在上游 shuffle 过，这里不重复 shuffle

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    train_src = [src_ids[i] for i in train_idx]
    train_tgt = [tgt_ids[i] for i in train_idx]
    valid_src = [src_ids[i] for i in valid_idx]
    valid_tgt = [tgt_ids[i] for i in valid_idx]
    test_src = [src_ids[i] for i in test_idx]
    test_tgt = [tgt_ids[i] for i in test_idx]

    # 7. Dataset & DataLoader
    pad_idx = PAD_IDX
    my_collate = partial(collate_fn, pad_idx=pad_idx)

    train_ds = TranslationDataset(train_src, train_tgt)
    valid_ds = TranslationDataset(valid_src, valid_tgt)
    test_ds = TranslationDataset(test_src, test_tgt)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=my_collate, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=my_collate, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=my_collate, num_workers=num_workers)

    return (src_token2idx, tgt_token2idx, tgt_idx2token,
            train_loader, valid_loader, test_loader)


# ═══════════════════════════════════════════════════════════
# 数据摘要（供 report 使用）
# ═══════════════════════════════════════════════════════════

def get_data_summary(pairs: List[Tuple[str, str]],
                     src_token2idx: Optional[dict] = None,
                     tgt_token2idx: Optional[dict] = None,
                     train_size: int = 0,
                     valid_size: int = 0,
                     test_size: int = 0) -> str:
    """返回格式化的数据摘要字符串。"""
    en_lengths = [len(en.split()) for en, _ in pairs]
    de_lengths = [len(de.split()) for _, de in pairs]

    lines = []
    lines.append(f'总样本数: {len(pairs)}')
    lines.append(f'训练集: {train_size}, 验证集: {valid_size}, 测试集: {test_size}')
    lines.append('')
    lines.append(f'英语句长 — min: {min(en_lengths)}, max: {max(en_lengths)}, '
                 f'mean: {sum(en_lengths)/len(en_lengths):.1f}')
    lines.append(f'德语句长 — min: {min(de_lengths)}, max: {max(de_lengths)}, '
                 f'mean: {sum(de_lengths)/len(de_lengths):.1f}')
    if src_token2idx:
        lines.append(f'源语言(德语)词表大小: {len(src_token2idx)}')
    if tgt_token2idx:
        lines.append(f'目标语言(英语)词表大小: {len(tgt_token2idx)}')
    return '\n'.join(lines)
