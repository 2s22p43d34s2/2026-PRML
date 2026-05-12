"""experiment.py — 训练、验证、测试、Greedy Decoding、五组位置编码实验调度。"""

import os
import csv
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.data_utils import (PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX,
                             create_dataloaders, read_parallel_data)
from src.transformer_model import (TransformerSeq2Seq,
                                   _make_pad_mask, _make_subsequent_mask)


# ═══════════════════════════════════════════════════════════
# 训练 / 验证
# ═══════════════════════════════════════════════════════════

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer,
                criterion, pad_idx: int, device: torch.device,
                clip: float = 1.0) -> float:
    """训练一个 epoch，返回平均 loss（忽略 <pad>）。"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # Teacher forcing: decoder 输入 tgt[:, :-1]，预测 tgt[:, 1:]
        src_mask = _make_pad_mask(src, pad_idx)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_pad_mask = _make_pad_mask(tgt_input, pad_idx)
        tgt_sub_mask = _make_subsequent_mask(tgt_input.size(1), device)
        tgt_mask = tgt_pad_mask + tgt_sub_mask

        enc_out = model.encoder(src, src_mask)
        logits = model.decoder(tgt_input, enc_out, tgt_mask, src_mask)

        loss = criterion(logits.reshape(-1, logits.size(-1)),
                         tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        n_tokens = (tgt_output != pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, criterion,
             pad_idx: int, device: torch.device) -> float:
    """验证，返回平均 loss。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        src_mask = _make_pad_mask(src, pad_idx)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_pad_mask = _make_pad_mask(tgt_input, pad_idx)
        tgt_sub_mask = _make_subsequent_mask(tgt_input.size(1), device)
        tgt_mask = tgt_pad_mask + tgt_sub_mask

        enc_out = model.encoder(src, src_mask)
        logits = model.decoder(tgt_input, enc_out, tgt_mask, src_mask)

        loss = criterion(logits.reshape(-1, logits.size(-1)),
                         tgt_output.reshape(-1))

        n_tokens = (tgt_output != pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


# ═══════════════════════════════════════════════════════════
# 推理 & 评价
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(model: TransformerSeq2Seq, src: torch.Tensor,
                  max_len: int, bos_idx: int, eos_idx: int, pad_idx: int,
                  device: torch.device) -> torch.Tensor:
    """贪心解码：逐 token 生成直到 <eos> 或达到 max_len。

    Returns:
        decoded token indices (batch, seq_len)
    """
    model.eval()
    batch = src.size(0)

    src_mask = _make_pad_mask(src, pad_idx)
    enc_out = model.encoder(src, src_mask)

    ys = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_pad_mask = _make_pad_mask(ys, pad_idx)
        tgt_sub_mask = _make_subsequent_mask(ys.size(1), device)
        tgt_mask = tgt_pad_mask + tgt_sub_mask

        logits = model.decoder(ys, enc_out, tgt_mask, src_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if (next_token == eos_idx).all():
            break

    return ys


def compute_token_accuracy(model: TransformerSeq2Seq, dataloader: DataLoader,
                           pad_idx: int, bos_idx: int, eos_idx: int,
                           max_len: int, device: torch.device
                           ) -> Dict[str, float]:
    """在给定 DataLoader 上计算 token-level accuracy（忽略 <pad>）。

    Returns:
        {'token_acc': float, 'short_acc': float, 'medium_acc': float, 'long_acc': float}
    """
    model.eval()
    correct = 0
    total = 0
    bucket_correct = defaultdict(int)
    bucket_total = defaultdict(int)

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        preds = greedy_decode(model, src, max_len, bos_idx, eos_idx, pad_idx, device)

        for i in range(src.size(0)):
            ref = tgt[i].tolist()
            pred = preds[i].tolist()

            # 去掉 <bos>，截取到第一个 <eos>
            ref_tokens = []
            for t in ref[1:]:
                if t == eos_idx:
                    break
                ref_tokens.append(t)

            pred_tokens = []
            for t in pred[1:]:
                if t == eos_idx:
                    break
                pred_tokens.append(t)

            # 对齐长度
            min_len = min(len(ref_tokens), len(pred_tokens))
            for j in range(min_len):
                if pred_tokens[j] == ref_tokens[j]:
                    correct += 1
                total += 1

            # 句长分桶：按参考译文长度
            ref_len = len(ref_tokens)
            bucket_key = _get_bucket(ref_len)
            for j in range(min_len):
                if pred_tokens[j] == ref_tokens[j]:
                    bucket_correct[bucket_key] += 1
                bucket_total[bucket_key] += 1

    overall = correct / max(total, 1)
    result = {'token_acc': overall}
    for bucket in ['short', 'medium', 'long']:
        bt = bucket_total.get(bucket, 0)
        bc = bucket_correct.get(bucket, 0)
        result[f'{bucket}_acc'] = bc / max(bt, 1)
    return result


def _get_bucket(length: int) -> str:
    if length <= 6:
        return 'short'
    elif length <= 12:
        return 'medium'
    else:
        return 'long'


def compute_simple_bleu(references: List[List[str]],
                        hypotheses: List[List[str]]) -> float:
    """简化 BLEU: 1-4 gram precision 几何平均 × brevity penalty。"""
    if not references or not hypotheses:
        return 0.0

    ref_lens = [len(r) for r in references]
    hyp_lens = [len(h) for h in hypotheses]
    total_ref = sum(ref_lens)
    total_hyp = sum(hyp_lens)

    bp = 1.0 if total_hyp >= total_ref else math.exp(1 - total_ref / max(total_hyp, 1))

    precisions = []
    for n in range(1, 5):
        match = 0
        total_n = 0
        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = set(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            hyp_ngrams = [tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)]
            match += sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
            total_n += max(len(hyp_ngrams), 1)
        precisions.append(match / max(total_n, 1))

    geo_mean = 1.0
    for p in precisions:
        geo_mean *= p
    geo_mean = geo_mean ** 0.25

    return bp * geo_mean


# ═══════════════════════════════════════════════════════════
# 单组实验运行
# ═══════════════════════════════════════════════════════════

def run_single_experiment(
    pe_type: str,
    data_config: dict,
    model_config: dict,
    train_config: dict,
    device: torch.device = None,
) -> Dict:
    """运行单组位置编码实验。

    Args:
        pe_type: 'sinusoidal' (stage 4) 或后续 'none'|'absolute'|'learned'|'gated_mix'
        data_config: 包含 data_path, max_samples, src_max_len, tgt_max_len,
                     src_max_vocab, tgt_max_vocab, batch_size
        model_config: 包含 d_model, n_heads, num_layers, dim_feedforward, max_len, dropout
        train_config: 包含 lr, n_epochs, clip, seed
        device: torch device

    Returns:
        包含 train_loss_history, valid_loss_history, token_acc_dict,
        gate_sin, gate_abs, translations 的结果字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\n{"="*60}')
    print(f'  Experiment: pe_type = {pe_type}')
    print(f'{"="*60}')

    # ── 数据 ──
    print('Building dataloaders...')
    (src_t2i, tgt_t2i, tgt_i2t,
     train_loader, valid_loader, test_loader) = create_dataloaders(
        data_path=data_config.get('data_path', 'data/deu.txt'),
        max_samples=data_config.get('max_samples'),
        src_max_len=data_config['src_max_len'],
        tgt_max_len=data_config['tgt_max_len'],
        src_max_vocab=data_config.get('src_max_vocab', 30000),
        tgt_max_vocab=data_config.get('tgt_max_vocab', 30000),
        batch_size=data_config['batch_size'],
        train_ratio=data_config.get('train_ratio', 0.8),
        valid_ratio=data_config.get('valid_ratio', 0.1),
        num_workers=data_config.get('num_workers', 0),
        seed=train_config.get('seed', 42),
    )
    print(f'  Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, '
          f'Test: {len(test_loader.dataset)}')
    print(f'  Src vocab: {len(src_t2i)}, Tgt vocab: {len(tgt_t2i)}')

    # ── 模型 ──
    print('Building model...')
    model = TransformerSeq2Seq(
        src_vocab_size=len(src_t2i),
        tgt_vocab_size=len(tgt_t2i),
        max_len=model_config['max_len'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        num_layers=model_config['num_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        pe_type=pe_type,
        dropout=model_config.get('dropout', 0.1),
        pad_idx=PAD_IDX,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total_params:,}, Trainable: {trainable_params:,}')

    # ── 优化器 & Loss ──
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['lr'],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        label_smoothing=train_config.get('label_smoothing', 0.0),
    )

    # ── 训练 ──
    n_epochs = train_config['n_epochs']
    train_losses = []
    valid_losses = []
    epoch_times = []
    epoch_accs = []
    best_valid_loss = float('inf')
    best_state_dict = None
    ckpt_prefix = train_config.get('ckpt_prefix', 'transformer')

    for epoch in range(1, n_epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 PAD_IDX, device, train_config.get('clip', 1.0))
        valid_loss = validate(model, valid_loader, criterion, PAD_IDX, device)

        # per-epoch token accuracy on validation set
        acc = compute_token_accuracy(
            model, valid_loader, PAD_IDX, BOS_IDX, EOS_IDX,
            model_config['max_len'], device,
        )
        elapsed = time.time() - start

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        epoch_times.append(elapsed)
        epoch_accs.append(acc)

        status = ''
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            status = ' *best'
        print(f'  Epoch {epoch:3d}/{n_epochs} | '
              f'train_loss: {train_loss:.4f} | valid_loss: {valid_loss:.4f} | '
              f'tok_acc: {acc["token_acc"]:.4f} | time: {elapsed:.1f}s{status}')

    # restore best model for final evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ── 测试（使用最佳模型）──
    print(f'Evaluating on test set (best epoch: {best_epoch})...')
    token_acc = compute_token_accuracy(
        model, test_loader, PAD_IDX, BOS_IDX, EOS_IDX,
        model_config['max_len'], device,
    )
    print(f'  Token accuracy: {token_acc["token_acc"]:.4f}')
    print(f'  Short: {token_acc["short_acc"]:.4f}, '
          f'Medium: {token_acc["medium_acc"]:.4f}, '
          f'Long: {token_acc["long_acc"]:.4f}')

    # ── 翻译样例 ──
    print('Generating translations...')
    translations = _generate_translations(
        model, test_loader, src_t2i, tgt_t2i, tgt_i2t,
        model_config['max_len'], PAD_IDX, BOS_IDX, EOS_IDX, device, limit=10,
    )

    # ── Gate values ──
    gate_sin, gate_abs = None, None
    pos_enc = model.get_pos_encoding()
    gates = pos_enc.get_gate_values()
    if gates is not None:
        gate_sin, gate_abs = gates
        print(f'  Gate values — sin: {gate_sin:.4f}, abs: {gate_abs:.4f}')

    # ── 保存最佳 checkpoint ──
    ckpt_path = os.path.join('checkpoints', f'{ckpt_prefix}_{pe_type}.pt')
    torch.save(best_state_dict if best_state_dict else model.state_dict(), ckpt_path)
    print(f'  Saved best checkpoint: {ckpt_path}')

    return {
        'pe_type': pe_type,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'epoch_times': epoch_times,
        'epoch_accs': epoch_accs,
        'best_epoch': best_epoch,
        'token_acc': token_acc['token_acc'],
        'short_acc': token_acc['short_acc'],
        'medium_acc': token_acc['medium_acc'],
        'long_acc': token_acc['long_acc'],
        'gate_sin': gate_sin,
        'gate_abs': gate_abs,
        'translations': translations,
        'model': model,
        'total_params': total_params,
        'src_token2idx': src_t2i,
        'tgt_token2idx': tgt_t2i,
        'tgt_idx2token': tgt_i2t,
        'src_vocab_size': len(src_t2i),
        'tgt_vocab_size': len(tgt_t2i),
        'train_size': len(train_loader.dataset),
        'valid_size': len(valid_loader.dataset),
        'test_size': len(test_loader.dataset),
    }


def _generate_translations(model, dataloader, src_t2i, tgt_t2i, tgt_i2t,
                           max_len, pad_idx, bos_idx, eos_idx, device,
                           limit=10) -> List[Dict]:
    """生成翻译样例。"""
    src_i2t = {v: k for k, v in src_t2i.items()}

    examples = []
    model.eval()
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            preds = greedy_decode(model, src, max_len, bos_idx, eos_idx, pad_idx, device)

            for i in range(src.size(0)):
                if len(examples) >= limit:
                    break

                src_tokens = [src_i2t.get(idx, '<unk>') for idx in src[i].tolist()
                              if idx not in (pad_idx, bos_idx, eos_idx)]
                ref_tokens = [tgt_i2t.get(idx, '<unk>') for idx in tgt[i].tolist()
                              if idx not in (pad_idx, bos_idx, eos_idx)]
                pred_ids = preds[i].tolist()
                pred_tokens = []
                for idx in pred_ids[1:]:  # 跳过 <bos>
                    if idx == eos_idx:
                        break
                    pred_tokens.append(tgt_i2t.get(idx, '<unk>'))

                examples.append({
                    'source': ' '.join(src_tokens),
                    'reference': ' '.join(ref_tokens),
                    'prediction': ' '.join(pred_tokens),
                })
            if len(examples) >= limit:
                break
    return examples


# ═══════════════════════════════════════════════════════════
# 日志保存辅助
# ═══════════════════════════════════════════════════════════

def save_train_log_csv(pe_type: str, train_losses: List[float],
                       valid_losses: List[float],
                       out_dir: str = 'results'):
    """保存训练日志为 CSV。"""
    path = os.path.join(out_dir, f'train_log_{pe_type}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'valid_loss'])
        for i, (tl, vl) in enumerate(zip(train_losses, valid_losses), 1):
            writer.writerow([i, f'{tl:.6f}', f'{vl:.6f}'])
    print(f'  Saved: {path}')


def save_translation_examples(pe_type: str, translations: List[Dict],
                              out_dir: str = 'results'):
    """保存翻译样例为 TXT。"""
    path = os.path.join(out_dir, f'translation_examples_{pe_type}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'=== Translation Examples (pe_type={pe_type}) ===\n\n')
        for i, ex in enumerate(translations):
            f.write(f'--- Example {i + 1} ---\n')
            f.write(f'Source (DE):     {ex["source"]}\n')
            f.write(f'Reference (EN):  {ex["reference"]}\n')
            f.write(f'Prediction (EN): {ex["prediction"]}\n\n')
    print(f'  Saved: {path}')


def save_checkpoint(model: nn.Module, pe_type: str,
                    out_dir: str = 'checkpoints'):
    """保存模型 checkpoint。"""
    path = os.path.join(out_dir, f'transformer_{pe_type}.pt')
    torch.save(model.state_dict(), path)
    print(f'  Saved: {path}')


def run_all_experiments(config: dict):
    """依次运行5组 pe_type 实验。"""
    raise NotImplementedError  # 阶段6实现


def save_comparison_csv(results_list: List[Dict], out_path: str = 'results/pe_comparison.csv'):
    """将多组实验结果汇总保存为 pe_comparison.csv。"""
    raise NotImplementedError  # 阶段6实现
