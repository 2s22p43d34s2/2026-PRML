"""transformer_model.py — PyTorch 手写 Transformer Encoder-Decoder。

包含：Embedding、五种位置编码(none/absolute/sinusoidal/learned/gated_mix)、
Multi-Head Attention、FFN、Encoder、Decoder、Seq2Seq 模型、Mask 和 Greedy Decoding。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════
# 位置编码
# ═══════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """统一位置编码模块，支持五种 pe_type。

    pe_type:
        'none'       — 不添加位置编码
        'absolute'   — P(pos,j) = (2*pos/(max_len-1)-1) * (2*j/(d_model-1)-1)
        'sinusoidal' — 原论文正弦/余弦位置编码
        'learned'    — nn.Embedding 可学习位置编码
        'gated_mix'  — sigmoid(alpha)*P_sin + sigmoid(beta)*P_abs，可学习门控
    """

    def __init__(self, max_len: int, d_model: int, pe_type: str = 'sinusoidal',
                 dropout: float = 0.1):
        super().__init__()
        self.pe_type = pe_type
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        if pe_type == 'none':
            self.pe = None

        elif pe_type == 'absolute':
            # 固定简单绝对位置编码
            pe = torch.zeros(max_len, d_model)
            for pos in range(max_len):
                for j in range(d_model):
                    pe[pos, j] = (2.0 * pos / max(max_len - 1, 1) - 1.0) * \
                                 (2.0 * j / max(d_model - 1, 1) - 1.0)
            self.register_buffer('pe', pe)

        elif pe_type == 'sinusoidal':
            # 原论文正弦位置编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) *
                (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        elif pe_type == 'learned':
            self.pe_embed = nn.Embedding(max_len, d_model)

        elif pe_type == 'gated_mix':
            # 预计算两个基础位置编码
            pe_sin = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) *
                (-math.log(10000.0) / d_model)
            )
            pe_sin[:, 0::2] = torch.sin(position * div_term)
            pe_sin[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe_sin', pe_sin)

            pe_abs = torch.zeros(max_len, d_model)
            for pos in range(max_len):
                for j in range(d_model):
                    pe_abs[pos, j] = (2.0 * pos / max(max_len - 1, 1) - 1.0) * \
                                     (2.0 * j / max(d_model - 1, 1) - 1.0)
            self.register_buffer('pe_abs', pe_abs)

            # 标量门控参数
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.beta = nn.Parameter(torch.tensor(0.0))

        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)，返回加上位置编码的结果。"""
        seq_len = x.size(1)

        if self.pe_type == 'none':
            return self.dropout(x)

        elif self.pe_type == 'learned':
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pe = self.pe_embed(positions)
            return self.dropout(x + pe)

        elif self.pe_type == 'gated_mix':
            gate_sin = torch.sigmoid(self.alpha)
            gate_abs = torch.sigmoid(self.beta)
            pe = gate_sin * self.pe_sin[:seq_len, :] + gate_abs * self.pe_abs[:seq_len, :]
            return self.dropout(x + pe)

        else:
            # absolute, sinusoidal — pe is a registered buffer
            return self.dropout(x + self.pe[:seq_len, :])

    def get_gate_values(self) -> Optional[Tuple[float, float]]:
        """返回 (gate_sin, gate_abs)，仅 gated_mix 有效；其它类型返回 None。"""
        if self.pe_type == 'gated_mix':
            return (torch.sigmoid(self.alpha).item(), torch.sigmoid(self.beta).item())
        return None


# ═══════════════════════════════════════════════════════════
# 多头注意力
# ═══════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Attention。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)"""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)"""
        batch, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scaled dot-product attention.

        mask: (batch, 1, seq_len, seq_len) 或 (batch, 1, 1, seq_len)，
              被屏蔽位置填 -inf 或一个很大的负数。
        """
        batch = query.size(0)

        Q = self._split_heads(self.w_q(query))
        K = self._split_heads(self.w_k(key))
        V = self._split_heads(self.w_v(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = self._merge_heads(context)
        return self.w_o(context)


# ═══════════════════════════════════════════════════════════
# 前馈网络
# ═══════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """Position-wise FFN: Linear → ReLU → Dropout → Linear → Dropout"""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ═══════════════════════════════════════════════════════════
# Encoder / Decoder Layers
# ═══════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """单层 Transformer Encoder：Self-Attention + FFN，含残差连接和 LayerNorm（post-norm）。"""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # Self-attention sublayer
        attn_out = self.self_attn(src, src, src, src_mask)
        x = self.norm1(src + self.dropout1(attn_out))
        # FFN sublayer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class DecoderLayer(nn.Module):
    """单层 Transformer Decoder：Masked Self-Attn + Cross-Attn + FFN。"""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Masked self-attention
        self_out = self.self_attn(tgt, tgt, tgt, tgt_mask)
        x = self.norm1(tgt + self.dropout1(self_out))
        # Cross-attention to encoder output
        cross_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(cross_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


# ═══════════════════════════════════════════════════════════
# Encoder / Decoder 完整模块
# ═══════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """完整 Encoder：Embedding + PositionalEncoding + N 层 EncoderLayer。"""

    def __init__(self, vocab_size: int, max_len: int, d_model: int, n_heads: int,
                 num_layers: int, dim_feedforward: int, pe_type: str = 'sinusoidal',
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model, pe_type, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """完整 Decoder：Embedding + PositionalEncoding + N 层 DecoderLayer + 输出投影。"""

    def __init__(self, vocab_size: int, max_len: int, d_model: int, n_heads: int,
                 num_layers: int, dim_feedforward: int, pe_type: str = 'sinusoidal',
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model, pe_type, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════
# Seq2Seq Transformer
# ═══════════════════════════════════════════════════════════

class TransformerSeq2Seq(nn.Module):
    """Transformer Encoder-Decoder Seq2Seq 模型。"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, max_len: int,
                 d_model: int = 128, n_heads: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 512, pe_type: str = 'sinusoidal',
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.pe_type = pe_type

        self.encoder = Encoder(src_vocab_size, max_len, d_model, n_heads,
                               num_layers, dim_feedforward, pe_type, dropout)
        self.decoder = Decoder(tgt_vocab_size, max_len, d_model, n_heads,
                               num_layers, dim_feedforward, pe_type, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """标准训练 forward。

        src: (batch, src_len)
        tgt: (batch, tgt_len) — 包含 <bos> 和 <eos>
        src_mask: (batch, 1, 1, src_len) — encoder 的 padding mask
        tgt_mask: (batch, 1, tgt_len, tgt_len) — decoder 的 combined mask

        返回 logits: (batch, tgt_len, tgt_vocab_size)
        """
        if src_mask is None:
            src_mask = _make_pad_mask(src, self.pad_idx)
        if tgt_mask is None:
            tgt_pad_mask = _make_pad_mask(tgt, self.pad_idx)
            tgt_sub_mask = _make_subsequent_mask(tgt.size(1), device=tgt.device)
            tgt_mask = tgt_pad_mask + tgt_sub_mask

        enc_out = self.encoder(src, src_mask)
        logits = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return logits

    def get_pos_encoding(self) -> PositionalEncoding:
        return self.encoder.pos_encoding


# ═══════════════════════════════════════════════════════════
# Mask 工具函数（模块级函数）
# ═══════════════════════════════════════════════════════════

def _make_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """由 token id 序列生成 padding mask。

    seq: (batch, seq_len)
    返回: (batch, 1, 1, seq_len)，padding 位置为 -inf，其余为 0。
    """
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2).float() * -1e9


def _make_subsequent_mask(sz: int, device: torch.device = None) -> torch.Tensor:
    """生成 causal mask (sz, sz) → (1, 1, sz, sz)，上三角为 -inf。"""
    mask = torch.triu(torch.ones(sz, sz, device=device) * -1e9, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


# ── 对外兼容旧骨架的别名 ──

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """生成 target 序列 causal mask。(sz, sz)，上三角 True。"""
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()


def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """生成 padding mask。(batch, 1, 1, seq_len)，padding 位置 True。"""
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回用于模型 forward 的四个 mask。"""
    src_mask = _make_pad_mask(src, pad_idx)
    tgt_pad_mask = _make_pad_mask(tgt, pad_idx)
    tgt_sub_mask = _make_subsequent_mask(tgt.size(1), device=tgt.device)
    tgt_mask = tgt_pad_mask + tgt_sub_mask
    return src_mask, tgt_mask, _make_pad_mask(src, pad_idx), _make_pad_mask(tgt, pad_idx)
