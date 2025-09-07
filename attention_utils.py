# coding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HType:
    WEAK = 0
    STRONG = 1


def ceil_log(x: int, b: int) -> int:
    return math.ceil(math.log(x, b))


def floor_log(x: int, b: int) -> int:
    return math.floor(math.log(x, b))


def get_num_levels(length: int, base: int) -> int:
    return ceil_log(length, base) + 1


def get_level_index_weak(t: int, s: int, b: int) -> int:
    l = floor_log(t, b)
    v = b ** l
    if s < v:
        return l + 1
    return get_level_index_weak(t - v, s - v, b)


def get_level_index_strong(t: int, s: int, b: int) -> int:
    if b != 2:
        raise NotImplementedError
    l = floor_log(t, b)
    v = b ** max(l - 1, 0)
    r = t - b ** l
    if s < v:
        return l + 1
    elif s < v * 2 and r >= v:
        return l + 1
    elif s < v * 2 and r < v:
        return get_level_index_strong(t - v * 1, s - v * 1, b)
    else:
        return get_level_index_strong(t - v * 2, s - v * 2, b)


def get_level_index(t: int, s: int, b: int, htype: int) -> int:
    if t <= s:
        raise ValueError
    if htype == HType.WEAK:
        return get_level_index_weak(t=t, s=s, b=b)
    if htype == HType.STRONG:
        return get_level_index_strong(t=t, s=s, b=b)
    raise ValueError


def make_hierarchical_attention_matrix(
    A: torch.Tensor,
    L: torch.Tensor,
    base: int,
    htype: int,
) -> torch.Tensor:
    """
    构建层次化注意力矩阵
    
    Args:
        A: 衰减矩阵 [batch, length, num_heads]
        L: 层次化权重 [batch, length, num_heads, num_levels]
        base: 层次化基数
        htype: 层次化类型 (0=WEAK, 1=STRONG)
    
    Returns:
        H: 层次化注意力矩阵 [batch, num_heads, length, length]
    """
    batch, length, num_heads = A.shape
    H = torch.zeros(
        (batch, num_heads, length, length),
        dtype=A.dtype,
        device=A.device)

    for target in range(length):
        H[..., target, target] = L[:, target, :, 0]

        for source in range(target):
            l = get_level_index(t=target, s=source, b=base, htype=htype)
            h_H = L[:, target, :, l]
            h_SSS = torch.prod(A[:, source + 1: target + 1, :], dim=1)
            H[..., target, source] = h_H * h_SSS

    return H


class HAttention(nn.Module):
    """
    简化的对数线性注意力机制
    基于层次化矩阵分解，实现O(T log T)复杂度的注意力计算
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_levels: int = None,
        base: int = 2,
        htype: int = HType.WEAK,
        dropout: float = 0.1,
        max_length: int = 2048
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_levels = num_levels or get_num_levels(max_length, base)
        self.base = base
        self.htype = htype
        self.dropout = dropout
        
        # 线性投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 衰减参数矩阵 A [batch, length, num_heads]
        self.A_log = nn.Parameter(torch.randn(max_length, num_heads))
        self.A_bias = nn.Parameter(torch.zeros(max_length, num_heads))
        
        # 层次化权重 L [batch, length, num_heads, num_levels]
        self.L = nn.Parameter(torch.randn(max_length, num_heads, self.num_levels))
        self.L_bias = nn.Parameter(torch.zeros(max_length, num_heads, self.num_levels))
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> tuple:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            output: 输出张量 [batch_size, seq_len, embed_dim]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len] (可选)
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算衰减矩阵 A
        A = torch.sigmoid(self.A_log[:seq_len] + self.A_bias[:seq_len])  # [seq_len, num_heads]
        A = A.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, num_heads]
        
        # 获取层次化权重 L
        L = self.L[:seq_len] + self.L_bias[:seq_len]  # [seq_len, num_heads, num_levels]
        L = L.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, seq_len, num_heads, num_levels]
        
        # 构建层次化注意力矩阵
        H = make_hierarchical_attention_matrix(A, L, self.base, self.htype)
        # H: [batch_size, num_heads, seq_len, seq_len] 已经是正确格式
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 扩展掩码维度
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            H = H.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力输出
        attn_weights = F.softmax(H, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 确保维度匹配，注意矩阵乘法的维度顺序
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        # v: [batch_size, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑回原始形状
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        
        # 输出投影
        output = self.out_proj(output)
        
        if return_attention_weights:
            return output, attn_weights
        return output, None


class SimplifiedHAttention(nn.Module):
    """
    更简化的对数线性注意力，使用固定的层次化结构
    适合资源有限的场景
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        base: int = 2,
        dropout: float = 0.1,
        max_length: int = 1024
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.base = base
        self.dropout = dropout
        
        # 线性投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 简化的层次化参数
        self.hierarchical_weights = nn.Parameter(
            torch.randn(max_length, num_heads, 3))  # 3 levels
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # 缓存层次化级别
        self.level_cache = {}
        
    def get_level_indices(self, seq_len: int) -> torch.Tensor:
        """获取序列的层次化级别索引"""
        if seq_len in self.level_cache:
            return self.level_cache[seq_len]
        
        indices = torch.zeros(seq_len, seq_len, dtype=torch.long)
        for target in range(seq_len):
            for source in range(target):
                level = get_level_index_weak(t=target, s=source, b=self.base)
                indices[target, source] = min(level, 2)  # 限制最大级别
        
        self.level_cache[seq_len] = indices
        return indices
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> tuple:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            output: 输出张量 [batch_size, seq_len, embed_dim]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len] (可选)
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 获取层次化级别
        level_indices = self.get_level_indices(seq_len)  # [seq_len, seq_len]
        level_indices = level_indices.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        level_indices = level_indices.expand(batch_size, self.num_heads, -1, -1)
        
        # 获取层次化权重
        hier_weights = self.hierarchical_weights[:seq_len, :, :3]  # [seq_len, num_heads, 3]
        hier_weights = hier_weights.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [seq_len, num_heads, seq_len, 3]
        
        # 根据级别选择权重
        weights = torch.gather(hier_weights, 3, level_indices)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 计算衰减因子
        decay_factors = torch.pow(0.5, level_indices.float())  # 指数衰减
        weights = weights * decay_factors
        
        # 归一化
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 应用注意力掩码
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            weights = weights.masked_fill(mask == 0, 0.0)
        
        # 应用dropout
        weights = self.dropout_layer(weights)
        
        # 计算注意力输出
        output = torch.matmul(weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑回原始形状
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        
        # 输出投影
        output = self.out_proj(output)
        
        if return_attention_weights:
            return output, weights
        return output, None
    