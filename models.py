# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.parametrizations import weight_norm
from attention_utils import HAttention, PositionalEncoding

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=1, dropout=0.25):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity='relu',    # 'tanh' or 'relu'
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        pred = self.linear(output[:, -1, :])
        return pred, hidden

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.25):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        pred = self.linear(output[:, -1, :])
        return pred, hidden

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.25):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        pred = self.linear(output[:, -1, :])
        return pred

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(self.num_layers, batch_size, self.hidden_size, device=device)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pred = self.linear(output[:, -1, :])
        return pred


class TCN_Attention(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_Attention, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        
        # 简单的时间注意力机制
        self.attention = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Linear(num_channels[-1] // 2, 1),
            nn.Sigmoid()
        )
        
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # TCN处理
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, features]
        
        # 时间注意力
        attention_weights = self.attention(output)  # [batch, seq_len, 1]
        attended_output = output * attention_weights  # [batch, seq_len, features]
        
        # 使用最后一个时间步
        last_step = attended_output[:, -1, :]
        pred = self.linear(last_step)
        
        return pred


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class STCN(nn.Module):
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout):
        super(STCN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU()
        )
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        conv_out = self.conv(x).squeeze(1)  # squeeze channel dimension after conv2d
        output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)
        pred = self.linear(output[:, -1, :])
        return pred

class STCN_Attention(nn.Module):
    """
    使用论文中时间注意力机制的STCN模型
    基于 TCN-attention-HAR 论文的注意力设计
    """

    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout,
                 attention_heads=8, use_rotary=True):
        super(STCN_Attention, self).__init__()

        # 原始STCN组件
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        # 论文中的时间注意力机制：e_t = U*tanh(w*h_t + b)
        attention_embed_dim = num_channels[-1]
        self.attention_linear = nn.Linear(attention_embed_dim, attention_embed_dim)
        self.attention_tanh = nn.Tanh()
        self.attention_score = nn.Linear(attention_embed_dim, 1)
        self.attention_dropout = nn.Dropout(dropout)

        # 输出层
        self.linear = nn.Linear(num_channels[-1], output_size)

        # 层归一化
        self.layer_norm = nn.LayerNorm(num_channels[-1])

    def forward(self, x):
        # 原始STCN处理
        conv_out = self.conv(x).squeeze(1)  # [batch, channels, seq_len]
        output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, features]

        # 论文中的时间注意力机制
        # Step 1: e_t = U*tanh(w*h_t + b)
        attention_input = self.attention_linear(output)  # [batch, seq_len, features]
        attention_tanh = self.attention_tanh(attention_input)  # [batch, seq_len, features]
        attention_scores = self.attention_score(attention_tanh).squeeze(-1)  # [batch, seq_len]

        # Step 2: a_t = softmax(e_t) - 归一化得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, seq_len]
        attention_weights = self.attention_dropout(attention_weights)

        # Step 3: s_t = Σ_t a_t * h_t - 加权求和
        # 扩展维度以便广播乘法
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [batch, seq_len, 1]
        attended_output = output * attention_weights_expanded  # [batch, seq_len, features]

        # 对时间维度求和得到最终的注意力表示
        attended_features = torch.sum(attended_output, dim=1)  # [batch, features]

        # 残差连接：使用原始输出的最后一个时间步
        residual = output[:, -1, :]  # [batch, features]

        # 残差连接 + 层归一化
        final_features = self.layer_norm(attended_features + residual)

        # 最终预测
        pred = self.linear(final_features)

        return pred
    
    def _apply(self, fn):
        # 确保所有子模块都应用相同的设备变换
        super()._apply(fn)
        return self


class ImprovedSTCN_Attention(nn.Module):
    """
    改进版STCN注意力模型，使用多头注意力和位置编码
    """
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout,
                 attention_heads: int = 4):
        super(ImprovedSTCN_Attention, self).__init__()

        # 原始STCN组件
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        # 位置编码
        self.pos_encoding = PositionalEncoding(num_channels[-1])
        
        # 多头自注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=num_channels[-1],
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] * 2, num_channels[-1])
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(num_channels[-1])
        self.norm2 = nn.LayerNorm(num_channels[-1])
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.linear = nn.Linear(num_channels[-1], output_size)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # 原始STCN处理
        conv_out = self.conv(x).squeeze(1)  # [batch, seq_len, features]
        tcn_output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, features]

        # 添加位置编码
        tcn_output = self.pos_encoding(tcn_output)
        
        # 多头自注意力 + 残差连接
        attn_output, _ = self.multihead_attn(tcn_output, tcn_output, tcn_output)
        attn_output = self.norm1(attn_output + tcn_output)  # 第一个残差连接
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(attn_output)
        final_output = self.norm2(ff_output + attn_output)  # 第二个残差连接
        
        # 使用最后一个时间步进行预测
        pred = self.linear(self.dropout(final_output[:, -1, :]))
        
        return pred


class AdvancedSTCN_Attention(nn.Module):
    """
    高级版STCN注意力模型，优化R2值的设计
    """
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout,
                 attention_heads: int = 2):
        super(AdvancedSTCN_Attention, self).__init__()

        # 原始STCN组件
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        # 可学习的位置编码（而不是固定的正弦余弦）
        self.learnable_pos_encoding = nn.Parameter(torch.randn(1, 100, num_channels[-1]) * 0.1)
        
        # 改进的多头注意力机制，使用更少的头和更深的网络
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=num_channels[-1],
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 更深的前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] * 4),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] * 4, num_channels[-1] * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] * 2, num_channels[-1])
        )
        
        # 改进的层归一化
        self.norm1 = nn.LayerNorm(num_channels[-1])
        self.norm2 = nn.LayerNorm(num_channels[-1])
        
        # 从TCN路径到最终输出的残差连接
        self.tcn_residual_proj = nn.Linear(num_channels[-1], num_channels[-1])
        
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.linear = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, output_size)
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # 原始STCN处理
        conv_out = self.conv(x).squeeze(1)  # [batch, seq_len, features]
        tcn_output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, features]
        
        # 获取序列长度并截取相应的位置编码
        seq_len = tcn_output.size(1)
        pos_encoding = self.learnable_pos_encoding[:, :seq_len, :].expand(tcn_output.size(0), -1, -1)
        
        # 添加可学习位置编码
        tcn_output_with_pos = tcn_output + pos_encoding
        
        # 多头自注意力 + 残差连接
        attn_output, _ = self.multihead_attn(tcn_output_with_pos, tcn_output_with_pos, tcn_output_with_pos)
        attn_output = self.norm1(attn_output + tcn_output)  # 残差连接回原始TCN输出
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(attn_output)
        final_output = self.norm2(ff_output + attn_output)  # 第二个残差连接
        
        # 融合原始TCN最后一个时间步的特征
        tcn_last = self.tcn_residual_proj(tcn_output[:, -1, :])  # [batch, features]
        combined_features = final_output[:, -1, :] + tcn_last  # 残差连接
        
        # 应用dropout和输出层
        combined_features = self.dropout(combined_features)
        pred = self.linear(combined_features)
        
        return pred

class STCN_LLAttention(nn.Module):
    """
    使用对数线性注意力的STCN模型
    基于层次化矩阵分解，实现O(T log T)复杂度的注意力计算
    """
    
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout, 
                 attention_heads=8, use_rotary=True, htype='weak', base=2):
        super(STCN_LLAttention, self).__init__()
        
        # 原始STCN组件 - 保持和原始STCN完全一致
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        
        # 对数线性注意力替换简单时间注意力
        attention_embed_dim = num_channels[-1]
        self.temporal_attention = HAttention(
            embed_dim=attention_embed_dim,
            num_heads=attention_heads,
            dropout=dropout,
            base=base,
            htype=0 if htype == 'weak' else 1,  # 0=WEAK, 1=STRONG
            max_length=24  # Match the actual sequence length from TCN output
        )
        
        # 输出层
        self.linear = nn.Linear(num_channels[-1], output_size)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(num_channels[-1])
        
        # 使用旋转位置编码
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary_dim = attention_embed_dim // attention_heads
        
    def rotary_position_encoding(self, seq_len, dim):
        """生成旋转位置编码"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.to(next(self.parameters()).device)
        positions = torch.arange(seq_len, dtype=torch.float32, device=inv_freq.device)
        freqs = torch.outer(positions, inv_freq)
        
        # 生成复数形式的位置编码
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        
        return cos_freqs, sin_freqs
    
    def apply_rotary(self, x, cos_freqs, sin_freqs):
        """应用旋转位置编码"""
        # x: [batch_size, seq_len, num_heads, head_dim]
        # 将x分成实部和虚部
        x_real, x_imag = x.chunk(2, dim=-1) if x.shape[-1] % 2 == 0 else x.chunk(2, dim=-1)
        
        # 应用旋转
        x_rot = torch.cat([
            x_real * cos_freqs.unsqueeze(0) - x_imag * sin_freqs.unsqueeze(0),
            x_real * sin_freqs.unsqueeze(0) + x_imag * cos_freqs.unsqueeze(0)
        ], dim=-1)
        
        return x_rot
    
    def forward(self, x):
        # 原始STCN处理 - 完全按照原始STCN的方式
        conv_out = self.conv(x).squeeze(1)  # squeeze channel dimension after conv2d
        # print(f"Debug: conv_out shape after squeeze: {conv_out.shape}")
        output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, features]
        # print(f"Debug: TCN output shape: {output.shape}")
        
        # 对数线性注意力处理
        attended_output, _ = self.temporal_attention(output)  # [batch, seq_len, features]
        
        # 残差连接：保持原始特征
        residual = output
        
        # 残差连接 + 层归一化
        final_features = self.layer_norm(attended_output + residual)
        
        # 使用最后一个时间步进行预测
        last_step = final_features[:, -1, :]  # [batch, features]
        pred = self.linear(last_step)
        
        return pred
    
