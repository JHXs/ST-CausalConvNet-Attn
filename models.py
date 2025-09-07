# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.parametrizations import weight_norm
from attention_utils import HAttention, SimplifiedHAttention, MultiHeadAttention

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

class STCN_MultiHeadAttention(nn.Module):
    """
    使用多头注意力的STCN模型
    """
    
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout, 
                 attention_heads=8, use_rotary=True):
        super(STCN_MultiHeadAttention, self).__init__()
        
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
        
        # Log-linear attention 替换简单时间注意力
        attention_embed_dim = num_channels[-1]
        self.temporal_attention = MultiHeadAttention(
            embed_dim=attention_embed_dim,
            num_heads=attention_heads,
            dropout=dropout,
            use_rotary=use_rotary
        )
        
        # 输出层
        self.linear = nn.Linear(num_channels[-1], output_size)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(num_channels[-1])

    def forward(self, x):
        # 原始STCN处理
        conv_out = self.conv(x).squeeze(1)  # [batch, channels, seq_len]
        output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, features]
        
        # Multi-head attention处理
        attended_output = self.temporal_attention(output)  # [batch, seq_len, features]
        
        # 残差连接：保持原始特征
        residual = output
        
        # 残差连接 + 层归一化
        final_features = self.layer_norm(attended_output + residual)
        
        # 使用最后一个时间步进行预测
        last_step = final_features[:, -1, :]  # [batch, features]
        pred = self.linear(last_step)
        
        return pred
    
    def _apply(self, fn):
        # 确保所有子模块都应用相同的设备变换
        super()._apply(fn)
        return self

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


class STCN_SimplifiedLLAttention(nn.Module):
    """
    使用简化对数线性注意力的STCN模型
    适合资源有限的场景，计算效率更高
    """
    
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout, 
                 attention_heads=4, use_rotary=True, base=2):
        super(STCN_SimplifiedLLAttention, self).__init__()
        
        # 原始STCN组件
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        
        # 简化对数线性注意力
        attention_embed_dim = num_channels[-1]
        self.temporal_attention = SimplifiedHAttention(
            embed_dim=attention_embed_dim,
            num_heads=attention_heads,
            base=base,
            dropout=dropout,
            max_length=input_size
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
        output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, features]
        
        # 简化对数线性注意力处理
        attended_output, _ = self.temporal_attention(output)  # [batch, seq_len, features]
        
        # 残差连接：保持原始特征
        residual = output
        
        # 残差连接 + 层归一化
        final_features = self.layer_norm(attended_output + residual)
        
        # 使用最后一个时间步进行预测
        last_step = final_features[:, -1, :]  # [batch, features]
        pred = self.linear(last_step)
        
        return pred
    
