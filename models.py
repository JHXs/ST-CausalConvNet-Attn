# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.parametrizations import weight_norm


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
    只保留时间注意力机制的STCN模型
    主要改进：
    1. 移除空间注意力机制，只保留时间注意力
    2. 简化特征融合策略
    3. 保持原有特征的残差连接
    4. 使用最后一个时间步作为输出
    """
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout):
        super(STCN_Attention, self).__init__()
        
        # 原始STCN组件 - 与原始STCN保持一致
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        
        # 时间注意力机制
        self.temporal_attention = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 4),
            nn.ReLU(),
            nn.Linear(num_channels[-1] // 4, 1),
            nn.Sigmoid()
        )
        
        # 输出层
        self.linear = nn.Linear(num_channels[-1], output_size)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(num_channels[-1])

    def forward(self, x):
        # 原始STCN处理
        conv_out = self.conv(x).squeeze(1)  # [batch, channels, seq_len]
        output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, features]
        
        # 时间注意力
        temporal_weights = self.temporal_attention(output)  # [batch, seq_len, 1]
        temporal_enhanced = output * temporal_weights  # [batch, seq_len, features]
        
        # 残差连接：保持原始特征
        residual = output
        
        # 残差连接 + 层归一化
        final_features = self.layer_norm(temporal_enhanced + residual)
        
        # 使用最后一个时间步进行预测
        last_step = final_features[:, -1, :]  # [batch, features]
        pred = self.linear(last_step)
        
        return pred


class STCN_LogLinearAttention(nn.Module):
    """
    使用log-linear attention的STCN模型
    替换原有STCN_Attention中的简单时间注意力
    """
    
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout, 
                 attention_heads=8, use_rotary=True, device=None):
        super(STCN_LogLinearAttention, self).__init__()
        self.device = device if device is not None else 'cpu'
        
        # 原始STCN组件
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        ).to(self.device)
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.tcn.to(self.device)
        
        # Log-linear attention 替换简单时间注意力
        attention_embed_dim = num_channels[-1]
        self.temporal_attention = LogLinearAttention(
            embed_dim=attention_embed_dim,
            num_heads=attention_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            device=self.device
        )
        
        # 输出层
        self.linear = nn.Linear(num_channels[-1], output_size).to(self.device)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(num_channels[-1]).to(self.device)

    def forward(self, x):
        # 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 原始STCN处理
        conv_out = self.conv(x).squeeze(1)  # [batch, channels, seq_len]
        output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, features]
        
        # 确保输出在正确的设备上
        output = output.to(self.device)
        
        # Log-linear attention处理
        attended_output = self.temporal_attention(output)  # [batch, seq_len, features]
        
        # 残差连接：保持原始特征
        residual = output
        
        # 残差连接 + 层归一化
        final_features = self.layer_norm(attended_output + residual)
        
        # 使用最后一个时间步进行预测
        last_step = final_features[:, -1, :]  # [batch, features]
        pred = self.linear(last_step)
        
        return pred


class LogLinearAttention(nn.Module):
    """
    Log-linear attention mechanism for temporal sequences
    替代简单的时间注意力机制
    """
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, use_rotary=True, device=None):
        super(LogLinearAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_rotary = use_rotary
        self.device = device if device is not None else 'cpu'
        
        # 线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            attention_mask: [batch_size, seq_len] (可选)
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        batch_size, seq_len, _ = x.shape
        
        # 线性变换 - 确保权重在正确的设备上
        self.q_proj = self.q_proj.to(self.device)
        self.k_proj = self.k_proj.to(self.device)
        self.v_proj = self.v_proj.to(self.device)
        
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用attention mask
        if attention_mask is not None:
            # 扩展mask维度
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)
        
        return output


def get_param_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num
