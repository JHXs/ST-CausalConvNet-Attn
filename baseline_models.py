import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=64, output_size=1, num_layers=2, dropout=0.25):
        """
        Bidirectional LSTM
        Architecture:
        1. 2 Bidirectional LSTM layers (64 units each)
        2. Final dense output layer
        """
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
            # dropout=dropout if num_layers > 1 else 0
        )

        # Output layer input size = hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: (batch, seq, feature)

        # BiLSTM
        # out: (batch, seq, hidden * 2)
        out, _ = self.lstm(x)

        # Global Max Pooling (to capture features from both directions effectively)
        out = out.permute(0, 2, 1) # (batch, hidden*2, seq)
        out = F.max_pool1d(out, out.size(2)).squeeze(2) # (batch, hidden*2)

        output = self.fc(out)
        return output

class LSTM_GRU(nn.Module):
    # 构造函数
    def __init__(self, input_size, hidden_size_lstm=64, hidden_size_gru=32, output_size=1, num_layers_lstm=2, num_layers_gru=2, dropout=0.25):
        """
        初始化混合模型

        :param input_size: 输入特征的维度 (例如，如果你用 5 个特征预测，input_size=5)
        :param hidden_size_lstm: LSTM 层的隐藏单元数 (64)
        :param hidden_size_gru: GRU 层的隐藏单元数 (32)
        :param output_size: 输出的维度 (预测 AQI，通常是 1)
        :param num_layers_lstm: LSTM 的层数 (通常为 1)
        :param num_layers_gru: GRU 的层数 (通常为 1)
        """
        super(LSTM_GRU, self).__init__()

        # --- 1. LSTM 层 (64 单元) ---
        # input_size: 特征数
        # hidden_size_lstm: 64
        # batch_first=True: 输入数据的形状为 (batch_size, sequence_length, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers_lstm,
            batch_first=True,
            # dropout=dropout if num_layers_lstm > 1 else 0  # 仅在多层时使用 dropout
        )

        # --- 2. GRU 层 (32 单元) ---
        # input_size: 必须是前一个 LSTM 层的 hidden_size (64)
        # hidden_size_gru: 32
        self.gru = nn.GRU(
            input_size=hidden_size_lstm, # 上一层 LSTM 的输出作为 GRU 的输入
            hidden_size=hidden_size_gru, # 32
            num_layers=num_layers_gru,
            batch_first=True,
            # dropout=dropout if num_layers_gru > 1 else 0
        )

        # --- 3. 全连接输出层 ---
        # input_size: 必须是 GRU 层的 hidden_size (32)
        # output_size: 1 (因为我们要预测单个 AQI 值)
        self.fc = nn.Linear(hidden_size_gru, output_size)

    # 前向传播
    def forward(self, x):
        # x 的形状假设为 (batch_size, sequence_length, input_size)

        # 1. 通过 LSTM 层
        # out_lstm: (batch_size, sequence_length, hidden_size_lstm)
        # _: 隐藏状态和细胞状态 (我们通常只需要 out_lstm)
        out_lstm, _ = self.lstm(x)
        # 

        # 2. 通过 GRU 层
        # out_gru: (batch_size, sequence_length, hidden_size_gru)
        out_gru, _ = self.gru(out_lstm)

        # 3. 提取最终时间步的输出
        # 由于是序列预测，我们通常只关心最后一个时间步的输出
        # out_gru[:, -1, :]: 形状 (batch_size, hidden_size_gru)
        out_final = out_gru[:, -1, :]

        # 4. 通过全连接层进行输出
        # out: 形状 (batch_size, output_size)
        output = self.fc(out_final)

        return output

class BiLSTM_GRU(nn.Module):
    def __init__(self, input_size, hidden_size_lstm=64, hidden_size_gru=32, output_size=1, num_layers_lstm=2, num_layers_gru=1, dropout=0.25):
        """
        BiLSTM with GRU
        Architecture:
        1. 2 Bidirectional LSTM layers (64 units each)
        2. 1 GRU layer (32 units)
        3. Final dense output layer
        """
        super(BiLSTM_GRU, self).__init__()

        # 1. BiLSTM Layer
        # input_size: feature dim
        # hidden_size_lstm: 64
        # num_layers_lstm: 2
        # bidirectional=True -> output dim = 64 * 2 = 128
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers_lstm,
            batch_first=True,
            bidirectional=True
            # dropout=dropout if num_layers_lstm > 1 else 0
        )

        # 2. GRU Layer
        # input_size: previous layer output size = 128
        # hidden_size_gru: 32
        # num_layers_gru: 1
        self.gru = nn.GRU(
            input_size=hidden_size_lstm * 2,
            hidden_size=hidden_size_gru,
            num_layers=num_layers_gru,
            batch_first=True
            # dropout=dropout if num_layers_gru > 1 else 0
        )

        # 3. Dense Output Layer
        self.fc = nn.Linear(hidden_size_gru, output_size)

    def forward(self, x):
        # x: (batch, seq, feature)

        # BiLSTM
        # out_lstm: (batch, seq, hidden_lstm * 2)
        out_lstm, _ = self.lstm(x)

        # GRU
        # out_gru: (batch, seq, hidden_gru)
        out_gru, _ = self.gru(out_lstm)

        # Extract last time step output
        out_final = out_gru[:, -1, :]

        # Output
        output = self.fc(out_final)

        return output

class BiLSTM_CNN(nn.Module):
    def __init__(self, input_size, num_classes, kernel_size=3):
        """
        参数说明:
        input_size: 输入特征的维度 (例如: 词向量维度或时间序列的特征数)
        num_classes: 输出层的维度 (分类任务的类别数, 或回归任务的输出数)
        kernel_size: CNN 卷积核的大小 (默认为 3)
        """
        super(BiLSTM_CNN, self).__init__()
        
        # 1. 双向 LSTM 层 (2层, 64 units)
        # hidden_size=64, bidirectional=True -> 输出维度将是 64 * 2 = 128
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=4,
            bidirectional=True,
            batch_first=True, # 输入格式为 (batch, seq, feature)
            # dropout=0.25 # 多层LSTM通常建议加一点dropout
        )
        
        # 2. 1D CNN 层 (32 units/filters)
        # 输入通道数 = LSTM输出维度 (128)
        # padding = kernel_size // 2 确保 output 长度和 input 一致 (当 stride=1)
        self.cnn = nn.Conv1d(
            in_channels=64 * 2, # 128
            out_channels=32,    # 32 units
            kernel_size=kernel_size,
            padding=kernel_size // 2 
        )
        
        # 3. Dense Output Layer (全连接层)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # --- LSTM Step ---
        # out shape: (batch_size, seq_len, hidden_size * 2) -> (batch, seq, 128)
        out, (h_n, c_n) = self.lstm(x)
        
        # --- Dimension Swap (Crucial Step) ---
        # PyTorch Conv1d 需要 (batch, channels, length)
        # 我们需要把 seq_len 放到最后，把特征维(128)作为 channels
        out = out.permute(0, 2, 1) # shape: (batch, 128, seq_len)
        
        # --- CNN Step ---
        out = self.cnn(out)        # shape: (batch, 32, seq_len)
        out = F.relu(out)          # 激活函数
        
        # --- Pooling Step ---
        # 修复逻辑：对于 BiLSTM，反向层的信息主要集中在序列前端，前向层在末端。
        # 仅取 -1 会丢失反向层的绝大部分信息。
        # 使用 Global Max Pooling 可以从整个时间窗口中提取最显著的特征（无论是过去还是未来的上下文中捕获的）。
        out = F.max_pool1d(out, out.size(2)).squeeze(2) # shape: (batch, 32)
        
        # --- Dense Layer ---
        out = self.fc(out)         # shape: (batch, num_classes)
        
        return out


class LSTM_CNN(nn.Module):
    def __init__(self, input_size, output_size=1, kernel_size=3):
        """
        LSTM with 1D Convolutional Network (CNN)
        Architecture:
        1. LSTM layer (64 units)
        2. 1D CNN layer (32 units)
        3. Final dense output layer
        """
        super(LSTM_CNN, self).__init__()
        
        # 1. LSTM Layer (64 units)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=4,
            batch_first=True
            # dropout=0.25
        )
        
        # 2. 1D CNN Layer (32 units)
        # Input channels = LSTM hidden size = 64
        self.cnn = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # 3. Dense Output Layer
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM
        # out: (batch_size, seq_len, 64)
        out, _ = self.lstm(x)
        
        # Permute for CNN: (batch, channels, length)
        out = out.permute(0, 2, 1)
        
        # CNN
        out = self.cnn(out)
        out = F.relu(out)
        
        # Global Max Pooling
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        
        # Output
        out = self.fc(out)
        return out

