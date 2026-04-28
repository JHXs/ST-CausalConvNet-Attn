# coding=utf-8
import torch

# model hyper-parameters
rand_seed = 314
# Choose data file based on model type
model_name = "PatchTST"  # ['RNN', "LSTM", "GRU", 'TCN', 'TCN_Attention', 'PatchTST', 'STCN', 'ST_PatchTST', 'STCN_Attention', 'ImprovedSTCN_Attention', 'AdvancedSTCN_Attention', 'STCN_LLAttention', 'BiLSTM', 'LSTM_GRU', 'BiLSTM_GRU', 'BiLSTM_CNN', 'LSTM_CNN']
if model_name in ["RNN", "LSTM", "GRU", "TCN", "TCN_Attention", "PatchTST", "BiLSTM", "LSTM_GRU", "BiLSTM_GRU", "BiLSTM_CNN", "LSTM_CNN"]:
    f_x = "./data/xy/x_9022_3d_mean.pkl"  # 3D data for sequential models
else:  # STCN use 4D data
    f_x = "./data/xy/x_9022.pkl"  # 4D data for STCN models
f_y = "./data/xy/y_9022.pkl"


device = "cuda" if torch.cuda.is_available() else "cpu"
# 在 ROCm 环境中禁用 MIOpen（torch.backends.cudnn），避免 HIPRTC 编译失败
disable_miopen = True
seq_len = 24*14  # 输入序列长度（历史时间步数）
input_size = 12
hidden_size = 32  # 32
output_size = 72  # 预测步长
num_layers = 4  # 4
levels = 4  # 4
kernel_size = 4  # 4
dropout = 0.25  # 0.25

in_channels = 12  ## for STCN 输入数据的通道数，选择的相关站点数：北京18，广州12

# PatchTST 特定参数（仅 model_name == "PatchTST" 时使用）
patchtst_patch_len = 16
patchtst_stride = 8
patchtst_d_model = 64
patchtst_d_ff = 256
patchtst_n_heads = 4
patchtst_n_layers = 3
patchtst_revin = True

# Log-linear attention 特定参数
attention_heads = 8  # 注意力头数 必须 hidden_size % attention_heads == 0
use_rotary = True  # 暂时禁用位置编码以避免维度错误

batch_size = 32
lr = 1e-3  # 1e-3
n_epochs = 50
weight_decay = 1e-5  # 1e-5  # L2正则化系数

# 基于指标的衰减学习率调度参数
lr_scheduler = False
lr_patience = 5  # 5个epoch没有改善就降低学习率
lr_factor = 0.5  # 学习率衰减因子0.2-0.5
min_lr = 1e-5  # 最小学习率

# 早停参数
early_stopping = False
es_patience = 20  # 早停次数
model_save_pth = "./models/model_{}.pth".format(model_name)

# 可视化
plt = False  # [True, False]

# 报告生成
generate_report = False  # 是否生成训练验证报告

# 数据加载方式
data_to_gpu_memory = True  # 是否将整个数据集加载到GPU显存（避免CPU-GPU传输瓶颈）

prediction_variables = ["PM25_Concentration", "SO2_Concentration", "O3_Concentration"]  # 预测变量列表，例如 ["PM25_Concentration", "SO2_Concentration", "O3_Concentration"]


def get_num_prediction_variables():
    return max(1, len(prediction_variables))


def get_model_output_size():
    return output_size * get_num_prediction_variables()

def print_params():
    print("\n------ Parameters ------")
    print("rand_seed = {}".format(rand_seed))
    print("f_x = {}".format(f_x))
    print("f_y = {}".format(f_y))
    print("device = {}".format(device))
    print("disable_miopen = {}".format(disable_miopen))
    print("input_size = {}".format(input_size))
    print("hidden_size = {}".format(hidden_size))
    print("num_layers = {}".format(num_layers))
    print("output_size = {}".format(output_size))
    print("levels (for TCN) = {}".format(levels))
    print("kernel_size (for TCN) = {}".format(kernel_size))
    print("dropout (for TCN) = {}".format(dropout))
    print("in_channels (for STCN) = {}".format(in_channels))
    print("patchtst_patch_len = {}".format(patchtst_patch_len))
    print("patchtst_stride = {}".format(patchtst_stride))
    print("patchtst_d_model = {}".format(patchtst_d_model))
    print("patchtst_d_ff = {}".format(patchtst_d_ff))
    print("patchtst_n_heads = {}".format(patchtst_n_heads))
    print("patchtst_n_layers = {}".format(patchtst_n_layers))
    print("patchtst_revin = {}".format(patchtst_revin))
    print("attention_heads (for *-Attention) = {}".format(attention_heads))
    print("use_rotary (for LogLinearAttention) = {}".format(use_rotary))
    print("batch_size = {}".format(batch_size))
    print("lr = {}".format(lr))
    print("n_epochs = {}".format(n_epochs))
    print("prediction_variables = {}".format(prediction_variables))
    print("model_output_size = {}".format(get_model_output_size()))
    print("model_save_pth = {}".format(model_save_pth))
    print("------------------------\n")
