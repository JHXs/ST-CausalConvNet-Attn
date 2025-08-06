# coding=utf-8

# model hyper-parameters
rand_seed = 314
# Choose data file based on model type
model_name = 'STCN_DualAttention'  # ['RNN', 'GRU', 'LSTM', 'TCN', 'STCN', 'STCN_DualAttention']
if model_name in ['RNN', 'GRU', 'LSTM', 'TCN']:
    f_x = './data/xy/x_1013_3d_mean.pkl'  # 3D data for sequential models
else:  # STCN and STCN_DualAttention use 4D data
    f_x = './data/xy/x_1013.pkl'  # 4D data for STCN models
f_y = './data/xy/y_1013.pkl'

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_size = 12
hidden_size = 32
output_size = 1
num_layers = 4
levels = 4
kernel_size = 4
dropout = 0.25
in_channels = 18  ## 输入数据的通道数，选择的相关站点数

batch_size = 32
lr = 1e-2
n_epochs = 100  # 先测试10个epoch

# 学习率调度参数
lr_scheduler = True
lr_patience = 5  # 5个epoch没有改善就降低学习率
lr_factor = 0.5  # 学习率衰减因子
min_lr = 1e-5   # 最小学习率

# 早停参数
early_stopping = True
es_patience = 10  # 10个epoch没有改善就停止训练
model_save_pth = './models/model_{}.pth'.format(model_name)


def print_params():
    print('\n------ Parameters ------')
    print('rand_seed = {}'.format(rand_seed))
    print('f_x = {}'.format(f_x))
    print('f_y = {}'.format(f_y))
    print('device = {}'.format(device))
    print('input_size = {}'.format(input_size))
    print('hidden_size = {}'.format(hidden_size))
    print('num_layers = {}'.format(num_layers))
    print('output_size = {}'.format(output_size))
    print('levels (for TCN) = {}'.format(levels))
    print('kernel_size (for TCN) = {}'.format(kernel_size))
    print('dropout (for TCN) = {}'.format(dropout))
    print('in_channels (for STCN) = {}'.format(in_channels))
    print('batch_size = {}'.format(batch_size))
    print('lr = {}'.format(lr))
    print('n_epochs = {}'.format(n_epochs))
    print('model_save_pth = {}'.format(model_save_pth))
    print('------------------------\n')
