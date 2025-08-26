# coding:utf-8

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
import models
import utils
import config as cfg


def eval(net, test_loader, plot=False):
    print('\nStart evaluating...\n')
    net.eval()
    criterion = nn.MSELoss().to(cfg.device)
    h_state = None
    
    # Initialize accumulators on device
    rmse_valid = 0.0
    mae_valid = 0.0
    cnt = 0
    
    # Lists for R2 calculation (collected at the end)
    y_valid_pred_final = []
    y_valid_true = []
    
    with torch.no_grad():
        for x_input_valid, y_true_valid in test_loader:
            x_input_valid = x_input_valid.to(cfg.device)
            y_true_valid = y_true_valid.to(cfg.device)
            
            if cfg.model_name == 'RNN' or cfg.model_name == 'GRU':
                actual_batch_size = x_input_valid.shape[0]
                h_state = net.init_hidden(actual_batch_size, cfg.device)
                y_valid_pred, _h_state = net(x_input_valid, h_state)
            else:
                y_valid_pred = net(x_input_valid)
            
            # Calculate metrics on GPU
            mse_batch = criterion(y_valid_pred, y_true_valid)
            rmse_batch = torch.sqrt(mse_batch)
            mae_batch = torch.mean(torch.abs(y_valid_pred - y_true_valid))
            
            rmse_valid += rmse_batch
            mae_valid += mae_batch
            cnt += 1
            
            # Collect predictions for R2 calculation
            y_valid_pred_final.append(y_valid_pred.cpu())
            y_valid_true.append(y_true_valid.cpu())
    
    # Calculate final metrics
    rmse_valid = rmse_valid / cnt
    mae_valid = mae_valid / cnt
    
    # Calculate R2 on CPU (requires sklearn)
    y_valid_pred_final = torch.cat(y_valid_pred_final).numpy().reshape((-1, 1))
    y_valid_true = torch.cat(y_valid_true).numpy().reshape((-1, 1))
    r2_valid = metrics.r2_score(y_valid_true, y_valid_pred_final)

    print('\nRMSE_valid: {:.4f}  MAE_valid: {:.4f}  R2_valid: {:.4f}\n'.format(
        rmse_valid.item(), mae_valid.item(), r2_valid))
    return rmse_valid.item(), mae_valid.item(), r2_valid


def main():
    # Hyper Parameters
    cfg.print_params()
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)

    # Load data
    print('\nLoading data...\n')
    train_loader, valid_loader, test_loader = utils.load_data(f_x=cfg.f_x, f_y=cfg.f_y, batch_size=cfg.batch_size)

    # Generate model
    net = None
    if cfg.model_name == 'RNN':
        net = models.SimpleRNN(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'GRU':
        net = models.SimpleGRU(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'LSTM':
        net = models.SimpleLSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'TCN':
        net = models.TCN(input_size=cfg.input_size, output_size=cfg.output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'TCN_Attention':
        net = models.TCN_Attention(input_size=cfg.input_size, output_size=cfg.output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'STCN':
        net = models.STCN(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                          num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'STCN_Attention':
        net = models.STCN_Attention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                          num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'STCN_LogLinearAttention':
        net = models.STCN_LogLinearAttention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                             num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                             attention_heads=cfg.attention_heads, use_rotary=cfg.use_rotary, device=cfg.device)
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, net))

    # Load model parameters
    net = net.to(cfg.device)
    net.load_state_dict(torch.load(cfg.model_save_pth, map_location=cfg.device))
    print(utils.get_param_number(net=net))

    # Evaluation
    eval(net, test_loader)


if __name__ == '__main__':
    main()
