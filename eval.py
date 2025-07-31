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
    y_valid_pred_final = []
    y_valid_true = []
    criterion = nn.MSELoss().to(cfg.device)
    h_state = None
    rmse_valid = 0.0
    cnt = 0
    
    for x_input_valid, y_true_valid in test_loader:
        x_input_valid = x_input_valid.to(cfg.device)
        y_true_valid = y_true_valid.to(cfg.device)
        
        if cfg.model_name == 'RNN' or cfg.model_name == 'GRU':
            # Initialize hidden state with actual batch size
            actual_batch_size = x_input_valid.shape[0]
            h_state = net.init_hidden(actual_batch_size, cfg.device)
            y_valid_pred, _h_state = net(x_input_valid, h_state)
        else:
            y_valid_pred = net(x_input_valid)
        
        y_valid_pred_final.extend(y_valid_pred.data.cpu().numpy())
        y_valid_true.extend(y_true_valid.data.cpu().numpy())
        loss_valid = criterion(y_valid_pred, y_true_valid).data
        mse_valid_batch = loss_valid.cpu().numpy()
        rmse_valid_batch = np.sqrt(mse_valid_batch)
        rmse_valid += mse_valid_batch
        cnt += 1
    
    y_valid_pred_final = np.array(y_valid_pred_final).reshape((-1, 1))
    y_valid_true = np.array(y_valid_true).reshape((-1, 1))
    rmse_valid = np.sqrt(rmse_valid / cnt)
    mae_valid = metrics.mean_absolute_error(y_valid_true, y_valid_pred_final)
    r2_valid = metrics.r2_score(y_valid_true, y_valid_pred_final)

    print('\nRMSE_valid: {:.4f}  MAE_valid: {:.4f}  R2_valid: {:.4f}\n'.format(rmse_valid, mae_valid, r2_valid))
    return rmse_valid, mae_valid, r2_valid


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
    elif cfg.model_name == 'STCN':
        net = models.STCN(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                          num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, net))

    # Load model parameters
    net = net.to(cfg.device)
    net.load_state_dict(torch.load(cfg.model_save_pth, map_location=cfg.device))
    print(utils.get_param_number(net=net))

    # Evaluation
    eval(net, test_loader)


if __name__ == '__main__':
    main()
