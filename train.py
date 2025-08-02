# coding:utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
import models
import utils
import config as cfg


def train(net, train_loader, valid_loader, test_loader, plot=False):
    rmse_train_list = []
    rmse_valid_list = []
    mae_valid_list = []
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss().to(cfg.device)
    h_state = None

    for epoch in range(1, cfg.n_epochs + 1):
        # Training phase
        net.train()
        rmse_train = 0.0
        cnt = 0
        
        for batch_idx, (x_input, y_true) in enumerate(train_loader):
            x_input = x_input.to(cfg.device)
            y_true = y_true.to(cfg.device)

            if cfg.model_name == 'RNN' or cfg.model_name == 'GRU':
                actual_batch_size = x_input.shape[0]
                h_state = net.init_hidden(actual_batch_size, cfg.device)
                y_pred, _h_state = net(x_input, h_state)
                h_state = _h_state.data
            else:
                y_pred = net(x_input)

            loss = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rmse_train_batch = torch.sqrt(loss)
                rmse_train += rmse_train_batch
            
            if batch_idx % int(len(train_loader) / 5) == 0:
                progress = batch_idx / len(train_loader)
                print('epoch: {}  progress: {:.0f}%  loss: {:.3f}  rmse: {:.3f}'.format(
                    epoch, progress * 100, loss, rmse_train_batch))
            cnt += 1
        
        rmse_train = rmse_train / cnt

        # Validation phase
        net.eval()
        rmse_valid = 0.0
        mae_valid = 0.0
        cnt = 0
        
        with torch.no_grad():
            for x_input_valid, y_true_valid in valid_loader:
                x_input_valid = x_input_valid.to(cfg.device)
                y_true_valid = y_true_valid.to(cfg.device)
                
                if cfg.model_name == 'RNN' or cfg.model_name == 'GRU':
                    actual_batch_size = x_input_valid.shape[0]
                    h_state = net.init_hidden(actual_batch_size, cfg.device)
                    y_valid_pred, _h_state = net(x_input_valid, h_state)
                    h_state = _h_state.data
                else:
                    y_valid_pred = net(x_input_valid)
                
                # Calculate metrics on GPU
                mse_batch = criterion(y_valid_pred, y_true_valid)
                rmse_batch = torch.sqrt(mse_batch)
                mae_batch = torch.mean(torch.abs(y_valid_pred - y_true_valid))
                
                rmse_valid += rmse_batch
                mae_valid += mae_batch
                cnt += 1
        
        rmse_valid = rmse_valid / cnt
        mae_valid = mae_valid / cnt

        # Convert to CPU for storage and display
        rmse_train_cpu = rmse_train.item()
        rmse_valid_cpu = rmse_valid.item()
        mae_valid_cpu = mae_valid.item()
        
        rmse_train_list.append(rmse_train_cpu)
        rmse_valid_list.append(rmse_valid_cpu)
        mae_valid_list.append(mae_valid_cpu)
        
        # save the best model
        if rmse_valid_cpu == np.min(rmse_valid_list):
            torch.save(net.state_dict(), cfg.model_save_pth)

        print('\n>>> epoch: {}  RMSE_train: {:.4f}  RMSE_valid: {:.4f} MAE_valid: {:.4f}\n'
              '    RMSE_valid_min: {:.4f}  MAE_valid_min: {:.4f}\n'
              .format(epoch, rmse_train_cpu, rmse_valid_cpu, mae_valid_cpu, 
                     np.min(rmse_valid_list), np.min(mae_valid_list)))


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
    net = net.to(cfg.device)
    # sys.exit(0)

    # Training
    print('\nStart training...\n')
    train(net, train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    main()
