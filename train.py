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
    y_valid_pred_final = []
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss().to(cfg.device)
    h_state = None

    for epoch in range(1, cfg.n_epochs + 1):
        rmse_train = 0.0
        cnt = 0
        # Initialize hidden state for each epoch - moved inside batch loop
        
        for batch_idx, (x_input, y_true) in enumerate(train_loader):
            net.train()
            progress = batch_idx / len(train_loader)
            
            # Get actual batch size for this iteration
            actual_batch_size = x_input.shape[0]

            x_input = x_input.to(cfg.device)
            y_true = y_true.to(cfg.device)

            if cfg.model_name == 'RNN' or cfg.model_name == 'GRU':
                # Initialize hidden state with actual batch size
                h_state = net.init_hidden(actual_batch_size, cfg.device)
                y_pred, _h_state = net(x_input, h_state)
                h_state = _h_state.data
            else:
                y_pred = net(x_input)

            loss = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse_train_batch = loss.data.cpu().numpy()
            rmse_train_batch = np.sqrt(mse_train_batch)
            rmse_train += mse_train_batch
            if batch_idx % int(len(train_loader) / 5) == 0:
                print('epoch: {}  progress: {:.0f}%  loss: {:.3f}  rmse: {:.3f}'.format(epoch, progress * 100, loss, rmse_train_batch))
            cnt += 1
        rmse_train = np.sqrt(rmse_train / cnt)

        # validation
        net.eval()
        y_valid_pred_final = []
        y_valid_true = []
        rmse_valid = 0.0
        cnt = 0
        
        for x_input_valid, y_true_valid in valid_loader:
            x_input_valid = x_input_valid.to(cfg.device)
            y_true_valid = y_true_valid.to(cfg.device)
            
            if cfg.model_name == 'RNN' or cfg.model_name == 'GRU':
                # Initialize hidden state with actual batch size
                actual_batch_size = x_input_valid.shape[0]
                h_state = net.init_hidden(actual_batch_size, cfg.device)
                y_valid_pred, _h_state = net(x_input_valid, h_state)
                h_state = _h_state.data
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

        rmse_train_list.append(rmse_train)
        rmse_valid_list.append(rmse_valid)
        mae_valid_list.append(mae_valid)
        
        # save the best model
        if rmse_valid == np.min(rmse_valid_list):
            torch.save(net.state_dict(), cfg.model_save_pth)

        print('\n>>> epoch: {}  RMSE_train: {:.4f}  RMSE_valid: {:.4f} MAE_valid: {:.4f}\n'
              '    RMSE_valid_min: {:.4f}  MAE_valid_min: {:.4f}\n'
              .format(epoch, rmse_train, rmse_valid, mae_valid, np.min(rmse_valid_list), np.min(mae_valid_list)))


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
