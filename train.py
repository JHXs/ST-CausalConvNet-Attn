# coding:utf-8

import sys
import time
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
    # 记录训练开始时间
    start_time = time.time()
    
    rmse_train_list = []
    rmse_valid_list = []
    mae_valid_list = []
    train_losses = []
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss().to(cfg.device)
    h_state = None
    
    # 学习率调度器
    if cfg.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.lr_factor, 
            patience=cfg.lr_patience, min_lr=cfg.min_lr
        )
    
    # 早停机制
    best_valid_loss = float('inf')
    patience_counter = 0

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
            
            train_losses.append(loss.item())
            
            if batch_idx % round(len(train_loader) / 5) == 0:
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

        print('\n>>> epoch: {}  RMSE_train: {:.4f}  RMSE_valid: {:.4f} MAE_valid: {:.4f}\n'
              '    RMSE_valid_min: {:.4f}  MAE_valid_min: {:.4f}'
              .format(epoch, rmse_train_cpu, rmse_valid_cpu, mae_valid_cpu, 
                     np.min(rmse_valid_list), np.min(mae_valid_list)))        
        
        # 学习率调度
        if cfg.lr_scheduler:
            scheduler.step(rmse_valid_cpu)
            print("    LR: {:.6f}\n".format(optimizer.param_groups[0]['lr']))
        
        # 早停检查
        if cfg.early_stopping:
            if rmse_valid_cpu < best_valid_loss:
                best_valid_loss = rmse_valid_cpu
                patience_counter = 0
                # 保存最佳模型
                torch.save(net.state_dict(), cfg.model_save_pth)
                # print(f"  -> New best model saved! (RMSE: {rmse_valid_cpu:.4f})\n")
            else:
                patience_counter += 1
                if patience_counter >= cfg.es_patience:
                    print(f"Early stopping triggered after {epoch} epochs!")
                    break
        else:
            # 原有的模型保存逻辑
            if rmse_valid_cpu == np.min(rmse_valid_list):
                torch.save(net.state_dict(), cfg.model_save_pth)
    
    # 计算训练总耗时
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"Training completed in {hours}h {minutes}m {seconds:.2f}s")
    
    # Calculate statistics for comprehensive report
    best_epoch = np.argmin(rmse_valid_list) + 1 if rmse_valid_list else 0
    avg_batch_time = total_time / len(train_loader) / cfg.n_epochs if len(train_losses) > 0 else 0
    estimated_total_batches = len(train_loader) * cfg.n_epochs
    
    if plot:
        plots_file = utils.get_plot_directory('training', cfg.model_name)
        utils.create_training_plots(rmse_train_list, rmse_valid_list, mae_valid_list, train_losses, plots_file)
    
    # Generate training report
    if cfg.generate_report:
        print("\nGenerating training report...")
        utils.generate_training_report(
            cfg=cfg,
            model=net,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            rmse_train_list=rmse_train_list,
            rmse_valid_list=rmse_valid_list,
            mae_valid_list=mae_valid_list,
            train_losses=train_losses
        )
    else:
        print("\nReport generation skipped (generate_report=False)")


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
                                             num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                             attention_heads=cfg.attention_heads, use_rotary=cfg.use_rotary)
    elif cfg.model_name == 'STCN_LLAttention':
        net = models.STCN_LLAttention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                     num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                     attention_heads=cfg.attention_heads, use_rotary=cfg.use_rotary, htype='weak', base=2)
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, net))
    net = net.to(cfg.device)
    # sys.exit(0)

    # Training
    print('\nStart training...\n')
    train(net, train_loader, valid_loader, test_loader, cfg.plt)


if __name__ == '__main__':
    main()
