# coding:utf-8

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn import metrics
import models
import utils
import config as cfg


def train_gpu_memory(net, x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, plot=False):
    """使用GPU内存数据的训练函数，避免DataLoader的CPU-GPU传输开销"""
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
    best_epoch_idx = 0
    patience_counter = 0
    early_stop_epoch = None
    
    # 获取数据集大小
    n_train = len(x_train)
    n_valid = len(x_valid)
    
    for epoch in range(1, cfg.n_epochs + 1):
        # Training phase
        net.train()
        rmse_train = 0.0
        cnt = 0
        
        # 手动实现批次处理，数据已在GPU上
        indices = torch.randperm(n_train)
        for batch_idx in range(0, n_train, batch_size):
            end_idx = min(batch_idx + batch_size, n_train)
            batch_indices = indices[batch_idx:end_idx]
            
            x_input = x_train[batch_indices]
            y_true = y_train[batch_indices]

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
            
            if batch_idx % round(n_train / batch_size / 5) == 0:
                progress = batch_idx / n_train
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
            for batch_idx in range(0, n_valid, batch_size):
                end_idx = min(batch_idx + batch_size, n_valid)
                
                x_input_valid = x_valid[batch_idx:end_idx]
                y_true_valid = y_valid[batch_idx:end_idx]
                
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
                best_epoch_idx = epoch
                patience_counter = 0
                # 保存最佳模型
                torch.save(net.state_dict(), cfg.model_save_pth)
            else:
                patience_counter += 1
                if patience_counter >= cfg.es_patience:
                    print(f"Early stopping triggered after {epoch} epochs!")
                    early_stop_epoch = epoch
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
    
    if plot:
        plots_file = utils.get_plot_directory('training', cfg.model_name)
        utils.create_training_plots(rmse_train_list, rmse_valid_list, mae_valid_list, train_losses, plots_file)
    
    # Generate training report
    if cfg.generate_report:
        print("\nGenerating training report...")
        # 创建虚拟的DataLoader用于报告生成
        train_dataset = torch.utils.data.TensorDataset(x_train.cpu(), y_train.cpu())
        valid_dataset = torch.utils.data.TensorDataset(x_valid.cpu(), y_valid.cpu())
        test_dataset = torch.utils.data.TensorDataset(x_test.cpu(), y_test.cpu())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
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

    # 返回训练元信息，便于 sweep 汇总
    return {
        'elapsed_seconds': total_time,
        'best_epoch': best_epoch_idx,
        'early_stop_epoch': early_stop_epoch,
        'final_rmse_train': rmse_train_list[-1] if rmse_train_list else None,
        'final_rmse_valid': rmse_valid_list[-1] if rmse_valid_list else None,
    }


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
    best_epoch_idx = 0
    patience_counter = 0
    early_stop_epoch = None

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
                best_epoch_idx = epoch
                patience_counter = 0
                # 保存最佳模型
                torch.save(net.state_dict(), cfg.model_save_pth)
            else:
                patience_counter += 1
                if patience_counter >= cfg.es_patience:
                    print(f"Early stopping triggered after {epoch} epochs!")
                    early_stop_epoch = epoch
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

    # 返回训练元信息，便于 sweep 汇总
    return {
        'elapsed_seconds': total_time,
        'best_epoch': best_epoch_idx,
        'early_stop_epoch': early_stop_epoch,
        'final_rmse_train': rmse_train_list[-1] if rmse_train_list else None,
        'final_rmse_valid': rmse_valid_list[-1] if rmse_valid_list else None,
    }


def main():
    # Hyper Parameters
    cfg.print_params()
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)

    # Load data - 根据配置选择数据加载方式
    if cfg.data_to_gpu_memory and torch.cuda.is_available():
        print('\nLoading data to GPU memory...\n')
        x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size = utils.load_data(
            f_x=cfg.f_x, f_y=cfg.f_y, batch_size=cfg.batch_size, data_to_gpu_memory=cfg.data_to_gpu_memory, device=cfg.device)
        data_to_gpu_memory = True
    else:
        print('\nLoading data with DataLoader...\n')
        train_loader, valid_loader, test_loader = utils.load_data(f_x=cfg.f_x, f_y=cfg.f_y, batch_size=cfg.batch_size)
        data_to_gpu_memory = False

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
    elif cfg.model_name == 'ImprovedSTCN_Attention':
        net = models.ImprovedSTCN_Attention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                             num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                             attention_heads=cfg.attention_heads)
    elif cfg.model_name == 'AdvancedSTCN_Attention':
        net = models.AdvancedSTCN_Attention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                             num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                             attention_heads=cfg.attention_heads)
    elif cfg.model_name == 'STCN_LLAttention':
        net = models.STCN_LLAttention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                     num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                     attention_heads=cfg.attention_heads, use_rotary=cfg.use_rotary, htype='weak', base=2)
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, net))
    net = net.to(cfg.device)
    # sys.exit(0)

    # Training - 根据数据加载方式选择训练函数
    if data_to_gpu_memory:
        print('\nStart training with GPU memory data...\n')
        train_gpu_memory(net, x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, cfg.plt)
    else:
        print('\nStart training with DataLoader...\n')
        train(net, train_loader, valid_loader, test_loader, cfg.plt)


if __name__ == '__main__':
    main()
