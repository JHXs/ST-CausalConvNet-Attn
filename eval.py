# coding:utf-8

import sys
import numpy as np
import pandas as pd
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


def eval_gpu_memory(net, x_test, y_test, batch_size, plot=False):
    """使用GPU内存数据的评估函数，避免DataLoader的CPU-GPU传输开销"""
    print('\nStart evaluating with GPU memory data...\n')
    net.eval()
    criterion = nn.MSELoss().to(cfg.device)
    h_state = None
    
    # Initialize accumulators on device
    total_mse_valid = torch.tensor(0.0, device=cfg.device)
    total_mae_valid = torch.tensor(0.0, device=cfg.device)
    total_samples_valid = 0
    
    # Lists for R2 calculation (collected at the end)
    y_valid_pred_final = []
    y_valid_true = []
    
    n_test = len(x_test)
    
    with torch.no_grad():
        # 手动实现批次处理，数据已在GPU上
        for batch_idx in range(0, n_test, batch_size):
            end_idx = min(batch_idx + batch_size, n_test)
            
            x_input_valid = x_test[batch_idx:end_idx]
            y_true_valid = y_test[batch_idx:end_idx]
            
            if cfg.model_name == 'RNN' or cfg.model_name == 'GRU':
                actual_batch_size = x_input_valid.shape[0]
                h_state = net.init_hidden(actual_batch_size, cfg.device)
                y_valid_pred, _h_state = net(x_input_valid, h_state)
            else:
                y_valid_pred = net(x_input_valid)
            
            # Calculate metrics on GPU            
            total_mse_valid += (y_valid_pred - y_true_valid).pow(2).sum()
            total_mae_valid += (y_valid_pred - y_true_valid).abs().sum()
            total_samples_valid += x_input_valid.size(0)

            # Collect predictions for R2 calculation
            y_valid_pred_final.append(y_valid_pred.cpu())
            y_valid_true.append(y_true_valid.cpu())
    
    # Calculate final metrics
    rmse_valid = torch.sqrt(total_mse_valid / total_samples_valid)
    mae_valid  = total_mae_valid / total_samples_valid
    
    # Calculate R2 on CPU (requires sklearn)
    y_valid_pred_final = torch.cat(y_valid_pred_final).numpy().reshape((-1, 1))
    y_valid_true = torch.cat(y_valid_true).numpy().reshape((-1, 1))
    r2_valid = metrics.r2_score(y_valid_true, y_valid_pred_final)
    
    # Calculate advanced metrics
    advanced_metrics = utils.calculate_advanced_metrics(y_valid_true, y_valid_pred_final)
    
    print('\nTest Set Metrics:')
    print('RMSE_valid: {:.4f}  MAE_valid: {:.4f}  R2_valid: {:.4f}'.format(
        rmse_valid.item(), mae_valid.item(), r2_valid))
    print('MAPE: {:.4f}%  SMAPE: {:.4f}%  MASE: {:.4f}'.format(
        advanced_metrics['MAPE'], advanced_metrics['SMAPE'], advanced_metrics['MASE']))
    print('Coverage (95%): {:.2f}%\n'.format(advanced_metrics['Coverage']))
    
    if plot:
        plots_file = utils.get_plot_directory('evaluation', cfg.model_name)
        utils.create_evaluation_plots(y_valid_true, y_valid_pred_final, advanced_metrics['Residuals'], plots_file)
    
    # Generate evaluation report
    if cfg.generate_report:
        print("\nGenerating evaluation report...")
        eval_results = [rmse_valid.item(), mae_valid.item(), r2_valid, 
                        advanced_metrics['MAPE'], advanced_metrics['SMAPE'], 
                        advanced_metrics['MASE'], advanced_metrics['Coverage']]
        
        # 创建虚拟的DataLoader用于报告生成
        test_dataset = torch.utils.data.TensorDataset(x_test.cpu(), y_test.cpu())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
        utils.generate_training_report(
            cfg=cfg,
            model=None,
            train_loader=None,
            valid_loader=None,
            test_loader=test_loader,
            rmse_train_list=None,
            rmse_valid_list=None,
            mae_valid_list=None,
            train_losses=None,
            eval_results=eval_results
        )
    else:
        print("\nReport generation skipped (generate_report=False)")
    
    return rmse_valid.item(), mae_valid.item(), r2_valid, advanced_metrics['MAPE'], advanced_metrics['SMAPE'], advanced_metrics['MASE'], advanced_metrics['Coverage']


def eval(net, test_loader, plot=False):
    print('\nStart evaluating...\n')
    net.eval()
    criterion = nn.MSELoss().to(cfg.device)
    h_state = None
    
    # Initialize accumulators on device
    total_mse_valid = torch.tensor(0.0, device=cfg.device)
    total_mae_valid = torch.tensor(0.0, device=cfg.device)
    total_samples_valid = 0
    
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
            total_mse_valid += (y_valid_pred - y_true_valid).pow(2).sum()
            total_mae_valid += (y_valid_pred - y_true_valid).abs().sum()
            total_samples_valid += x_input_valid.size(0)
            
            # Collect predictions for R2 calculation
            y_valid_pred_final.append(y_valid_pred.cpu())
            y_valid_true.append(y_true_valid.cpu())
    
    # Calculate final metrics
    rmse_valid = torch.sqrt(total_mse_valid / total_samples_valid)
    mae_valid  = total_mae_valid / total_samples_valid
    
    # Calculate R2 on CPU (requires sklearn)
    y_valid_pred_final = torch.cat(y_valid_pred_final).numpy().reshape((-1, 1))
    y_valid_true = torch.cat(y_valid_true).numpy().reshape((-1, 1))
    r2_valid = metrics.r2_score(y_valid_true, y_valid_pred_final)
    
    # Calculate advanced metrics
    advanced_metrics = utils.calculate_advanced_metrics(y_valid_true, y_valid_pred_final)
    
    print('\nTest Set Metrics:')
    print('RMSE_valid: {:.4f}  MAE_valid: {:.4f}  R2_valid: {:.4f}'.format(
        rmse_valid.item(), mae_valid.item(), r2_valid))
    print('MAPE: {:.4f}%  SMAPE: {:.4f}%  MASE: {:.4f}'.format(
        advanced_metrics['MAPE'], advanced_metrics['SMAPE'], advanced_metrics['MASE']))
    print('Coverage (95%): {:.2f}%\n'.format(advanced_metrics['Coverage']))
    
    if plot:
        plots_file = utils.get_plot_directory('evaluation', cfg.model_name)
        utils.create_evaluation_plots(y_valid_true, y_valid_pred_final, advanced_metrics['Residuals'], plots_file)
    
    # Generate evaluation report
    if cfg.generate_report:
        print("\nGenerating evaluation report...")
        eval_results = [rmse_valid.item(), mae_valid.item(), r2_valid, 
                        advanced_metrics['MAPE'], advanced_metrics['SMAPE'], 
                        advanced_metrics['MASE'], advanced_metrics['Coverage']]
        
        utils.generate_training_report(
            cfg=cfg,
            model=None,
            train_loader=None,
            valid_loader=None,
            test_loader=test_loader,
            rmse_train_list=None,
            rmse_valid_list=None,
            mae_valid_list=None,
            train_losses=None,
            eval_results=eval_results
        )
    else:
        print("\nReport generation skipped (generate_report=False)")
    
    return rmse_valid.item(), mae_valid.item(), r2_valid, advanced_metrics['MAPE'], advanced_metrics['SMAPE'], advanced_metrics['MASE'], advanced_metrics['Coverage']


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

    # Load model parameters
    net = net.to(cfg.device)
    net.load_state_dict(torch.load(cfg.model_save_pth, map_location=cfg.device))
    print(utils.get_param_number(net=net))

    # Evaluation - 根据数据加载方式选择评估函数
    if data_to_gpu_memory:
        eval_gpu_memory(net, x_test, y_test, batch_size, cfg.plt)
    else:
        eval(net, test_loader, cfg.plt)


if __name__ == '__main__':
    main()
