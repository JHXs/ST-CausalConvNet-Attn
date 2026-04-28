# coding:utf-8

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn import metrics
import models
from utils import utils, report_tools
import config as cfg
import baseline_models


def configure_runtime_backend():
    """配置运行后端，优先保证评估可用性。"""
    is_rocm = torch.version.hip is not None
    if cfg.device == 'cuda' and is_rocm and getattr(cfg, 'disable_miopen', False):
        torch.backends.cudnn.enabled = False
        print('[Runtime] ROCm detected. MIOpen disabled (torch.backends.cudnn.enabled=False).')


def _calculate_single_output_metrics(y_true, y_pred):
    """Calculate metrics for one flattened output series."""
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))
    r2 = metrics.r2_score(y_true, y_pred)
    advanced_metrics = report_tools.calculate_advanced_metrics(y_true.reshape((-1, 1)), y_pred.reshape((-1, 1)))
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': advanced_metrics['MAPE'],
        'SMAPE': advanced_metrics['SMAPE'],
        'MASE': advanced_metrics['MASE'],
        'Coverage': advanced_metrics['Coverage'],
    }


def _reshape_by_prediction_variable(values):
    """Reshape flattened model outputs to [sample, horizon, prediction_variable]."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    elif values.ndim > 2:
        values = values.reshape((values.shape[0], -1))

    n_vars = cfg.get_num_prediction_variables() if hasattr(cfg, 'get_num_prediction_variables') else 1
    horizon = cfg.output_size
    expected_width = horizon * n_vars

    if values.shape[1] == expected_width:
        return values.reshape((values.shape[0], horizon, n_vars))
    if n_vars == 1:
        return values.reshape((values.shape[0], values.shape[1], 1))
    raise ValueError(
        f'Cannot reshape output with width {values.shape[1]} to '
        f'output_size={horizon} * prediction_variables={n_vars}. '
        'Regenerate y data with data_process.py after changing prediction_variables.'
    )


def _print_per_variable_metrics(y_true, y_pred):
    """Print metrics grouped by prediction variable while keeping the existing overall metrics."""
    y_true = _reshape_by_prediction_variable(y_true)
    y_pred = _reshape_by_prediction_variable(y_pred)

    variable_names = list(getattr(cfg, 'prediction_variables', []))
    if len(variable_names) < y_true.shape[2]:
        variable_names.extend([f'var_{idx + 1}' for idx in range(len(variable_names), y_true.shape[2])])

    print('Per Prediction-Variable Metrics:')
    print('{:<24} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}'.format(
        'Variable', 'RMSE', 'MAE', 'R2', 'MAPE%', 'SMAPE%', 'MASE', 'Coverage%'))
    for idx in range(y_true.shape[2]):
        var_true = y_true[:, :, idx].reshape(-1)
        var_pred = y_pred[:, :, idx].reshape(-1)
        var_metrics = _calculate_single_output_metrics(var_true, var_pred)
        print('{:<24} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.2f}'.format(
            variable_names[idx],
            var_metrics['RMSE'],
            var_metrics['MAE'],
            var_metrics['R2'],
            var_metrics['MAPE'],
            var_metrics['SMAPE'],
            var_metrics['MASE'],
            var_metrics['Coverage'],
        ))
    print()

def eval_gpu_memory(net, x_test, y_test, batch_size, plot=False):
    """使用GPU内存数据的评估函数，避免DataLoader的CPU-GPU传输开销"""
    print('\nStart evaluating with GPU memory data...\n')
    net.eval()
    criterion = nn.MSELoss().to(cfg.device)
    h_state = None
    
    # Initialize accumulators on device
    total_mse_valid = torch.tensor(0.0, device=cfg.device)
    total_mae_valid = torch.tensor(0.0, device=cfg.device)
    total_values_valid = 0
    
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
            total_values_valid += y_true_valid.numel()

            # Collect predictions for R2 calculation
            y_valid_pred_final.append(y_valid_pred.cpu())
            y_valid_true.append(y_true_valid.cpu())
    
    # Calculate final metrics
    rmse_valid = torch.sqrt(total_mse_valid / total_values_valid)
    mae_valid  = total_mae_valid / total_values_valid
    
    # Calculate R2 on CPU (requires sklearn)
    y_valid_pred_2d = torch.cat(y_valid_pred_final).numpy()
    y_valid_true_2d = torch.cat(y_valid_true).numpy()
    y_valid_pred_final = y_valid_pred_2d.reshape((-1, 1))
    y_valid_true = y_valid_true_2d.reshape((-1, 1))
    r2_valid = metrics.r2_score(y_valid_true, y_valid_pred_final)
    
    # Calculate advanced metrics
    advanced_metrics = report_tools.calculate_advanced_metrics(y_valid_true, y_valid_pred_final)
    
    print('\nTest Set Metrics:')
    print('RMSE_valid: {:.4f}  MAE_valid: {:.4f}  R2_valid: {:.4f}'.format(
        rmse_valid.item(), mae_valid.item(), r2_valid))
    print('MAPE: {:.4f}%  SMAPE: {:.4f}%  MASE: {:.4f}'.format(
        advanced_metrics['MAPE'], advanced_metrics['SMAPE'], advanced_metrics['MASE']))
    print('Coverage (95%): {:.2f}%\n'.format(advanced_metrics['Coverage']))
    _print_per_variable_metrics(y_valid_true_2d, y_valid_pred_2d)
    
    if plot:
        plots_file = report_tools.get_plot_directory('evaluation', cfg.model_name)
        report_tools.create_evaluation_plots(y_valid_true, y_valid_pred_final, advanced_metrics['Residuals'], plots_file)
    
    # Generate evaluation report
    if cfg.generate_report:
        print("\nGenerating evaluation report...")
        eval_results = [rmse_valid.item(), mae_valid.item(), r2_valid, 
                        advanced_metrics['MAPE'], advanced_metrics['SMAPE'], 
                        advanced_metrics['MASE'], advanced_metrics['Coverage']]
        
        # 创建虚拟的DataLoader用于报告生成
        test_dataset = torch.utils.data.TensorDataset(x_test.cpu(), y_test.cpu())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
        report_tools.generate_training_report(
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
    total_values_valid = 0
    
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
            total_values_valid += y_true_valid.numel()
            
            # Collect predictions for R2 calculation
            y_valid_pred_final.append(y_valid_pred.cpu())
            y_valid_true.append(y_true_valid.cpu())
    
    # Calculate final metrics
    rmse_valid = torch.sqrt(total_mse_valid / total_values_valid)
    mae_valid  = total_mae_valid / total_values_valid
    
    # Calculate R2 on CPU (requires sklearn)
    y_valid_pred_2d = torch.cat(y_valid_pred_final).numpy()
    y_valid_true_2d = torch.cat(y_valid_true).numpy()
    y_valid_pred_final = y_valid_pred_2d.reshape((-1, 1))
    y_valid_true = y_valid_true_2d.reshape((-1, 1))
    r2_valid = metrics.r2_score(y_valid_true, y_valid_pred_final)
    
    # Calculate advanced metrics
    advanced_metrics = report_tools.calculate_advanced_metrics(y_valid_true, y_valid_pred_final)
    
    print('\nTest Set Metrics:')
    print('RMSE_valid: {:.4f}  MAE_valid: {:.4f}  R2_valid: {:.4f}'.format(
        rmse_valid.item(), mae_valid.item(), r2_valid))
    print('MAPE: {:.4f}%  SMAPE: {:.4f}%  MASE: {:.4f}'.format(
        advanced_metrics['MAPE'], advanced_metrics['SMAPE'], advanced_metrics['MASE']))
    print('Coverage (95%): {:.2f}%\n'.format(advanced_metrics['Coverage']))
    _print_per_variable_metrics(y_valid_true_2d, y_valid_pred_2d)
    
    if plot:
        plots_file = report_tools.get_plot_directory('evaluation', cfg.model_name)
        report_tools.create_evaluation_plots(y_valid_true, y_valid_pred_final, advanced_metrics['Residuals'], plots_file)
    
    # Generate evaluation report
    if cfg.generate_report:
        print("\nGenerating evaluation report...")
        eval_results = [rmse_valid.item(), mae_valid.item(), r2_valid, 
                        advanced_metrics['MAPE'], advanced_metrics['SMAPE'], 
                        advanced_metrics['MASE'], advanced_metrics['Coverage']]
        
        report_tools.generate_training_report(
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
    configure_runtime_backend()

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
    model_output_size = cfg.get_model_output_size() if hasattr(cfg, 'get_model_output_size') else cfg.output_size

    # Generate model
    net = None
    if cfg.model_name == 'RNN':
        net = models.SimpleRNN(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=model_output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'GRU':
        net = models.SimpleGRU(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=model_output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'LSTM':
        net = models.SimpleLSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=model_output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'TCN':
        net = models.TCN(input_size=cfg.input_size, output_size=model_output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'TCN_Attention':
        net = models.TCN_Attention(input_size=cfg.input_size, output_size=model_output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'PatchTST':
        net = models.PatchTST(input_size=cfg.input_size, output_size=model_output_size, seq_len=cfg.seq_len,
                              patch_len=cfg.patchtst_patch_len, stride=cfg.patchtst_stride,
                              d_model=cfg.patchtst_d_model, d_ff=cfg.patchtst_d_ff,
                              n_heads=cfg.patchtst_n_heads, n_layers=cfg.patchtst_n_layers,
                              revin=cfg.patchtst_revin, dropout=cfg.dropout)
    elif cfg.model_name == 'STCN':
        net = models.STCN(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=model_output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'STCN_Attention':
        net = models.STCN_Attention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=model_output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout, attention_heads=cfg.attention_heads, use_rotary=cfg.use_rotary)
    elif cfg.model_name == 'ImprovedSTCN_Attention':
        net = models.ImprovedSTCN_Attention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=model_output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout, attention_heads=cfg.attention_heads)
    elif cfg.model_name == 'AdvancedSTCN_Attention':
        net = models.AdvancedSTCN_Attention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=model_output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout, attention_heads=cfg.attention_heads)
    elif cfg.model_name == 'STCN_LLAttention':
        net = models.STCN_LLAttention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=model_output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout, attention_heads=cfg.attention_heads, use_rotary=cfg.use_rotary, htype='weak', base=2)
    elif cfg.model_name == 'BiLSTM':
        net = baseline_models.BiLSTM(input_size=cfg.input_size, output_size=model_output_size)
    elif cfg.model_name == 'LSTM_GRU':
        net = baseline_models.LSTM_GRU(input_size=cfg.input_size, hidden_size_lstm=cfg.hidden_size, hidden_size_gru=cfg.hidden_size, output_size=model_output_size, dropout=cfg.dropout)
    elif cfg.model_name == 'BiLSTM_GRU':
        net = baseline_models.BiLSTM_GRU(input_size=cfg.input_size, hidden_size_lstm=cfg.hidden_size, hidden_size_gru=cfg.hidden_size, output_size=model_output_size, dropout=cfg.dropout)
    elif cfg.model_name == 'BiLSTM_CNN':
        net = baseline_models.BiLSTM_CNN(input_size=cfg.input_size, num_classes=model_output_size)
    elif cfg.model_name == 'LSTM_CNN':
        net = baseline_models.LSTM_CNN(input_size=cfg.input_size, output_size=model_output_size)
    elif cfg.model_name == 'ST_PatchTST':
        net = models.ST_PatchTST(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=model_output_size,
                                    seq_len=cfg.seq_len, dropout=cfg.dropout)
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
