# coding:utf-8

import sys
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from datetime import datetime

# 当前运行的标识，用于将报告与图表落入独立子目录
CURRENT_RUN_ID = None

def set_run_id(run_id: str):
    global CURRENT_RUN_ID
    CURRENT_RUN_ID = run_id


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_ids_for_tvt(hz):
    train_ids = []
    valid_ids = []
    test_ids = []
    if hz == 0:
        days_in_months = [31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30-1]  # May to April
    else:
        days_in_months = [
            31, 29, 30, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2020
            31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2021
            31, 28, 29, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2022
            31, 28, 31, 30, 31, 30, 1, 31, 30, 31, 30, 31,  # 2023
            31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 2024
            31, 28, 31, 15-7-1  # 2025.1 ~ 2025.4.15, 扣除滑动窗口消失的样本数量(time_step+预测多少时长)
        ]  # 2020.1.1 - 2025.4.15
    start_id = 0
    for i in range(len(days_in_months)):
        days = days_in_months[i]
        split_id_0 = start_id
        split_id_1 = start_id + int(days * 24 * 0.6)
        split_id_2 = start_id + int(days * 24 * 0.8)
        split_id_3 = start_id + int(days * 24)
        train_ids.extend(np.arange(split_id_0, split_id_1, 1))
        valid_ids.extend(np.arange(split_id_1, split_id_2, 1))
        test_ids.extend(np.arange(split_id_2, split_id_3, 1))
        start_id += int(days * 24)
    return train_ids, valid_ids, test_ids


def load_data(f_x, f_y, batch_size=32):
    x = load_pickle(f_x)
    y = load_pickle(f_y)
    y = np.array(y[:, np.newaxis])
    if len(x.shape) == 3:
        ss = preprocessing.StandardScaler()
        for i in range(x.shape[-1]):
            ss.fit(x[:, :, i])
            x[:, :, i] = ss.transform(x[:, :, i])
    if len(y) > 10000:
        train_ids, valid_ids, test_ids = get_ids_for_tvt(1)
    else:
        train_ids, valid_ids, test_ids = get_ids_for_tvt(0)
    x_train = x[train_ids]
    y_train = y[train_ids]
    x_valid = x[valid_ids]
    y_valid = y[valid_ids]
    x_test = x[test_ids]
    y_test = y[test_ids]

    print('x_shape: {}  y_shape: {}\nx_train_shape: {}  y_train_shape: {}  x_valid_shape: {}  y_valid_shape: {}  x_test_shape: {}  y_test_shape: {}\n'
          .format(x.shape, y.shape, x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape))
    
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_valid_tensor = torch.FloatTensor(x_valid)
    y_valid_tensor = torch.FloatTensor(y_valid)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, valid_loader, test_loader

def get_param_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

def replicate_to_hourly(df):
    """
    将每日空气质量数据均匀复制为每小时数据，每天的值在该天的每个小时保持一致。
    
    参数:
    df (pd.DataFrame): 包含每日空气质量数据的 DataFrame，需包括列 
                       'time', 'AQI', 'SO2', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3_8h'。
                       'time' 列应包含可由 pd.to_datetime 解析的日期字符串。
    
    返回:
    pd.DataFrame: 新的 DataFrame，包含每小时数据，每天的每个小时具有该天的相同值。
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()    
    if 'time' not in df.columns:
        raise ValueError("The DataFrame must contain a 'time' column.")      
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])    
    df.set_index('time', inplace=True) # Set 'time' as the index
    df.sort_index(inplace=True)  
    start_date = df.index.min()
    end_date = df.index.max()
    hours = pd.date_range(start=start_date, end=end_date + pd.Timedelta(days=1), freq='h') # Create hourly time index from start to end + 1 day  
    hourly_index = hours[hours.normalize().isin(df.index)]   # 找出 df 里确实存在的那些小时
    dates = hourly_index.normalize()  # 获取每小时对应的日期
    # dates = [d for d in dates if d in df.index]
    hourly_df = df.loc[dates, :].copy()  # 将每日数据复制到每小时
    hourly_index.name = 'time'
    hourly_df.index = hourly_index
    # hourly_df = hourly_df.reset_index().rename(columns={'index': 'time'})
    
    return hourly_df


def calculate_advanced_metrics(y_true, y_pred):
    """Calculate advanced evaluation metrics"""
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # SMAPE
    smape = np.mean(np.abs(y_true - y_pred) / 
                   (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # MASE (using naive forecast as baseline)
    naive_forecast = np.roll(y_true, 1)
    naive_forecast[0] = naive_forecast[1]  # Handle first element
    mae_naive = np.mean(np.abs(y_true - naive_forecast))
    mase = np.mean(np.abs(y_true - y_pred)) / mae_naive if mae_naive > 0 else 0
    
    # Coverage (assuming 95% prediction interval)
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)
    lower_bound = y_pred - 1.96 * std_residuals
    upper_bound = y_pred + 1.96 * std_residuals
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
    
    return {
        'MAPE': mape,
        'SMAPE': smape,
        'MASE': mase,
        'Coverage': coverage,
        'Residuals': residuals
    }

def create_evaluation_plots(y_true, y_pred, residuals=None, save_path=None):
    """Create comprehensive evaluation plots"""
    if residuals is None:
        residuals = y_true - y_pred
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    _create_scatter_plot(y_true, y_pred)
    _create_time_series_plot(y_true[:200], y_pred[:200])
    _create_residual_histogram(residuals)
    _create_qq_plot(residuals)
    _create_residual_vs_predicted(y_pred, residuals)
    _create_error_over_time_plot(np.abs(residuals.flatten()))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plots saved to {save_path}')
    else:
        plt.show()
    plt.close(fig)


def _create_scatter_plot(y_true, y_pred):
    """Create predicted vs true scatter plot"""
    plt.subplot(2, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    plt.grid(True, alpha=0.3)


def _create_time_series_plot(y_true, y_pred):
    """Create time series comparison plot"""
    plt.subplot(2, 3, 2)
    time_points = range(len(y_true))
    plt.plot(time_points, y_true, 'b-', label='True', alpha=0.7)
    plt.plot(time_points, y_pred, 'r-', label='Predicted', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Time Series Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)


def _create_residual_histogram(residuals):
    """Create residual distribution histogram"""
    plt.subplot(2, 3, 3)
    plt.hist(residuals.flatten(), bins=50, density=True, alpha=0.7, color='green')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)


def _create_qq_plot(residuals):
    """Create Q-Q plot for residuals"""
    plt.subplot(2, 3, 4)
    stats.probplot(residuals.flatten(), dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True, alpha=0.3)


def _create_residual_vs_predicted(y_pred, residuals):
    """Create residuals vs predicted values plot"""
    plt.subplot(2, 3, 5)
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)


def _create_error_over_time_plot(absolute_errors):
    """Create error distribution over time plot"""
    plt.subplot(2, 3, 6)
    time_points = range(min(200, len(absolute_errors)))
    plt.plot(time_points, absolute_errors[:200], 'purple', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error Over Time')
    plt.grid(True, alpha=0.3)


def create_training_plots(rmse_train_list, rmse_valid_list, mae_valid_list, train_losses, save_path=None):
    """Create training progress plots"""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    _plot_training_loss(train_losses)
    _plot_rmse_comparison(rmse_train_list, rmse_valid_list)
    _plot_validation_mae(mae_valid_list)
    _plot_loss_distribution(train_losses)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training plots saved to {save_path}')
    else:
        plt.show()
    plt.close(fig)


def _plot_training_loss(train_losses):
    """Create training loss per batch plot"""
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, alpha=0.7, color='blue')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.grid(True, alpha=0.3)


def _plot_rmse_comparison(rmse_train_list, rmse_valid_list):
    """Create RMSE training and validation comparison plot"""
    plt.subplot(2, 2, 2)
    epochs = range(1, len(rmse_train_list) + 1)
    plt.plot(epochs, rmse_train_list, 'b-', label='Train RMSE', alpha=0.7)
    plt.plot(epochs, rmse_valid_list, 'r-', label='Validation RMSE', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE per Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)


def _plot_validation_mae(mae_valid_list):
    """Create validation MAE per epoch plot"""
    plt.subplot(2, 2, 3)
    epochs = range(1, len(mae_valid_list) + 1)
    plt.plot(epochs, mae_valid_list, 'g-', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE per Epoch')
    plt.grid(True, alpha=0.3)


def _plot_loss_distribution(train_losses):
    """Create loss distribution plot"""
    plt.subplot(2, 2, 4)
    if len(train_losses) > 0:
        recent_losses = train_losses[-1000:] if len(train_losses) > 1000 else train_losses
        plt.hist(recent_losses, bins=50, alpha=0.7, color='orange', density=True)
        plt.xlabel('Loss')
        plt.ylabel('Density')
        plt.title('Loss Distribution (Recent Batches)')
        plt.grid(True, alpha=0.3)


def generate_training_report(cfg, model, train_loader, valid_loader, test_loader, 
                           rmse_train_list, rmse_valid_list, mae_valid_list, train_losses,
                           eval_results=None, report_path=None):
    """Generate comprehensive training validation report"""
    from datetime import datetime
    
    # Initialize report context
    context = _initialize_report_context(cfg, model, rmse_train_list, rmse_valid_list, 
                                       mae_valid_list, train_losses, train_loader, eval_results)
    
    # Generate report file path
    report_file = _get_report_file_path(report_path, cfg.model_name, context['prefix'])
    
    # Generate report content
    report_content = _generate_report_content(cfg, context, eval_results)
    
    # Save and print report
    _save_and_print_report(report_file, report_content, context)
    
    return report_file


def _initialize_report_context(cfg, model, rmse_train_list, rmse_valid_list, mae_valid_list, 
                              train_losses, train_loader, eval_results):
    """Initialize report context with common data"""
    from datetime import datetime
    
    # Determine mode and basic stats
    is_training = bool(rmse_valid_list and len(rmse_valid_list) > 0)
    
    if is_training:
        best_epoch = np.argmin(rmse_valid_list) + 1
        best_rmse = rmse_valid_list[best_epoch - 1]
        final_rmse_train = rmse_train_list[-1] if rmse_train_list else 0
        final_mae_valid = mae_valid_list[-1] if mae_valid_list else 0
        total_epochs = len(rmse_train_list)
        prefix = "training"
    else:
        best_epoch = 1
        best_rmse = 0
        final_rmse_train = 0
        final_mae_valid = 0
        total_epochs = 0
        prefix = "evaluation"
    
    # Get model statistics
    model_stats = None
    if model is not None:
        total_params, trainable_params = get_param_number(model)
        model_size_mb = (total_params * 4.0) / (1024.0 * 1024.0)
        model_stats = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb
        }
    
    return {
        'date_str': datetime.now().strftime('%m%d'),
        'time_str': datetime.now().strftime('%H%M%S'),
        'model_name': cfg.model_name,
        'is_training': is_training,
        'best_epoch': best_epoch,
        'best_rmse': best_rmse,
        'final_rmse_train': final_rmse_train,
        'final_mae_valid': final_mae_valid,
        'total_epochs': total_epochs,
        'prefix': prefix,
        'model_stats': model_stats,
        'avg_batch_time': np.mean(train_losses) if train_losses else 0,
        'estimated_total_batches': total_epochs * len(train_loader) if train_loader else 0,
        'rmse_train_list': rmse_train_list,
        'rmse_valid_list': rmse_valid_list,
        'mae_valid_list': mae_valid_list,
        'eval_results': eval_results
    }


def _get_report_file_path(report_path, model_name, prefix):
    """Generate report file path"""
    import os
    from datetime import datetime
    
    if report_path is None:
        report_path = './reports'
    os.makedirs(report_path, exist_ok=True)
    
    date_str = datetime.now().strftime('%m%d')
    time_str = datetime.now().strftime('%H%M%S')
    date_dir = os.path.join(report_path, date_str)
    # 如果存在 run_id，则在日期目录下创建 run_id 子目录
    if CURRENT_RUN_ID:
        date_dir = os.path.join(date_dir, CURRENT_RUN_ID)
    os.makedirs(date_dir, exist_ok=True)
    
    return os.path.join(date_dir, f'{prefix}_{date_str}_{time_str}_{model_name}.txt')


def _generate_report_content(cfg, context, eval_results):
    """Generate report content based on context"""
    content_parts = []
    
    # Header
    content_parts.append(f"""
=======================================================
TRAINING VALIDATION REPORT
==========================
=======================================================

Model: {context['model_name']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=======================================================

""")
    
    # Model configuration
    content_parts.append(_generate_config_section(cfg))
    
    # Training performance (if applicable)
    if context['is_training']:
        content_parts.append(_generate_training_performance_section(cfg, context))
        content_parts.append(_generate_model_statistics_section(context))
        content_parts.append(_generate_detailed_metrics_section(context))
        content_parts.append(_generate_convergence_analysis_section(context))
    
    # Evaluation results
    if eval_results:
        content_parts.append(_generate_evaluation_results_section(eval_results))
    
    # Footer
    content_parts.append(f"""
=======================================================
END OF REPORT - Model: {context['model_name']}
=======================================================

""")
    
    return ''.join(content_parts)


def _generate_config_section(cfg):
    """Generate model configuration section"""
    config_str = f"""1. MODEL CONFIGURATION
-------------------------------------------------------
Input Size: {cfg.input_size}
Hidden Size: {cfg.hidden_size}
Output Size: {cfg.output_size}
Number of Layers: {cfg.num_layers}
Levels (for TCN): {cfg.levels}
Kernel Size (for TCN): {cfg.kernel_size}
Dropout (for TCN): {cfg.dropout}
Batch Size: {cfg.batch_size}
Learning Rate: {cfg.lr}
Number of Epochs: {cfg.n_epochs}
Early Stopping: {cfg.early_stopping} (Patience: {cfg.es_patience})
Learning Rate Scheduler: {cfg.lr_scheduler}
Prediction Variables: {cfg.prediction_variables}

"""
    
    if cfg.model_name in ['STCN', 'STCN_Attention', 'ImprovedSTCN_Attention', 'AdvancedSTCN_Attention', 'STCN_LLAttention']:
        config_str += f"""Input Channels: {cfg.in_channels}
Attention Heads: {cfg.attention_heads}
Use Rotary: {cfg.use_rotary}

"""
    
    return config_str


def _generate_training_performance_section(cfg, context):
    """Generate training performance section"""
    return f"""2. TRAINING PERFORMANCE
-------------------------------------------------------
Best Epoch: {context['best_epoch']}
Best Validation RMSE: {context['best_rmse']:.6f}
Final Training RMSE: {context['final_rmse_train']:.6f}
Final Validation MAE: {context['final_mae_valid']:.6f}

Training Progress Summary:
- Total training batches processed: {context['estimated_total_batches']}
- Average batch loss: {context['avg_batch_time']:.6f}
- Training improvement from first to last epoch: {context['rmse_train_list'][0] - context['rmse_train_list'][-1]:.6f}
- Validation improvement from first to last epoch: {context['rmse_valid_list'][0] - context['rmse_valid_list'][-1]:.6f}

"""


def _generate_model_statistics_section(context):
    """Generate model statistics section"""
    if context['model_stats'] is None:
        return ""
    
    stats = context['model_stats']
    return f"""3. MODEL STATISTICS
-------------------------------------------------------
Total Parameters: {stats['total_params']:,}
Trainable Parameters: {stats['trainable_params']:,}
Non-trainable Parameters: {stats['total_params'] - stats['trainable_params']:,}
Model Size: {stats['model_size_mb']:.2f} MB (assuming 4 bytes per parameter)

"""


def _generate_detailed_metrics_section(context):
    """Generate detailed metrics section"""
    content = """4. DETAILED METRICS BY EPOCH
-------------------------------------------------------
Epoch      Train RMSE   Valid RMSE   Valid MAE   Improvement
-------------------------------------------------------
"""
    
    for i in range(0, min(context['total_epochs'], 10)):
        improvement = ""
        if i > 0:
            improvement = f"{context['rmse_valid_list'][i-1] - context['rmse_valid_list'][i]:+6f}"
        content += f"{i+1:5d} {context['rmse_train_list'][i]:12.6f} {context['rmse_valid_list'][i]:12.6f} {context['mae_valid_list'][i]:12.6f} {improvement:12}\n"
    
    if context['total_epochs'] > 10:
        content += f"... (showing first 10 of {context['total_epochs']} epochs) ...\n"
    
    # Summary statistics
    content += f"""
5. SUMMARY STATISTICS
-------------------------------------------------------
Training RMSE Statistics:
  - Mean: {np.mean(context['rmse_train_list']):.6f}
  - Std: {np.std(context['rmse_train_list']):.6f}
  - Min: {np.min(context['rmse_train_list']):.6f}
  - Max: {np.max(context['rmse_train_list']):.6f}

Validation RMSE Statistics:
  - Mean: {np.mean(context['rmse_valid_list']):.6f}
  - Std: {np.std(context['rmse_valid_list']):.6f}
  - Min: {np.min(context['rmse_valid_list']):.6f}
  - Max: {np.max(context['rmse_valid_list']):.6f}

Validation MAE Statistics:
  - Mean: {np.mean(context['mae_valid_list']):.6f}
  - Std: {np.std(context['mae_valid_list']):.6f}
  - Min: {np.min(context['mae_valid_list']):.6f}
  - Max: {np.max(context['mae_valid_list']):.6f}

"""
    
    return content


def _generate_convergence_analysis_section(context):
    """Generate convergence analysis section"""
    last_10_rmse = context['rmse_valid_list'][-10:]
    convergence_slope = np.polyfit(range(len(last_10_rmse)), last_10_rmse, 1)[0]
    
    if abs(convergence_slope) < 1e-6:
        convergence_status = "CONVERGED"
    elif convergence_slope < 0:
        convergence_status = "STILL IMPROVING"
    else:
        convergence_status = "POTENTIAL OVERFITTING"
    
    return f"""6. CONVERGENCE ANALYSIS
-------------------------------------------------------
Convergence Status: {convergence_status}
Last 10 epochs RMSE trend: {convergence_slope:.8f} (slope)
Final 10 epochs RMSE range: {np.max(last_10_rmse) - np.min(last_10_rmse):.6f}

7. FINAL TRAINING METRICS
-------------------------------------------------------
Final Training RMSE: {context['rmse_train_list'][-1]:.6f}
Final Validation RMSE: {context['rmse_valid_list'][-1]:.6f}
Final Validation MAE: {context['mae_valid_list'][-1]:.6f}
Best Validation RMSE: {context['best_rmse']:.6f} at epoch {context['best_epoch']}
Total Training Time Improvement: {context['rmse_train_list'][0] - context['rmse_train_list'][-1]:.6f}
Training Efficiency: {(context['rmse_train_list'][0] - context['rmse_train_list'][-1]) / context['total_epochs']:.6f} RMSE improvement per epoch

"""


def _generate_evaluation_results_section(eval_results):
    """Generate evaluation results section"""
    return f"""8. TEST SET EVALUATION RESULTS
-------------------------------------------------------
RMSE: {eval_results[0]:.6f}
MAE: {eval_results[1]:.6f}
R²: {eval_results[2]:.6f}
MAPE: {eval_results[3]:.6f}%
SMAPE: {eval_results[4]:.6f}%
MASE: {eval_results[5]:.6f}
Coverage (95%): {eval_results[6]:.2f}%

"""


def _save_and_print_report(report_file, content, context):
    """Save report to file and print summary"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if context['is_training']:
        print(f"\nTraining report generated: {report_file}")
        print(f"Best validation RMSE: {context['best_rmse']:.6f} at epoch {context['best_epoch']}")
        if context['model_stats']:
            print(f"Total parameters: {context['model_stats']['total_params']:,}")
    else:
        print(f"\nEvaluation report generated: {report_file}")
        if context.get('eval_results'):
            print(f"Test RMSE: {context['eval_results'][0]:.6f}")
            print(f"Test R²: {context['eval_results'][2]:.6f}")
        if context['model_stats']:
            print(f"Total parameters: {context['model_stats']['total_params']:,}")


def get_plot_directory(plot_type, model_name):
    """Generate plot file path with optional run_id subdirectory"""
    from datetime import datetime
    date_str = datetime.now().strftime('%m%d')
    time_str = datetime.now().strftime('%H%M%S')
    base_dir = os.path.join('./reports', date_str)
    # 在 plots 之前插入 run_id 子目录（如已设置）
    if CURRENT_RUN_ID:
        base_dir = os.path.join(base_dir, CURRENT_RUN_ID)
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plots_file = os.path.join(plots_dir, f'{plot_type}_{date_str}_{time_str}_{model_name}.png')
    return plots_file
