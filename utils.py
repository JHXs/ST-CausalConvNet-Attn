# coding:utf-8

import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats


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
    
    # 1. Predicted vs True scatter plot
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    plt.grid(True, alpha=0.3)
    
    # 2. Time series comparison (first 200 points)
    ax2 = plt.subplot(2, 3, 2)
    time_points = range(min(200, len(y_true)))
    plt.plot(time_points, y_true[:200], 'b-', label='True', alpha=0.7)
    plt.plot(time_points, y_pred[:200], 'r-', label='Predicted', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Time Series Comparison (First 200 points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Residual histogram
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(residuals.flatten(), bins=50, density=True, alpha=0.7, color='green')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    ax4 = plt.subplot(2, 3, 4)
    stats.probplot(residuals.flatten(), dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True, alpha=0.3)
    
    # 5. Residuals vs Predicted
    ax5 = plt.subplot(2, 3, 5)
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # 6. Error distribution over time
    ax6 = plt.subplot(2, 3, 6)
    absolute_errors = np.abs(residuals.flatten())
    time_points = range(min(200, len(absolute_errors)))
    plt.plot(time_points, absolute_errors[:200], 'purple', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error Over Time (First 200 points)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plots saved to {save_path}')
    else:
        plt.show()

def create_training_plots(rmse_train_list, rmse_valid_list, mae_valid_list, train_losses, save_path=None):
    """Create training progress plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Training loss per batch
    ax1 = axes[0, 0]
    ax1.plot(train_losses, alpha=0.7, color='blue')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss per Batch')
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE training and validation per epoch
    ax2 = axes[0, 1]
    epochs = range(1, len(rmse_train_list) + 1)
    ax2.plot(epochs, rmse_train_list, 'b-', label='Train RMSE', alpha=0.7)
    ax2.plot(epochs, rmse_valid_list, 'r-', label='Validation RMSE', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE per Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MAE validation per epoch
    ax3 = axes[1, 0]
    ax3.plot(epochs, mae_valid_list, 'g-', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE')
    ax3.set_title('Validation MAE per Epoch')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss distribution (final epoch losses)
    ax4 = axes[1, 1]
    if len(train_losses) > 0:
        # Get recent losses for better visualization
        recent_losses = train_losses[-1000:] if len(train_losses) > 1000 else train_losses
        ax4.hist(recent_losses, bins=50, alpha=0.7, color='orange', density=True)
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Density')
        ax4.set_title('Loss Distribution (Recent Batches)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training plots saved to {save_path}')
    else:
        plt.show()
