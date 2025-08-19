# coding:utf-8

import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset


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
            31, 28, 31, 15-7-1  # 2025.1 ~ 2025.4.15, 扣除滑动窗口消失的样本数量
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
