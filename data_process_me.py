# coding:utf-8

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time, datetime
import utils


def main():
    # 空气质量数据插值到小时级
    df_airq = pd.read_csv('./data/hezhou_air_data/20200101-20250415最大值实况-十四五城市日均值.csv')
    hourly_df = utils.replicate_to_hourly(df_airq)
    print(hourly_df.head(24))
    # hourly_df.to_csv(f'./data/hezhou_air_data/aqi.csv', index=True)

    # 空气质量与气象数据聚合
    df_weather = pd.read_csv('./data/hezhou_air_data/20200101-20250415逐小时气象要素.csv')
    df_weather['time'] = pd.to_datetime(df_weather['time'], format='%Y%m%d%H')
    df_weather.drop_duplicates(subset=['time'], inplace=True)  # 去重
    df_weather.set_index('time', inplace=True)
    df_weather.sort_index(inplace=True)   # 推荐写法
    print(df_weather.head(24))
    # print(hourly_df.index.is_unique)  # 检查索引是否有重复值，应该返回 True
    # print(df_weather.index.is_unique)  # 应该返回 True
    # duplicates = df_weather[df_weather.index.duplicated(keep=False)]  # 查看重复的索引值
    # print(duplicates.sort_index())
    df_merg = pd.concat([hourly_df, df_weather], axis=1, join='inner')    
    print(df_merg.head(24))
    df_merg.to_csv('./data/hezhou_air_data/aqi_meteo_merged.csv', index=True)

    df_processed = df_merg.copy()

    # 处理异常、缺失值
    print("数据预览：")
    print(df_processed.head())
    # 查看数据统计概要
    print("\n数据概要：")
    print(df_processed.describe().T)
    # df_processed.describe().T

    # 风向：把 >360 或 <0 的设为 NaN
    df_processed.loc[df_processed['风向'] > 360, '风向'] = pd.NA
    df_processed.loc[df_processed['风向'] < 0,   '风向'] = pd.NA

    # 气压：把 0 设为 NaN
    df_processed.loc[df_processed['气压'] == 0, '气压'] = pd.NA

    # 方法 A：用前后有效风向的线性角度插值
    df_processed['风向'] = df_processed['风向'].interpolate(method='linear')
    # 只有一条缺失，最稳妥的是线性插值：
    df_processed['气压'] = df_processed['气压'].interpolate(method='time')
    print('\n处理后NA值数量:', df_processed.isna().sum().sum())   # 期望输出 0

    print("\n处理后的数据概要：")
    print(df_processed.describe().T)
    # df_processed.to_csv('./data/hezhou_air_data/processed_data.csv')

    # generate x and y
    # x_shape: [example_count, num_releated, seq_step, feat_size]
    # y_shape: [example_count,]

    # 使用processed_data.csv文件直接处理单站点数据
    # df_processed = pd.read_csv('./data/hezhou_air_data/processed_data.csv')
    # df_processed['time'] = pd.to_datetime(df_processed['time'])
    # df_processed.set_index('time', inplace=True)
    # df_processed.sort_index(inplace=True)
    
    print('Processed data shape:', df_processed.shape)
    print('Columns:', df_processed.columns.tolist())
    
    # 定义特征名称（映射到原项目特征）
    feat_names = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3_8h', 'SO2', 
                  '风向', '气温', '气压', '相对湿度', '风速', '1小时降水量']
    
    # 检查特征是否存在
    missing_features = [feat for feat in feat_names if feat not in df_processed.columns]
    if missing_features:
        print('Missing features:', missing_features)
        # 使用可用特征
        feat_names = [feat for feat in feat_names if feat in df_processed.columns]
    
    print('Using features:', feat_names)
    
    # 序列参数
    x_length = 24  # 24小时历史数据
    y_length = 1   # 预测1小时
    y_step = 1     # 预测步长
    
    # 生成训练数据
    x = []
    y = []
    
    for start_id in range(0, len(df_processed) - x_length - y_length + 1, y_length):
        # 提取特征数据
        x_data = np.array(df_processed[feat_names].iloc[start_id: start_id + x_length])
        
        # 提取目标值（PM2.5）
        y_target = np.array(df_processed['PM2.5'].iloc[start_id + x_length: start_id + x_length + y_length])
        
        # 检查是否有NaN值
        if np.isnan(x_data).any() or np.isnan(y_target).any():
            continue
            
        x.append(x_data)
        y.append(np.mean(y_target))
    
    # 转换为numpy数组
    x = np.array(x)
    y = np.array(y)
    
    # 重塑为模型需要的格式 [样本数, 站点数, 时间步长, 特征维度]
    # 单站点数据，站点数为1
    x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
    
    print('x_shape: {}  y_shape: {}'.format(x.shape, y.shape))
    
    # 保存为pickle文件
    center_station_id = 'hezhou'  # 使用城市名称作为标识
    utils.save_pickle('./data/xy/x_{}.pkl'.format(center_station_id), x)
    utils.save_pickle('./data/xy/y_{}.pkl'.format(center_station_id), y)
    
    print('Data saved successfully!')
    # Save the four dimensional data as pickle file (for STCN model)
    utils.save_pickle('./data/xy/x_{}.pkl'.format(center_station_id), x)
    utils.save_pickle('./data/xy/y_{}.pkl'.format(center_station_id), y)
    print('4D data saved: x_shape: {}  y_shape: {}'.format(x.shape, y.shape))
    
    # Convert 4D to 3D by aggregating spatial information (for GRU/LSTM/RNN/TCN models)
    # Method 1: Mean aggregation across stations
    x_3d_mean = np.mean(x, axis=1)  # [example_count, seq_step, feat_size]
    
    # Method 2: Use only center station data (first station after transpose)
    x_3d_center = x[:, 0, :, :]  # [example_count, seq_step, feat_size]
    
    # Save 3D versions
    utils.save_pickle('./data/xy/x_{}_3d_mean.pkl'.format(center_station_id), x_3d_mean)
    utils.save_pickle('./data/xy/x_{}_3d_center.pkl'.format(center_station_id), x_3d_center)
    print('3D data saved - Mean aggregation: x_shape: {}'.format(x_3d_mean.shape))
    print('3D data saved - Center only: x_shape: {}'.format(x_3d_center.shape))
if __name__ == '__main__':
    main()
