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
    hourly_df.to_csv(f'./data/hezhou_air_data/aqi.csv', index=True)

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

    # generate x and y
    # x_shape: [example_count, num_releated, seq_step, feat_size]
    # y_shape: [example_count,]
    print('Center station: {}\nRelated stations: {}'.format(center_station_id, station_id_related_list))
    feat_names = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration',
                  'weather', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']
    x_length = 24
    y_length = 1
    y_step = 1
    x = []
    y = []
    for station_id in station_id_related_list:
        df_one_station = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(station_id))
        x_one = []
        for start_id in range(0, len(df_one_station)-x_length-y_length+1-y_step+1, y_length):
            x_data = np.array(df_one_station[feat_names].iloc[start_id: start_id+x_length])
            y_list = np.array(df_one_station['PM25_Concentration'].iloc[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1])
            if np.isnan(x_data).any() or np.isnan(y_list).any():
                continue
            x_one.append(x_data)
            if station_id == center_station_id:
                y.append(np.mean(y_list))
        if len(x_one) <= 0:
            continue
        x_one = np.array(x_one)
        x.append(x_one)
        print('station_id: {}  x_shape: {}'.format(station_id, x_one.shape))

    x = np.array(x)
    x = x.transpose((1, 0, 2, 3))
    y = np.array(y)
    print('x_shape: {}  y_shape: {}'.format(x.shape, y.shape))
    
    # Save the four dimensional data as pickle file
    utils.save_pickle('./data/xy/x_{}.pkl'.format(center_station_id), x)
    utils.save_pickle('./data/xy/y_{}.pkl'.format(center_station_id), y)
    print('x_shape: {}\ny_shape: {}'.format(x.shape, y.shape))


if __name__ == '__main__':
    main()
