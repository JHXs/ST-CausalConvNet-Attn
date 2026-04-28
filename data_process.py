# coding:utf-8

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
from utils import utils

prediction_variables = cfg.prediction_variables

DATA_DIR = 'stations_data_Guangzhou'

def main():
    primary_var = prediction_variables[0]  # 使用第一个变量筛选相关站点
    print(f"预测变量列表: {prediction_variables} (相关站点筛选使用: {primary_var})")

    # extract station id list in Beijing
    df_airq = pd.read_csv("./data/microsoft_urban_air_data/airquality.csv")
    # station_id_list = np.unique(df_airq["station_id"])[:36]  # first 36 stations are in Beijing
    station_id_list = np.arange(9017, 9047).tolist() + [9058]
    station_id_list = np.array(station_id_list)
    print(station_id_list)

    # Calculate the influence degree (defined as the Pearson correlation coefficient) between the center station and other stations
    r_thred = 0.85
    center_station_id = 9022  # 9022 1013
    station_id_related_list = []
    df_one_station = pd.read_csv(
        f"./data/{DATA_DIR}/df_station_{center_station_id}.csv"
    )
    v_list_1 = list(df_one_station[primary_var])
    for station_id_other in station_id_list:
        df_one_station_other = pd.read_csv(
            f"./data/{DATA_DIR}/df_station_{station_id_other}.csv"
        )
        v_list_2 = list(df_one_station_other[primary_var])
        r, p = stats.pearsonr(v_list_1, v_list_2)  ## 计算与中心站点皮尔逊系数
        if r > r_thred:
            station_id_related_list.append(station_id_other)
        print("{}  {}  {:.3f}".format(center_station_id, station_id_other, r))
    print(len(station_id_related_list), station_id_related_list)

    # generate x and y
    # x_shape: [example_count, num_releated, seq_step, feat_size]
    # y_shape: [example_count, y_length] for a single target variable,
    # or [example_count, y_length * target_count] for multiple target variables.
    print(
        "Center station: {}\nRelated stations: {}".format(
            center_station_id, station_id_related_list
        )
    )
    feat_names = [
        "PM25_Concentration",
        "PM10_Concentration",
        "NO2_Concentration",
        "CO_Concentration",
        "O3_Concentration",
        "SO2_Concentration",
        "weather",
        "temperature",
        "pressure",
        "humidity",
        "wind_speed",
        "wind_direction",
    ]
    x_length = cfg.seq_len  # 168 hours history
    y_length = cfg.output_size  # Predict next 24 hours (Multi-step)
    y_step = 1  # 滑动窗口步长
    x = []
    y = []
    for station_id in station_id_related_list:
        df_one_station = pd.read_csv(
            f"./data/{DATA_DIR}/df_station_{station_id}.csv"
        )
        x_one = []
        # Use y_step (1) as stride to maximize data samples (Sliding Window)
        for start_id in range(0, len(df_one_station) - x_length - y_length + 1 - y_step + 1, y_step):
            x_data = np.array(
                df_one_station[feat_names].iloc[start_id : start_id + x_length]
            )
            y_data = np.array(
                df_one_station[prediction_variables].iloc[
                    start_id + x_length + y_step - 1 : start_id + x_length + y_length + y_step - 1
                ]
            )
            if len(prediction_variables) == 1:
                y_data = y_data.reshape(-1)
            else:
                # Flatten as [t1_var1, t1_var2, ..., t2_var1, ...].
                # eval.py reshapes this back to [sample, horizon, variable].
                y_data = y_data.reshape(-1)
            if np.isnan(x_data).any() or np.isnan(y_data).any():
                continue
            x_one.append(x_data)
            if station_id == center_station_id:
                # Keep full vector for multi-step prediction
                y.append(y_data)
                # y.append(np.mean(y_list))
        if len(x_one) <= 0:
            continue
        x_one = np.array(x_one)
        x.append(x_one)
        print("station_id: {}  x_shape: {}".format(station_id, x_one.shape))

    x = np.array(x)
    x = x.transpose((1, 0, 2, 3))
    y = np.array(y)
    print("x_shape: {}  y_shape: {}".format(x.shape, y.shape))

    # Save the four dimensional data as pickle file (for STCN model)
    utils.save_pickle("./data/xy/x_{}.pkl".format(center_station_id), x)
    utils.save_pickle("./data/xy/y_{}.pkl".format(center_station_id), y)
    print("4D data saved: x_shape: {}  y_shape: {}".format(x.shape, y.shape))

    # Convert 4D to 3D by aggregating spatial information (for GRU/LSTM/RNN/TCN models)
    # Method 1: Mean aggregation across stations
    x_3d_mean = np.mean(x, axis=1)  # [example_count, seq_step, feat_size]

    # Method 2: Use only center station data (first station after transpose)
    x_3d_center = x[:, 0, :, :]  # [example_count, seq_step, feat_size]

    # Save 3D versions
    utils.save_pickle("./data/xy/x_{}_3d_mean.pkl".format(center_station_id), x_3d_mean)
    utils.save_pickle(
        "./data/xy/x_{}_3d_center.pkl".format(center_station_id), x_3d_center
    )
    print("3D data saved - Mean aggregation: x_shape: {}".format(x_3d_mean.shape))
    print("3D data saved - Center only: x_shape: {}".format(x_3d_center.shape))


if __name__ == "__main__":
    main()
