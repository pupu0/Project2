# prediction_model.py
import configparser

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.initializers import lecun_normal
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
from util import convert_csv_to_json
from util.csv_to_json_converter import convert_csv_to_json_ele, convert_csv_to_json_fore_ele


def create_model_2mev():
    """
    创建并返回预测模型。

    返回:
    - model: 创建的2mev的预测模型
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(41, kernel_initializer='normal', activation='relu', input_shape=(41,)))
    model.add(tf.keras.layers.Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))  # 丢弃比例为40%
    model.add(Dense(1))
    return model


def create_model_800kev():
    """
    创建并返回预测模型。

    返回:
    - model: 创建的800kev的预测模型
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(41, kernel_initializer='normal', activation='relu', input_shape=(41,)))
    model.add(tf.keras.layers.Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    return model


def create_model_475kev():
    """
    创建并返回预测模型。

    返回:
    - model: 创建的475kev的预测模型
    """
    model = tf.keras.models.Sequential()
    model.add(Dense(41, input_dim=41))
    model.add(Activation('relu'))
    model.add(Dense(32, kernel_regularizer=l1(0.001)))
    model.add(Activation('relu'))
    model.add(Dense(25, kernel_regularizer=l2(0.001)))
    model.add(Activation('selu'))
    model.add(Dropout(0.4))  # 丢弃比例为40%
    model.add(Dense(1))
    return model


def create_model_275kev():
    """
    创建并返回预测模型。

    返回:
    - model: 创建的275kev的预测模型
    """
    model = tf.keras.models.Sequential()
    model.add(Dense(41, input_dim=41, activation='relu'))
    model.add(Dense(32, activation='relu', kernel_regularizer=l1(0.0005)))
    model.add(Dense(25, activation='selu', kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


def load_and_predict(model, weights_path, test_data_path, thoroughfare):
    """
    加载模型权重并进行预测。
    参数:
    - model: 预测模型
    - weights_path: 模型权重文件路径
    - test_data_path: 测试数据文件路径
    - thoroughfare: 电子通道列下标

    返回:
    - y_pred_test: 预测值
    """
    model.load_weights(weights_path)

    # 最大值和最小值列表
    max_values = [924, 790, 785, 781, 72, 68, 66, 63, 37, 36, 36, 35.7, 36.5, 33.6, 32.4, 29.6, 8.3, 8.3, 8.3, 8.17,
                  7.38, 7.18, 7.02, 6.9, 6.68, 6.48, 6.35, 6.25, 5.85, 5.68, 5.63,
                  5.57, 5.45, 5.45, 5.44, 5.44, 5.31, 5.29, 5.27, 5.24, 24]
    min_values = [244, 267, 270, 270, 0.12, 0.371, 0.378, 0.382, -27.5, -24.8, -19.5, -19.3, -37.6, -19.7, -18.6, -18.5,
                  0, 0, 0, 0, 0.982, 1.96, 2.25, 2.33, 1.07, 1.35, 1.5, 1.59, 0.724, 0.808, 0.841, 0.866, 0.379, 0.407,
                  0.415, 0.418, 0.864, 1.07, 1.08, 1.08, 0.0151]

    # 创建MinMaxScaler实例，并使用训练集的最大值和最小值来初始化
    mm = MinMaxScaler()
    mm.fit(np.vstack([min_values, max_values]))  # 使用最小值和最大值来fit scaler

    # 加载测试数据并使用训练集的归一化参数进行归一化
    data_test = np.genfromtxt(test_data_path, delimiter=',', skip_header=1)
    datax_test = mm.transform(data_test[:, 1:42])  # 对数据进行归一化

    """
    mm = MinMaxScaler()
    data_test = np.genfromtxt(test_data_path, delimiter=',', skip_header=1)
    datax_test = mm.fit_transform(data_test[:, 1:42])
    """

    y_pred_test = model.predict(datax_test)

    """
    # 提取 thoroughfare 列的最大值和最小值
    thoroughfare_column = data_test[:, thoroughfare]
    thoroughfare_min = np.min(thoroughfare_column)
    thoroughfare_max = np.max(thoroughfare_column)
    """
    thoroughfare_min = min_values[thoroughfare-1]
    thoroughfare_max = max_values[thoroughfare-1]

    # 反归一化预测结果
    y_pred_test = y_pred_test*(thoroughfare_max-thoroughfare_min)+thoroughfare_min

    return y_pred_test.ravel()


def load_and_predict2(model, weights_path, test_data_path, thoroughfare):
    """
    加载模型权重并进行预测。
    参数:
    - model: 预测模型
    - weights_path: 模型权重文件路径
    - test_data_path: 测试数据文件路径
    - thoroughfare: 电子通道列下标

    返回:
    - y_pred_test: 预测值
    """
    model.load_weights(weights_path)
    mm = MinMaxScaler()
    data_test = np.genfromtxt(test_data_path, delimiter=',', skip_header=1)
    datax_test = mm.fit_transform(data_test[:, 1:42])

    y_pred_test = model.predict(datax_test)

    # 提取 thoroughfare 列的最大值和最小值
    thoroughfare_column = data_test[:, thoroughfare]
    thoroughfare_min = np.min(thoroughfare_column)
    thoroughfare_max = np.max(thoroughfare_column)

    # 创建一个新的 MinMaxScaler 实例，用于反归一化
    mm_pred = MinMaxScaler(feature_range=(thoroughfare_min, thoroughfare_max))
    mm_pred.fit(y_pred_test)

    # 反归一化预测结果
    y_pred_test = mm_pred.inverse_transform(y_pred_test)

    return y_pred_test.ravel()


def divide_columns_by_factors2(input_file, output_file, config_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 读取配置文件
    config = configparser.ConfigParser()
    config.read(config_file)

    # 根据配置文件中的设置对列进行操作
    for col in df.columns:
        column_name = col.split('-')[0]  # 提取列名中的关键字
        if column_name in config:
            a = float(config[column_name]['a'])
            b = float(config[column_name]['b'])
            df[col] = (df[col] - b) / a

    # 将结果写入新的CSV文件
    df.to_csv(output_file, index=False)


def divide_columns_by_factors(input_file, output_file, config_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 读取配置文件
    config = configparser.ConfigParser()
    config.read(config_file)

    # 定义文件名中的关键词与配置文件中的段落的映射
    keyword_to_section = {
        '2MeV': '2MeV',
        '800KeV': '906keV',
        '475KeV': '380keV'
    }

    # 检查输入文件名中包含的关键词，并选择对应的配置段落
    selected_section = None
    for keyword, section in keyword_to_section.items():
        if keyword in input_file:
            selected_section = section
            break

    # 如果找到了对应的配置段落，读取a和b的值，并计算第二和第三列
    if selected_section and selected_section in config:
        a = float(config[selected_section]['a'])
        b = float(config[selected_section]['b'])
        df.iloc[:, 1] = (df.iloc[:, 1] - b) / a
        # df.iloc[:, 2] = (df.iloc[:, 2] - b) / a

    # 将修改后的DataFrame保存到新的CSV文件中
    df.to_csv(output_file, index=False)


def merge_and_save_results(test_data_path, output_file_path, y_pred_test):
    """
    合并预测结果并保存到文件。

    参数:
    - test_data_path: 测试数据文件路径
    - output_file_path: 输出文件路径
    - y_val_test: 真实值
    - y_pred_test: 预测值
    """
    data = {
        'predictive_value': y_pred_test
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file_path, index=False)
    print(f'预测值已保存到 {output_file_path}')

    df1 = pd.read_csv(test_data_path)
    df2 = pd.read_csv(output_file_path)
    df2['predictive_value'] = df2.iloc[:, 0]
    merged_df = pd.concat([df1.iloc[:, 0], df2], axis=1)

    merged_df.to_csv(output_file_path, index=False)
    print(f'数据已整合到 {output_file_path}')


def reverse_log10_transform(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # Apply reverse log10 (10**value) to the 2nd and 3rd columns
    df.iloc[:, 1] = 10 ** df.iloc[:, 1]
    # df.iloc[:, 2] = 10 ** df.iloc[:, 2]
    df.to_csv(output_csv, index=False)


def process_and_save_predictions(model, weights_path, input_data, energy_level, thoroughfare, config_file, d_start, d_end, csv_path, json_path):
    # 加载模型和权重，进行预测（2015年数据）
    y_pred = load_and_predict(model, weights_path, input_data, thoroughfare)

    # 合并并保存结果（2015年数据）
    product_path = os.path.join(json_path, d_start[:4], d_start[:8])
    os.makedirs(product_path, exist_ok=True)
    output_csv_1 = os.path.join(csv_path, f"results_{d_start}_{d_end}_{energy_level}_1.csv")
    output_csv_2 = os.path.join(csv_path, f"results_{d_start}_{d_end}_{energy_level}.csv")
    # ele_json = f"Z_NAFP_C_OBS--_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_{energy_level}_ZXGC_NCSW_V1A.json"
    fore_ele_json = f"Z_NAFP_C_AIFC-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_{energy_level}_ZXGC_NCSW_V1A.json"
    # output_json_ele = os.path.join(product_path, ele_json)
    output_json_fore_ele = os.path.join(product_path, fore_ele_json)
    merge_and_save_results(input_data, output_csv_1, y_pred)
    # 返回去值
    divide_columns_by_factors(output_csv_1, output_csv_2, config_file)
    reverse_log10_transform(output_csv_2, output_csv_2)
    # convert_csv_to_json_ele(output_csv_2, output_json_ele)
    convert_csv_to_json_fore_ele(output_csv_2, output_json_fore_ele)
    print(f"测试集的{energy_level}预测数据内容保存至{output_json_fore_ele}")
    return f"results_{d_start}_{d_end}_{energy_level}.csv"


def main(input_path, d_start, d_end, csv_path, json_path):
    Model_2mev = create_model_2mev()
    Model_475kev = create_model_475kev()
    Model_800kev = create_model_800kev()

    weights_Path_2mev = '/src/ElectronModel/src/model/model_2mev.h5'
    weights_Path_475kev = '/src/ElectronModel/src/model/model_475kev.h5'
    weights_Path_800kev = '/src/ElectronModel/src/model/model_800kev.h5'

    config_file = '/src/ElectronModel/util/set_a_b_avg.ini'

    # 对于2mev模型
    result2_csv = process_and_save_predictions(
        Model_2mev, weights_Path_2mev, input_path, '2MeV', 37, config_file,d_start, d_end, csv_path, json_path)
    # 对于475kev模型
    result800_csv = process_and_save_predictions(
        Model_800kev, weights_Path_800kev, input_path, '800KeV', 33,config_file, d_start, d_end, csv_path, json_path)
    # 对于800kev模型
    result475_csv = process_and_save_predictions(
        Model_475kev, weights_Path_475kev, input_path, '475KeV', 29,config_file, d_start, d_end, csv_path, json_path)
    return [result2_csv, result800_csv, result475_csv]
