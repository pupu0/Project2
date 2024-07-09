# evaluate_model_results.py
import pandas as pd
import numpy as np
import json


def evaluate_model(csv_file):
    """
    评估模型性能，并返回评估指标
    :param csv_file: 包含真值和预测值的CSV文件路径
    :return: 返回字典类型的评估指标
    """

    df = pd.read_csv(csv_file)

    true_values = df.iloc[:, 1].values
    predicted_values = df.iloc[:, 2].values

    # 过滤空值
    valid_indices = ~np.isnan(true_values) & ~np.isnan(predicted_values)

    true_values = true_values[valid_indices]
    predicted_values = predicted_values[valid_indices]

    # 计算评估指标
    rmse = np.sqrt(np.mean((predicted_values - true_values) ** 2))  # 均方差
    cc = np.corrcoef(predicted_values, true_values)[0, 1]   # 相关系数
    pe = 100.0*(1 - (np.sum(np.square(predicted_values - true_values)) / np.sum(np.square(predicted_values - np.mean(true_values)))))   # 预测效率
    mean_relative_error = np.mean(np.abs((predicted_values - true_values) / true_values)) * 100.0   # 平均相对误差

    # 将评估指标保存为字典类型
    metrics = {
        "PE": f"{round(pe, 4)}%",
        "RMSE": round(rmse, 4),
        "CC": round(cc, 4),
        "MRE": f"{round(mean_relative_error, 4)}%"
    }

    return metrics


def print_evaluation_results(metrics_2015,energy_level):
    print(f"{energy_level}: Evaluation indicator results for the test set :")
    for key, value in metrics_2015.items():
        print(f"{key}: {value}")
    print("\n")


def main():
    # 评估CSV文件路径
    csv_file_2015_2mev = '/src/PredictionModel/forecast_results/results_2015_2mev.csv'
    # csv_file_2019_2mev = '/src/PredictionModel/forecast_results/results_2019_2mev.csv'
    csv_file_2015_800kev = '/src/PredictionModel/forecast_results/results_2015_800kev.csv'
    # csv_file_2019_800kev = '/src/PredictionModel/forecast_results/results_2019_800kev.csv'
    csv_file_2015_475kev = '/src/PredictionModel/forecast_results/results_2015_475kev.csv'
    # csv_file_2019_475kev = '/src/PredictionModel/forecast_results/results_2019_475kev.csv'

    # 评估模型性能
    metrics_2015_2mev = evaluate_model(csv_file_2015_2mev)
    # metrics_2019_2mev = evaluate_model(csv_file_2019_2mev)
    metrics_2015_800kev = evaluate_model(csv_file_2015_800kev)
    # metrics_2019_800kev = evaluate_model(csv_file_2019_800kev)
    metrics_2015_475kev = evaluate_model(csv_file_2015_475kev)
    # metrics_2019_475kev = evaluate_model(csv_file_2019_475kev)

    metrics = {
        "2mev:Evaluation indicator results for the test set (October December 2015):":
            metrics_2015_2mev,
        "800kev:Evaluation indicator results for the test set (October December 2015):":
            metrics_2015_800kev,
        "475kev:Evaluation indicator results for the test set (October December 2015):":
            metrics_2015_475kev
    }

    # 将评估指标保存为JSON文件
    with open('/src/PredictionModel/SpaceWeather/ybfx/product/Z_NAFP_C_VERIF_20240209000000_P_OTHE_00_FLUX_L1_20240209000000_0005MP_XXXXX_ZXGC_NCSW_V1A.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # 打印2mev的评估指标结果
    print_evaluation_results(metrics_2015_2mev, "2mev")
    # 打印800kev的评估指标结果
    print_evaluation_results(metrics_2015_800kev, "800kev")
    # 打印475kev的评估指标结果
    print_evaluation_results(metrics_2015_475kev, "475kev")


