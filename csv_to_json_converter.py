# csv_to_json_converter.py
import pandas as pd
import json
from datetime import datetime, timedelta


def convert_csv_to_json(csv_file_path, json_file_path, orient='records', lines=True):
    """
    将CSV文件转换为JSON文件。

    参数:
    - csv_file_path: str, CSV文件的路径。
    - json_file_path: str, 输出JSON文件的路径。
    - orient: str, 输出JSON的格式。
    - lines: bool, 是否每条记录为一行。
    """
    # 使用pandas读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 将DataFrame转换为JSON格式，并保存到文件
    df.to_json(json_file_path, orient=orient, lines=lines)

    print(f'CSV文件已成功转换为JSON文件并保存至：{json_file_path}')


def convert_csv_to_json_ele(input_file, output_file):
    """
    将CSV文件转换为JSON格式，并写入新的JSON文件中。

    Args:
        input_file (str): 输入的CSV文件路径。
        output_file (str): 输出的JSON文件路径。
    """

    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 将每条记录转换为JSON格式
    data = []
    for row in df.itertuples(index=False):
        # 转换时间格式为 '%Y-%m-%d %H:%M:%S'
        formatted_date = datetime.strptime(row.Formatted_Date, '%Y-%m-%d-%H-%M').strftime('%Y-%m-%d %H:%M:%S')

        record = {
            "datetime": formatted_date,
            "observed_value": row.true_value,
        }
        data.append(record)

    # 构建JSON数据
    json_data = {
        "Attributes": {
            "Global": {
                "QualityFlag": 1,
                "TimeRes": "2 Hours",
                "Create_Time": "2024-02-09 00:00:00",
                "Level": "L4",
                "Project": "ZXGC",
                "Construction": "NCSW",
                "Format": "JSON"
            },
            "Variables": {
                "datetime": {"datatype": "str", "description": u"资料时间"},
                "observed_value": {"datatype": "float", "description": u"flux观测数据"}
            }
        },
        "Data": data
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


def convert_csv_to_json_fore_ele(input_file, output_file):
    """
    将CSV文件转换为JSON格式，并写入新的JSON文件中。

    Args:
        input_file (str): 输入的CSV文件路径。
        output_file (str): 输出的JSON文件路径。
    """

    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 将每条记录转换为JSON格式
    data = []
    for row in df.itertuples(index=False):
        # 转换时间格式为 '%Y-%m-%d %H:%M:%S'
        formatted_date = datetime.strptime(row.Formatted_Date, '%Y-%m-%d-%H-%M').strftime('%Y-%m-%d %H:%M:%S')

        # 计算未来两小时后的时间
        future_time = datetime.strptime(formatted_date, '%Y-%m-%d %H:%M:%S') + timedelta(hours=2)
        d_foretime = future_time.strftime('%Y-%m-%d %H:%M:%S')
        record = {
            "d_datetime": formatted_date,
            "d_foretime": d_foretime,
            "valid_time": 2,
            "valid_time_unit": "hours",
            # "observed_value": None,
            "predicted_value": row.predictive_value
        }
        data.append(record)

    # 构建JSON数据
    json_data = {
        "Attributes": {
            "Global": {
                "QualityFlag": 1,
                "TimeRes": "2 Hours",
                "Create_Time": "2024-02-09 00:00:00",
                "Level": "L4",
                "Project": "ZXGC",
                "Construction": "NCSW",
                "Format": "JSON"
            },
            "Variables": {
                "datetime": {"datatype": "str", "description": u"资料时间"},
                "foretime": {"datatype": "str", "description": u"预报时间"},
                "valid_time": {"datatype": "int", "description": u"预报时效"},
                "valid_time_unit": {"datatype": "str", "description": u"预报时效单位"},
                "predicted_value": {"datatype": "float", "description": u"flux预报数据"}
            }
        },
        "Data": data
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

