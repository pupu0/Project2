import os
import json
import numpy as np
import pandas as pd
import configparser
import csv
from datetime import datetime, timedelta
import datetime


def json_to_csv(json_file, columns, output_csv_file):
    existing_time_tags = set()  # 初始化existing_time_tags
    # 检查CSV文件是否存在
    if os.path.isfile(output_csv_file):
        # 如果CSV文件存在，加载CSV数据
        df_csv = pd.read_csv(output_csv_file)

        # 检查是否存在"time_tag"列
        if "time_tag" not in df_csv.columns:
            # 如果不存在"time_tag"列但存在"Formatted_Date"列，则当作CSV文件不存在处理
            if "Formatted_Date" in df_csv.columns:
                df_csv = pd.DataFrame(columns=columns)
                existing_time_tags = set()
        else:
            # 读取CSV文件中已有的时间标签
            existing_time_tags = set(df_csv["time_tag"])
    else:
        # 如果CSV文件不存在，创建一个空的DataFrame
        df_csv = pd.DataFrame(columns=columns)
    """
    # 检查CSV文件是否存在
    if os.path.isfile(output_csv_file):
        # 如果CSV文件存在，加载CSV数据
        df_csv = pd.read_csv(output_csv_file)

        # 读取CSV文件中已有的时间标签
        existing_time_tags = set(df_csv["time_tag"])
    else:
        # 如果CSV文件不存在，创建一个空的DataFrame
        df_csv = pd.DataFrame(columns=columns)

        # 设置已有的时间标签为空集合
        existing_time_tags = set()
    """
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 创建一个列表来存储所有行的字典数据
    rows = []

    # 遍历JSON数据
    for item in data:
        # 获取energy键对应的值
        energy_value = item.get("energy")

        # 检查energy值是否在CSV文件中存在对应的列名
        if energy_value not in df_csv.columns:
            df_csv[energy_value] = None

        # 提取指定列及其对应的值
        selected_item = {col: item[col] for col in columns if col != 'energy'}

        # 检查时间标签是否已存在于CSV文件中
        if selected_item["time_tag"] in existing_time_tags:
            # 如果时间标签已存在，更新CSV文件中的相应行
            for col in columns[1:]:
                if col == 'energy':
                    continue
                df_csv.loc[df_csv["time_tag"] == selected_item["time_tag"], col] = selected_item[col]
        else:
            # 如果时间标签不存在，将行的字典数据添加到列表中
            rows.append(selected_item)

        # 将flux值填充到对应的行和列
        df_csv.loc[df_csv["time_tag"] == selected_item["time_tag"], energy_value] = item.get("flux")

    # 将所有行的字典数据转换为DataFrame并与现有的CSV数据进行合并
    new_df = pd.DataFrame(rows)
    df_csv = pd.concat([df_csv, new_df], ignore_index=True)

    # 保存数据到CSV文件
    df_csv.to_csv(output_csv_file, index=False)


def calculate_rolling_means(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    """
    # 删除所有值为空的列
    df.dropna(axis=1, how='all', inplace=True)

    # 将第一列列名修改为'Formatted_Date'
    df.rename(columns={'time_tag': 'Formatted_Date'}, inplace=True)

    df['Formatted_Date'] = pd.to_datetime(df['Formatted_Date'], format='%Y-%m-%dT%H:%M:%SZ')

    # 对除了'Formatted_Date'列进行线性插值补孔
    df.interpolate(method='linear', inplace=True)
    """
    # 将Formatted_Date列转换为datetime格式
    df['Formatted_Date'] = pd.to_datetime(df['Formatted_Date'], format='%Y-%m-%d-%H-%M')

    # 获取除了'mlt'和 'Formatted_Date'列之外的列名
    data_columns = [col for col in df.columns if col not in ['mlt', 'Formatted_Date']]

    # 计算滑动窗口移动平均值
    def rolling_mean(data, window):
        return data.rolling(window=window, min_periods=1).mean()  # 添加min_periods参数

    # 计算新列的值
    def calculate_new_columns(column_name, hours, df):
        new_column_name = f"{column_name}-{hours}h"
        if len(df) >= hours:
            df[new_column_name] = rolling_mean(df[column_name], window=hours)
        else:
            df[new_column_name] = None

    # 除了指定的列之外，对所有列进行计算
    for column in df.columns:
        if column != 'Formatted_Date' and column != 'mlt':
            calculate_new_columns(column, 12, df)
            calculate_new_columns(column, 24, df)
            calculate_new_columns(column, 36, df)
    # 将Formatted_Date列改为%Y-%m-%d-%H-%M
    df['Formatted_Date'] = df['Formatted_Date'].dt.strftime('%Y-%m-%d-%H-%M')

    # 将结果写入新的CSV文件
    df.to_csv(output_file, index=False)


def process_csv(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 将Formatted_Date列转换为datetime格式
    df['Formatted_Date'] = pd.to_datetime(df['Formatted_Date'], format='%Y-%m-%d-%H-%M')

    df['131keV-next'] = None
    df['272keV-next'] = None
    df['380keV-next'] = None
    df['906keV-next'] = None
    df['2MeV-next'] = None

    # 遍历数据框中的每一行
    for index, row in df.iterrows():
        # 计算未来两小时的时间
        future_time = row['Formatted_Date'] + pd.Timedelta(hours=2)

        # 查找未来两小时对应的记录
        future_record = df[df['Formatted_Date'] == future_time]

        # 如果找到了对应的记录，则将其 '150keV', '275keV', '475keV', '800keV', '2MeV' 列的值填入对应的 'next' 列
        if not future_record.empty:
            df.at[index, '131keV-next'] = future_record['131keV'].iloc[0]
            df.at[index, '272keV-next'] = future_record['272keV'].iloc[0]
            df.at[index, '380keV-next'] = future_record['380keV'].iloc[0]
            df.at[index, '906keV-next'] = future_record['906keV'].iloc[0]
            df.at[index, '2MeV-next'] = future_record['2MeV'].iloc[0]

    # 删除空缺值的数据
    df.dropna(inplace=True)

    # 将Formatted_Date列改为 %Y-%m-%d-%H-%M
    df['Formatted_Date'] = df['Formatted_Date'].dt.strftime('%Y-%m-%d-%H-%M')
    df.to_csv(output_file, index=False)


def multiply_columns_by_factors(input_file, output_file, config_file):
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
            df[col] = df[col] * a + b

    # 将结果写入新的CSV文件
    df.to_csv(output_file, index=False)


def reorder_columns(input_file, output_file, new_order):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 按指定顺序重新排列列
    df = df[new_order]

    # 保留最后60条数据记录
    df = df[-24:]
    # 将结果保存到新的CSV文件
    df.to_csv(output_file, index=False)


def process_log10(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    # 判断每一列是否全为空
    all_null_columns = df.columns[df.isnull().all()]
    # 删除所有全为空的列
    df.drop(all_null_columns, axis=1, inplace=True)

    # 删除所有值为空的列
    # df.dropna(axis=1, how='all', inplace=True)
    # df.dropna(axis=0, how='all', subset=df.columns[1:], inplace=True)

    # 将第一列列名修改为'Formatted_Date'
    df.rename(columns={'time_tag': 'Formatted_Date'}, inplace=True)

    df['Formatted_Date'] = pd.to_datetime(df['Formatted_Date'], format='%Y-%m-%dT%H:%M:%S.000Z')

    # 将空字符串转换为 NaN
    df.replace('', np.nan, inplace=True)

    columns = df.columns
    """
    for column in df.columns:
        # 检查列名是否包含 'eV'
        if 'eV' in column:
            # 对该列中的 NaN 值不进行处理
            df[column] = df[column].replace(0, np.nan)
    """
    for column in df.columns:
        # 检查列名是否包含 'eV'
        if 'eV' in column:
            # 将该列中小于等于1的数值置为 NaN
            df[column] = df[column].apply(lambda x: np.nan if x <= 1 else x)

    # 对除了'Formatted_Date'列进行线性插值补孔
    # df.interpolate(method='linear', inplace=True)
    # df.dropna(axis=0, how='any', subset=['Vsw', 'Nsw'], inplace=True)
    df.fillna(method='bfill', inplace=True)
    # 删除所有不包含数值的行
    # df.dropna(how='all', inplace=True)
    # 获取所有列名
    # columns = df.columns
    # 遍历每一列
    """
    for column in columns:
        # 检查列名是否包含 'eV'
        if 'eV' in column:
            # 对该列取以10为底的对数
            df[column] = np.log10(df[column])
            # 删除该列中小于0的行
            df = df[df[column] >= 0]
    """
    for column in columns:
        if 'eV' in column:
            # 对该列取以10为底的对数
            df[column] = np.log10(df[column])
            # 将该列中小于0的数值改为NaN
            df[column] = df[column].where(df[column] >= 0, np.nan)

    df.fillna(method='bfill', inplace=True)

    # 将Formatted_Date列改为 %Y-%m-%d-%H-%M
    df['Formatted_Date'] = df['Formatted_Date'].dt.strftime('%Y-%m-%d-%H-%M')
    # 保存修改后的数据到CSV文件
    df.to_csv(output_file, index=False)


def process_kp(input_file_path, output_file_path):
    # 检查输出文件是否存在
    file_exists = os.path.exists(output_file_path)

    # 打开输入文件
    with open(input_file_path, 'r') as file:
        # 创建或追加到CSV文件
        with open(output_file_path, 'a' if file_exists else 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if not file_exists:
                # 写入CSV文件头部
                writer.writerow(['time_tag', 'Kp'])

            # 逐行读取文件内容
            for line in file:
                if line[0].isdigit():
                    date_str = line[:10]
                    current_datetime = datetime.strptime(date_str + ' ' + '000000', '%Y %m %d %H%M%S')

                    kp_data = line.split()[-8:]
                    all_empty = all(kp == '-1.00' for kp in kp_data)
                    if not all_empty:
                        for kp_value in kp_data:
                            if kp_value == '-1.00':
                                kp_value = ''
                            for _ in range(3 * 12):
                                time_tag = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
                                writer.writerow([time_tag, kp_value])
                                current_datetime += timedelta(minutes=5)


"""
def read_csv_data(file_path):
    data = {}
    with open(file_path, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            time_tag = row['time_tag']
            kp_value = row['Kp']
            if kp_value.isdigit() or is_float(kp_value):  # 确保Kp值是数字
                kp_value = float(kp_value)
                if time_tag in data:
                    data[time_tag].append(kp_value)
                else:
                    data[time_tag] = [kp_value]
    return data
"""


def read_csv_data(file_path):
    data = {}
    with open(file_path, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            time_tag = row['time_tag']
            kp_value = row['Kp']
            if kp_value.isdigit() or is_float(kp_value):  # 确保Kp值是数字
                kp_value = float(kp_value)
                if time_tag in data:
                    # kp_value = (kp_value + pre_kp_value)/2.0
                    # data[time_tag].append(kp_value)
                    data[time_tag] = [kp_value]
                    # pre_kp_value = kp_value
                else:
                    data[time_tag] = [kp_value]
                    # pre_kp_value = kp_value

    return data


def is_float(element):
    """检查字符串是否可以转换为浮点数。"""
    try:
        float(element)
        return True
    except ValueError:
        return False


def write_csv_data(file_path, data):
    """将处理后的数据写入CSV文件。"""
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time_tag', 'Kp'])  # 写入头部
        for time_tag in sorted(data):
            avg_kp = sum(data[time_tag]) / len(data[time_tag])
            writer.writerow([time_tag, avg_kp])


def process_csv_file(input_file_path, output_file_path):
    """读取CSV，处理数据，然后将结果写回新的CSV文件。"""
    data = read_csv_data(input_file_path)
    write_csv_data(output_file_path, data)


def process_energy_data(json_file, columns, output_csv_file):
    df_csv = pd.read_csv(output_csv_file)

    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 遍历JSON数据
    for item in data:
        energy_value = item.get("energy")

        selected_energy = None
        if energy_value == "115-165 keV":
            selected_energy = "131keV"
        elif energy_value == "235-340 keV":
            selected_energy = "272keV"
        elif energy_value == "340-500 keV":
            selected_energy = "380keV"
        elif energy_value == "700-1000 keV":
            selected_energy = "906keV"
        elif energy_value == ">=2 MeV":
            selected_energy = "2MeV"

        if selected_energy is None:
            continue

        if selected_energy not in df_csv.columns:
            df_csv[selected_energy] = None

        selected_item = {col: item[col] for col in columns if col != 'energy'}
        df_csv.loc[df_csv["time_tag"] == selected_item["time_tag"], selected_energy] = item.get("flux")

    df_csv.to_csv(output_csv_file, index=False)


def process_solar_wind(json_file, output_csv_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    data = json_data[1:]
    columns = json_data[0]
    df = pd.DataFrame(data, columns=columns)
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df['speed'] = df['speed'].astype(float)

    # 计算所需分钟数（四舍五入）
    R = 1500000
    df['required_minutes'] = (R / df['speed'] / 60).round()
    df['new_time_tag'] = df['time_tag'] + pd.to_timedelta(df['required_minutes'], 'm')
    df['new_time_tag'] = df['new_time_tag'].dt.strftime('%Y-%m-%dT%H:%M:00Z')

    new_df = df[['new_time_tag', 'density', 'speed']]
    existing_df = pd.read_csv(output_csv_file)
    if 'Vsw' in existing_df.columns and 'Nsw' in existing_df.columns:
        merged_df = pd.merge(existing_df, new_df, left_on='time_tag', right_on='new_time_tag', how='left')
        merged_df['Vsw'] = merged_df['Vsw'].fillna(merged_df['speed'])
        merged_df[['speed', 'Vsw']] = merged_df[['speed', 'Vsw']].astype(float)  # 将数据类型转换为浮点数
        merged_df['Vsw'] = merged_df[['speed', 'Vsw']].mean(axis=1)

        merged_df['Nsw'] = merged_df['Nsw'].fillna(merged_df['density'])
        merged_df[['density', 'Nsw']] = merged_df[['density', 'Nsw']].astype(float)  # 将数据类型转换为浮点数
        merged_df['Nsw'] = merged_df[['density', 'Nsw']].mean(axis=1)
        merged_df = merged_df.drop('density', axis=1)
        merged_df = merged_df.drop('speed', axis=1)
    else:
        merged_df = pd.merge(existing_df, new_df, left_on='time_tag', right_on='new_time_tag', how='left')
        merged_df = merged_df.rename(columns={'density': 'Nsw', 'speed': 'Vsw'})
    merged_df = merged_df.drop('new_time_tag', axis=1)
    merged_df.to_csv(output_csv_file, index=False)


def process_tyf_and_by_bz2(json_file, output_csv_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    data = json_data[1:]
    columns = json_data[0]
    df = pd.DataFrame(data, columns=columns)
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df['speed'] = df['speed'].astype(float)

    # 计算所需分钟数（四舍五入）
    R = 1500000
    df['required_minutes'] = (R / df['speed'] / 60).round()
    df['new_time_tag'] = df['time_tag'] + pd.to_timedelta(df['required_minutes'], 'm')
    df['new_time_tag'] = df['new_time_tag'].dt.strftime('%Y-%m-%dT%H:%M:00.000Z')

    new_df = df[['new_time_tag', 'density', 'speed', 'by_gsm', 'bz_gsm']]
    existing_df = pd.read_csv(output_csv_file)
    merged_df = pd.merge(existing_df, new_df, left_on='time_tag', right_on='new_time_tag', how='left')

    # 为新字段进行填充和平均计算
    for field in ['density', 'speed', 'by_gsm', 'bz_gsm']:
        new_field = 'Nsw' if field == 'density' else 'Vsw' if field == 'speed' else 'by' if field == 'by_gsm' else 'bz'
        if new_field in existing_df.columns:
            merged_df[new_field] = merged_df[new_field].fillna(merged_df[field])
            merged_df[[field, new_field]] = merged_df[[field, new_field]].astype(float)
            merged_df[new_field] = merged_df[[field, new_field]].mean(axis=1)
            merged_df = merged_df.drop(field, axis=1)
        else:
            merged_df = merged_df.rename(columns={field: new_field})

    merged_df = merged_df.drop('new_time_tag', axis=1)
    merged_df.to_csv(output_csv_file, index=False)


def process_tyf_and_by_bz(json_file, output_csv_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    data = json_data[1:]
    columns = json_data[0]
    df = pd.DataFrame(data, columns=columns)
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df['speed'] = df['speed'].astype(float)

    R = 1500000
    df['required_minutes'] = (R / df['speed'] / 60).round()
    df['new_time_tag'] = df['time_tag'] + pd.to_timedelta(df['required_minutes'], 'm')
    df['new_time_tag'] = df['new_time_tag'].dt.strftime('%Y-%m-%dT%H:%M:00.000Z')

    new_df = df[['new_time_tag', 'density', 'speed', 'by_gsm', 'bz_gsm']]
    existing_df = pd.read_csv(output_csv_file)
    merged_df = pd.merge(existing_df, new_df, left_on='time_tag', right_on='new_time_tag', how='left')

    # Fill and average the fields
    for field in ['density', 'speed', 'by_gsm', 'bz_gsm']:
        new_field = 'Nsw' if field == 'density' else 'Vsw' if field == 'speed' else 'by' if field == 'by_gsm' else 'bz'
        if new_field in existing_df.columns:
            merged_df[new_field] = merged_df[new_field].fillna(merged_df[field])
            merged_df[[field, new_field]] = merged_df[[field, new_field]].astype(float)
            merged_df[new_field] = merged_df[[field, new_field]].mean(axis=1)
            merged_df = merged_df.drop(field, axis=1)
        else:
            merged_df = merged_df.rename(columns={field: new_field})

    merged_df.drop('new_time_tag', axis=1, inplace=True)
    merged_df.rename(columns={'time_tag': 'new_time_tag'}, inplace=True)

    merged_df.drop_duplicates(subset='new_time_tag', keep='first', inplace=True)
    merged_df.rename(columns={'new_time_tag': 'time_tag'}, inplace=True)
    merged_df.to_csv(output_csv_file, index=False)


"""
def process_by_bz(json_file, output_csv_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    data = json_data[1:]
    columns = json_data[0]
    df = pd.DataFrame(data, columns=columns)
    # 将 'time_tag' 列转换为日期时间类型
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    # 修改 new_time_tag 列的时间格式为 '2024-02-03T00:50:00Z'
    df['new_time_tag'] = df['time_tag'].dt.strftime('%Y-%m-%dT%H:%M:00Z')

    new_df = df[['new_time_tag', 'by_gsm', 'bz_gsm']]
    existing_df = pd.read_csv(output_csv_file)

    merged_df = pd.merge(existing_df, new_df, left_on='time_tag', right_on='new_time_tag', how='left')

    merged_df = merged_df.rename(columns={'by_gsm': 'by', 'bz_gsm': 'bz'})
    merged_df = merged_df.drop('new_time_tag', axis=1)

    merged_df.to_csv(output_csv_file, index=False)
"""


def process_by_bz(json_file, output_csv_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    data = json_data[1:]
    columns = json_data[0]
    df = pd.DataFrame(data, columns=columns)
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df['new_time_tag'] = df['time_tag'].dt.strftime('%Y-%m-%dT%H:%M:000Z')

    new_df = df[['new_time_tag', 'by_gsm', 'bz_gsm']]
    existing_df = pd.read_csv(output_csv_file)
    if 'by' in existing_df.columns and 'bz' in existing_df.columns:
        merged_df = pd.merge(existing_df, new_df, left_on='time_tag', right_on='new_time_tag', how='left')
        merged_df['by'] = merged_df['by'].fillna(merged_df['by_gsm'])
        merged_df['bz'] = merged_df['bz'].fillna(merged_df['bz_gsm'])
        merged_df = merged_df.drop('by_gsm', axis=1)
        merged_df = merged_df.drop('bz_gsm', axis=1)
    else:
        merged_df = pd.merge(existing_df, new_df, left_on='time_tag', right_on='new_time_tag', how='left')
        merged_df = merged_df.rename(columns={'by_gsm': 'by', 'bz_gsm': 'bz'})
    merged_df = merged_df.drop('new_time_tag', axis=1)
    merged_df.to_csv(output_csv_file, index=False)

"""
def process_mlt(json_file, output_csv_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    time_tag = [record['EPOCH_']['DAT'] for record in json_data]
    mlt = [record['SM_LCT_T_']['DAT'] for record in json_data]
    df = pd.DataFrame({'time_tag': time_tag, 'mlt': mlt})
    df = df.apply(pd.Series.explode)
    df = df.reset_index(drop=True)
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df['new_time_tag'] = df['time_tag'].dt.strftime('%Y-%m-%dT%H:%M:00.000Z')
    new_df = df[['new_time_tag', 'mlt']]
    existing_df = pd.read_csv(output_csv_file)
    merged_df = pd.merge(existing_df, new_df, left_on='time_tag', right_on='new_time_tag', how='left')
    merged_df = merged_df.drop('new_time_tag', axis=1)
    merged_df.to_csv(output_csv_file, index=False)
"""


def process_mlt(json_file, output_csv_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # 提取时间和MLT数据
    time_tag = json_data[0]['EPOCH_']  # 时间标签列表
    mlt = json_data[0]['SM_LCT_T_']  # MLT值列表

    # 创建DataFrame
    df = pd.DataFrame({'time_tag': time_tag, 'mlt': mlt})
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df['new_time_tag'] = df['time_tag'].dt.strftime('%Y-%m-%dT%H:%M:00.000Z')

    # 只选择需要的列进行合并
    new_df = df[['new_time_tag', 'mlt']]
    existing_df = pd.read_csv(output_csv_file)

    # 使用左连接合并DataFrame
    merged_df = pd.merge(existing_df, new_df, left_on='time_tag', right_on='new_time_tag', how='left')
    merged_df = merged_df.drop('new_time_tag', axis=1)

    # 保存更新后的CSV文件
    merged_df.to_csv(output_csv_file, index=False)


def merge_json_files(input_path1, input_path2, output_path):
    # 读取第一个文件
    with open(input_path1, 'r') as file:
        data1 = json.load(file)

    # 读取第二个文件
    with open(input_path2, 'r') as file:
        data2 = json.load(file)

    # 提取并合并列名
    headers = data1[0] + data2[0][1:]  # 避免重复添加 'time_tag'

    # 将第二个数据文件转换为字典以便快速查找
    data2_dict = {row[0]: row[1:] for row in data2[1:]}

    # 初始化合并后的数据列表
    final_data = [headers]

    # 遍历第一个数据集，根据时间标签查找并合并第二个数据集的数据
    for row1 in data1[1:]:
        time_tag = row1[0]
        # 查找第二个数据集中对应的数据行
        if time_tag in data2_dict:
            merged_row = row1 + data2_dict[time_tag]
        else:
            # 如果第二个数据集中没有对应的时间标签，填充为null
            merged_row = row1 + [None] * (len(data2[0]) - 1)
        final_data.append(merged_row)

    # 写入到输出文件
    with open(output_path, 'w') as file:
        json.dump(final_data, file, indent=4)


def process_and_convert_json(input_path, output_path):
    # 从文件读取 JSON 数据
    def load_data_from_file(filename):
        with open(filename, 'r') as file:
            return json.load(file)

    # 函数：将类 CSV 结构的 JSON 转换为标准的 JSON 对象列表
    def convert_csv_like_json_to_standard_json(data_csv_like):
        headers = data_csv_like[0]  # 列标题
        converted_json = []
        for row in data_csv_like[1:]:
            entry = {}
            for i, value in enumerate(row):
                if headers[i] == 'time_tag':
                    # 格式化时间
                    formatted_time = datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
                    entry[headers[i]] = formatted_time.isoformat() + '.000Z'
                else:
                    entry[headers[i]] = value
            converted_json.append(entry)
        return converted_json

    # 读取文件中的数据
    data_json = load_data_from_file(input_path)
    # 转换数据
    converted_data = convert_csv_like_json_to_standard_json(data_json)
    # 将结果写入到新的 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=4)

def main(kp_path,energy1_path,energy2_path,tyf_path,by_bz,mlt_path,input_path,run_cofig):
    # 处理kp
    kp_path2 = "/src/ElectronModel/src/kp2.json"
    process_and_convert_json(kp_path, kp_path2)
    columns = ['time_tag', 'Kp']
    json_to_csv(kp_path2, columns, input_path)

    # 处理各个能量
    columns = ['time_tag', 'energy']
    process_energy_data(energy1_path, columns, input_path)
    process_energy_data(energy2_path, columns, input_path)

    # 处理太阳风参数和by_bz
    json_file = "/src/ElectronModel/src/tyf_by_bz.json"
    merge_json_files(tyf_path, by_bz, json_file)
    process_and_convert_json(json_file, json_file)
    process_tyf_and_by_bz(json_file, input_path)

    #处理mlt
    process_mlt(mlt_path, input_path)

    # log化
    process_log10(input_path, input_path)
    
    # 计算历史时期值
    calculate_rolling_means(input_path, input_path)
    
    new_order = ['Formatted_Date', 'Vsw', 'Vsw-12h', 'Vsw-24h', 'Vsw-36h', 'Nsw', 'Nsw-12h', 'Nsw-24h', 'Nsw-36h',
                 'by', 'by-12h', 'by-24h', 'by-36h', 'bz', 'bz-12h', 'bz-24h', 'bz-36h', 'Kp', 'Kp-12h', 'Kp-24h',
                 'Kp-36h', '131keV', '131keV-12h', '131keV-24h', '131keV-36h', '272keV', '272keV-12h', '272keV-24h',
                 '272keV-36h', '380keV', '380keV-12h', '380keV-24h', '380keV-36h', '906keV', '906keV-12h', '906keV-24h',
                 '906keV-36h',  '2MeV', '2MeV-12h', '2MeV-24h',
                 '2MeV-36h', 'mlt']    
    reorder_columns(input_path, input_path, new_order)

    multiply_columns_by_factors(input_path, input_path, run_cofig)

