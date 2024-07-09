import os.path
from configparser import ConfigParser
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogFormatter, FuncFormatter
from datetime import datetime, timedelta


plt.rcParams['font.family'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


def load_from_cfg_en(cfg):
    """
    从配置文件中加载样式配置项

    参数:
    - cfg: str, 配置文件路径

    返回:
    - style: dict, 样式配置项
    """
    config = ConfigParser()
    style_cfg = config.read(cfg, encoding='utf-8')
    if not style_cfg:
        raise ValueError(f'无法读取配置文件 {cfg}')
    if not config.has_section('flux_style'):
        raise ValueError(f'配置文件中没有 [flux_style] 部分')
    style = dict(config.items('flux_style'))
    return style


def load_from_cfg_cn(cfg):
    """
    从配置文件中加载样式配置项

    参数:
    - cfg: str, 配置文件路径

    返回:
    - style: dict, 样式配置项
    """
    config = ConfigParser()
    style_cfg = config.read(cfg, encoding='utf-8')
    if not style_cfg:
        raise ValueError(f'无法读取配置文件 {cfg}')
    if not config.has_section('flux_style_cn'):
        raise ValueError(f'配置文件中没有 [flux_style_cn] 部分')
    style = dict(config.items('flux_style_cn'))
    return style


def get(self, name, default=None):
    return self.origin.get(name, default)


def load_data_from_csv(csv_file):
    """
    从CSV文件中加载数据

    参数:
    - csv_file: str, CSV文件路径

    返回:
    - data: DataFrame, 加载的数据
    """
    data = pd.read_csv(csv_file, header=0)
    return data


def plot_graph_en1(data, style, save_path, energy_level):
    """
    绘制图形函数，并保存到指定路径

    参数:
    - data: DataFrame, 要绘制的数据
    - style: dict, 样式配置项
    - save_path: str, 保存路径

    返回:
    无返回值
    """
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文支持的字体，如黑体
    # matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots(figsize=(8, 6), dpi=int(style.get('dpi', 240)))

    # 设置图形标题
    ax.set_title(style.get('title'), fontsize=style.get('title_size', 16))

    # 设置x轴标签
    ax.set_xlabel(style.get('x_label'),
                  fontsize=style.get('x_fontsize', 14),
                  color=style.get('x_color', 'k'))

    # 设置y轴标签
    ax.set_ylabel(style.get('y_label'),
                  fontsize=style.get('y_fontsize', 14),
                  color=style.get('y_color', 'k'))
    
    # 设置刻度样式
    ax.tick_params(direction='in',
                   labelsize=style.get('t_size'),
                   which='both')
    
    def format_future_date(date):
        # 添加两小时到当前时间戳
        future_date = date + timedelta(hours=2)
        # 格式化日期为“日-时”
        return future_date.strftime('%d-%H')     
    
    # 获取数据总数
    data_count = len(data)

    # 设置刻度间隔
    tick_interval = 6

    # 计算刻度位置
    tick_positions = np.arange(0, data_count, tick_interval)

    tick_dates = pd.to_datetime(data['Formatted_Date'].iloc[tick_positions])

    # 设置 x 轴刻度为每隔4000个数据点的日期
    ax.set_xticks(tick_positions)
    # ax.set_xticklabels([date.strftime('%d-%H') for date in tick_dates])
    ax.set_xticklabels([format_future_date(date) for date in tick_dates])
    # 绘制数据
    ax.plot(data['Formatted_Date'], data['predictive_value'], label=f'{energy_level}: Predictive Value', linewidth=1)
    # ax.plot(data['Formatted_Date'], data['true_value'], color='red', label=f'{energy_level}：True Value', linewidth=1)

    # 添加制作单位到图形左下角
    unit_text = 'Created by:NCSW/CMA'
    unit_fontsize = 12
    unit_fontname = 'Microsoft YaHei'

    fig.text(0.03, 0.03, unit_text, fontsize=unit_fontsize, fontname=unit_fontname)

    logo_path = '/src/ElectronModel/util/logo.png'
    # 添加logo到图形最右下角
    logo = plt.imread(logo_path)
    imagebox = OffsetImage(logo, zoom=0.15)
    ab = AnnotationBbox(imagebox, (1, -0.1), xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)

    # 显示图例
    ax.legend()

    # 保存图形到指定路径
    plt.savefig(save_path)


def plot_graph_cn1(data, style, save_path, energy_level):
    """
    绘制图形函数，并保存到指定路径

    参数:
    - data: DataFrame, 要绘制的数据
    - style: dict, 样式配置项
    - save_path: str, 保存路径

    返回:
    无返回值
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots(figsize=(8, 6), dpi=int(style.get('dpi', 240)))

    # 设置图形标题
    ax.set_title(style.get('title'), fontsize=style.get('title_size', 16))

    # 设置x轴标签
    ax.set_xlabel(style.get('x_label'),
                  fontsize=style.get('x_fontsize', 14),
                  color=style.get('x_color', 'k'))

    # 设置y轴标签
    ax.set_ylabel(style.get('y_label'),
                  fontsize=style.get('y_fontsize', 14),
                  color=style.get('y_color', 'k'))

    # 设置刻度样式
    ax.tick_params(direction='in',
                   labelsize=style.get('t_size'),
                   which='both')

    # 获取数据总数
    data_count = len(data)

    # 设置刻度间隔
    tick_interval = 6

    # 计算刻度位置
    tick_positions = np.arange(0, data_count, tick_interval)

    tick_dates = pd.to_datetime(data['Formatted_Date'].iloc[tick_positions])

    # 设置 x 轴刻度为每隔4000个数据点的日期
    ax.set_xticks(tick_positions)

    def format_chinese_date(date):
        # 增加两小时
        future_date = date + timedelta(hours=2)
        month = future_date.strftime('%d').lstrip('0') + '日'
        day = future_date.strftime('%H').lstrip('0') + '时'
        return month + day

    # 将日期转换为中文格式'%月%日'
    ax.set_xticklabels([format_chinese_date(date) for date in tick_dates])

    # 绘制数据
    ax.plot(data['Formatted_Date'], data['predictive_value'], label=f'{energy_level}: 预测值', linewidth=1)
    # ax.plot(data['Formatted_Date'], data['true_value'], color='red', label=f'{energy_level}：真实值', linewidth=1)

    # 添加制作单位到图形左下角
    unit_text = '制作单位: 国家空间天气监测预警中心'
    unit_fontsize = 12
    unit_fontname = 'Microsoft YaHei'

    fig.text(0.03, 0.03, unit_text, fontsize=unit_fontsize, fontname=unit_fontname)

    logo_path = '/src/ElectronModel/util/logo.png'
    # 添加logo到图形最右下角
    logo = plt.imread(logo_path)
    imagebox = OffsetImage(logo, zoom=0.15)
    ab = AnnotationBbox(imagebox, (1, -0.1), xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)

    # 显示图例
    ax.legend()

    # 保存图形到指定路径
    plt.savefig(save_path)


def plot_graph_en2(data, style, save_path, energy_level):
    """
    绘制图形函数，并保存到指定路径

    参数:
    - data: DataFrame, 要绘制的数据
    - style: dict, 样式配置项
    - save_path: str, 保存路径

    返回:
    无返回值
    """
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文支持的字体，如黑体
    # matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots(figsize=(8, 6), dpi=int(style.get('dpi', 240)))

    # 设置图形标题
    ax.set_title(style.get('title'), fontsize=style.get('title_size', 16))

    # 设置x轴标签
    ax.set_xlabel(style.get('x_label'),
                  fontsize=style.get('x_fontsize', 14),
                  color=style.get('x_color', 'k'))

    # 设置y轴标签
    ax.set_ylabel(style.get('y_label'),
                  fontsize=style.get('y_fontsize', 14),
                  color=style.get('y_color', 'k'))
    
    # 设置刻度样式
    ax.tick_params(direction='in',
                   labelsize=style.get('t_size'),
                   which='both')
    
    # 设置y轴为对数刻度，并指定固定刻度
    ax.set_yscale('log')
    ax.set_ylim(10**0, 10**5)  # 设置y轴的范围从10^1到10^5
    ax.set_yticks([10**i for i in range(0, 5)])  # 设置固定的y轴刻度

    # 清晰设置FuncFormatter来定义y轴标签的格式化
    def format_func(value, tick_number):
        # 因为这是对数刻度，我们只在10的整数次幂处设置标签
        N = int(np.log10(value))
        if value == 10**N:
            return r'$10^{%d}$' % N
        return ''

    # 应用FuncFormatter到y轴的主要格式器
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    def format_future_date(date):
        # 添加两小时到当前时间戳
        future_date = date + timedelta(hours=2)
        # 格式化日期为“日-时”
        return future_date.strftime('%d-%H')    

    # 获取数据总数
    data_count = len(data)

    # 设置刻度间隔
    tick_interval = 6

    # 计算刻度位置
    tick_positions = np.arange(0, data_count, tick_interval)

    tick_dates = pd.to_datetime(data['Formatted_Date'].iloc[tick_positions])

    # 设置 x 轴刻度为每隔4000个数据点的日期
    ax.set_xticks(tick_positions)
    # ax.set_xticklabels([date.strftime('%d-%H') for date in tick_dates])
    ax.set_xticklabels([format_future_date(date) for date in tick_dates])    

    # 绘制数据
    ax.plot(data['Formatted_Date'], data['predictive_value'], label=f'{energy_level}: Predictive Value', linewidth=1)
    # ax.plot(data['Formatted_Date'], data['true_value'], color='red', label=f'{energy_level}：True Value', linewidth=1)

    # 添加制作单位到图形左下角
    unit_text = 'Created by:NCSW/CMA'
    unit_fontsize = 12
    unit_fontname = 'Microsoft YaHei'

    fig.text(0.03, 0.03, unit_text, fontsize=unit_fontsize, fontname=unit_fontname)

    logo_path = '/src/ElectronModel/util/logo.png'
    # 添加logo到图形最右下角
    logo = plt.imread(logo_path)
    imagebox = OffsetImage(logo, zoom=0.15)
    ab = AnnotationBbox(imagebox, (1, -0.1), xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)

    # 显示图例
    ax.legend()

    # 保存图形到指定路径
    plt.savefig(save_path)


def plot_graph_cn2(data, style, save_path, energy_level):
    """
    绘制图形函数，并保存到指定路径

    参数:
    - data: DataFrame, 要绘制的数据
    - style: dict, 样式配置项
    - save_path: str, 保存路径

    返回:
    无返回值
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots(figsize=(8, 6), dpi=int(style.get('dpi', 240)))

    # 设置图形标题
    ax.set_title(style.get('title'), fontsize=style.get('title_size', 16))

    # 设置x轴标签
    ax.set_xlabel(style.get('x_label'),
                  fontsize=style.get('x_fontsize', 14),
                  color=style.get('x_color', 'k'))

    # 设置y轴标签
    ax.set_ylabel(style.get('y_label'),
                  fontsize=style.get('y_fontsize', 14),
                  color=style.get('y_color', 'k'))

    # 设置刻度样式
    ax.tick_params(direction='in',
                   labelsize=style.get('t_size'),
                   which='both')
    
    # 设置y轴为对数刻度，并指定固定刻度
    ax.set_yscale('log')
    ax.set_ylim(10**0, 10**5)  # 设置y轴的范围从10^1到10^5
    ax.set_yticks([10**i for i in range(0, 5)])  # 设置固定的y轴刻度

    # 清晰设置FuncFormatter来定义y轴标签的格式化
    def format_func(value, tick_number):
        # 因为这是对数刻度，我们只在10的整数次幂处设置标签
        N = int(np.log10(value))
        if value == 10**N:
            return r'$10^{%d}$' % N
        return ''

    # 应用FuncFormatter到y轴的主要格式器
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    # 获取数据总数
    data_count = len(data)

    # 设置刻度间隔
    tick_interval = 6

    # 计算刻度位置
    tick_positions = np.arange(0, data_count, tick_interval)

    tick_dates = pd.to_datetime(data['Formatted_Date'].iloc[tick_positions])

    # 设置 x 轴刻度为每隔4000个数据点的日期
    ax.set_xticks(tick_positions)

    def format_chinese_date(date):
        future_date = date + timedelta(hours=2)
        month = future_date.strftime('%d').lstrip('0') + '日'
        day = future_date.strftime('%H').lstrip('0') + '时'
        return month + day

    # 将日期转换为中文格式'%月%日'
    ax.set_xticklabels([format_chinese_date(date) for date in tick_dates])

    # 绘制数据
    ax.plot(data['Formatted_Date'], data['predictive_value'], label=f'{energy_level}: 预测值', linewidth=1)
    # ax.plot(data['Formatted_Date'], data['true_value'], color='red', label=f'{energy_level}：真实值', linewidth=1)

    # 添加制作单位到图形左下角
    unit_text = '制作单位: 国家空间天气监测预警中心'
    unit_fontsize = 12
    unit_fontname = 'Microsoft YaHei'

    fig.text(0.03, 0.03, unit_text, fontsize=unit_fontsize, fontname=unit_fontname)

    logo_path = '/src/ElectronModel/util/logo.png'
    # 添加logo到图形最右下角
    logo = plt.imread(logo_path)
    imagebox = OffsetImage(logo, zoom=0.15)
    ab = AnnotationBbox(imagebox, (1, -0.1), xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)

    # 显示图例
    ax.legend()

    # 保存图形到指定路径
    plt.savefig(save_path)


def combine_images(left_image_path, right_image_path, save_path):
    """
    将左右两张图片组合成一张图片

    参数:
    - left_image_path: str, 左侧图片的文件路径
    - right_image_path: str, 右侧图片的文件路径
    - save_path: str, 保存路径

    返回:
    无返回值
    """
    # 打开左右两张图片
    left_image = Image.open(left_image_path)
    right_image = Image.open(right_image_path)

    # 获取左右两张图片的尺寸
    left_width, left_height = left_image.size
    right_width, right_height = right_image.size

    # 计算组合后图片的尺寸
    combined_width = left_width + right_width
    combined_height = max(left_height, right_height)

    # 创建一个新的空白图片
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # 将左右两张图片粘贴到新的图片上
    combined_image.paste(left_image, (0, 0))
    combined_image.paste(right_image, (left_width, 0))

    # 保存组合后的图片
    combined_image.save(save_path)


def main(csv_files, csv_path, png_path,log_scale_flag):
    csv2, csv800, csv475 = csv_files
    d_start = csv2.split("_")[1]
    d_end = csv2.split("_")[2]
    csv2 = os.path.join(csv_path, csv2)
    csv800 = os.path.join(csv_path, csv800)
    csv475 = os.path.join(csv_path, csv475)

    product_png_path = os.path.join(png_path, d_start[:4], d_end[:8])
    os.makedirs(product_png_path, exist_ok=True)
    mev2_file = f"{product_png_path}/Z_NAFP_C_DRAW-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_2mev_ZXGC_NCSW_V1A.png"
    mev2en_file = f"{product_png_path}/Z_NAFP_C_DRAW-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_2mev_ZXGC_NCSW_V1A_EN.png"
    mev2cn_file = f"{product_png_path}/Z_NAFP_C_DRAW-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_2mev_ZXGC_NCSW_V1A_CN.png"

    kev800_file = f"{product_png_path}/Z_NAFP_C_DRAW-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_800kev_ZXGC_NCSW_V1A.png"
    kev800en_file = f"{product_png_path}/Z_NAFP_C_DRAW-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_800kev_ZXGC_NCSW_V1A_EN.png"
    kev800cn_file = f"{product_png_path}/Z_NAFP_C_DRAW-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_800kev_ZXGC_NCSW_V1A_CN.png"

    kev475_file = f"{product_png_path}/Z_NAFP_C_DRAW-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_475kev_ZXGC_NCSW_V1A.png"
    kev475en_file = f"{product_png_path}/Z_NAFP_C_DRAW-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_475kev_ZXGC_NCSW_V1A_EN.png"
    kev475cn_file = f"{product_png_path}/Z_NAFP_C_DRAW-_{d_start}_P_OTHE_00_FLUX_L1_{d_end}_0005MP_475kev_ZXGC_NCSW_V1A_CN.png"
    data_2mev = load_data_from_csv(csv2)
    data_800kev = load_data_from_csv(csv800)
    data_475kev = load_data_from_csv(csv475)

    # 从配置文件加载样式配置项
    style_en = load_from_cfg_en("/src/ElectronModel/util/style.ini")
    style_cn = load_from_cfg_cn("/src/ElectronModel/util/style.ini")

    # 调用绘制图形函数
    plot_graph_en1(data_2mev, style_en, mev2en_file, '2MeV')
    plot_graph_en1(data_800kev, style_en, kev800en_file, '800KeV')
    plot_graph_en1(data_475kev, style_en, kev475en_file, '475KeV')

    plot_graph_cn1(data_2mev, style_cn, mev2cn_file, '2MeV')
    plot_graph_cn1(data_800kev, style_cn, kev800cn_file, '800KeV')
    plot_graph_cn1(data_475kev, style_cn, kev475cn_file, '475KeV')

    if(log_scale_flag == "True"):
        # 调用绘制图形函数
        plot_graph_en2(data_2mev, style_en, mev2en_file, '2MeV')
        plot_graph_en2(data_800kev, style_en, kev800en_file, '800KeV')
        plot_graph_en2(data_475kev, style_en, kev475en_file, '475KeV')

        plot_graph_cn2(data_2mev, style_cn, mev2cn_file, '2MeV')
        plot_graph_cn2(data_800kev, style_cn, kev800cn_file, '800KeV')
        plot_graph_cn2(data_475kev, style_cn, kev475cn_file, '475KeV')    

    # 组合图片
    combine_images(mev2cn_file, mev2en_file, mev2_file)
    combine_images(kev800cn_file, kev800en_file, kev800_file)
    combine_images(kev475cn_file, kev475en_file, kev475_file)

    print("图片已保存如下路径：")
    print(mev2_file)
    print(kev800_file)
    print(kev475_file)