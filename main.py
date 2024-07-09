from src import prediction_model
from util import image_processing
from src import data_processing_for_csv
import sys
from datetime import datetime
from configparser import ConfigParser

config = ConfigParser()
config.read("/src/ElectronModel/config.ini")

# 六类基础数据绝对路径
kp_path = config.get("path", "kp")
energy1_path = config.get("path", "energy1")
energy2_path = config.get("path", "energy2")
tyf_path = config.get("path", "tyf")
by_bz = config.get("path", "by_bz")
mlt_path = config.get("path", "mlt")

# 算法固定输入数据位置
input_path = config.get("path", "program_input")
# 算法运行时的参数配置文件
run_config = config.get("path", "run_config")
# 算法过程中的csv文件存储路径
csv_path = config.get("path", "csv")
# 算法json产品位置
json_path = config.get("path", "json")
# 算法png产品位置
png_path = config.get("path", "png")

# 图片是否需要log_scale
log_scale_flag = config.get("path", "log_scale_flag")

if __name__ == "__main__":
    data_end_time = datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S")
    data_start_time = datetime.strptime(sys.argv[2], "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S")

    data_processing_for_csv.main(kp_path,energy1_path,energy2_path,tyf_path,by_bz,mlt_path,input_path,run_config)
    csv_results = prediction_model.main(input_path, data_start_time, data_end_time, csv_path, json_path)
    image_processing.main(csv_results, csv_path, png_path, log_scale_flag)