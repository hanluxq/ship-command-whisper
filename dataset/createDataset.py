


import os
import glob
import json

folder_path = "3"  # 文件夹路径

# 使用glob模块匹配文件夹中所有的WAV文件
wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

data = []

for wav_file in wav_files:
    wav_filename = os.path.basename(wav_file)
    textgrid_file = os.path.join(folder_path, wav_filename.replace(".wav", ".TextGrid"))

    # 检查同名的TextGrid文件是否存在
    if os.path.exists(textgrid_file):
        with open(textgrid_file, "r", encoding="utf-16") as f:
            lines = f.readlines()

        # 提取文本信息
        text = ""
        duration = 0
        for line in lines:
            if "xmax" in line:
                duration = round(float(line.strip().split()[-1]),2)
            if "text" in line:
                text = line.strip().split()[-1].strip('"')
                # 保存为JSON格式
                audio_data = {
                    "path": f"dataset/"+folder_path+"/"+wav_filename
                }
                output = {
                    "audio": audio_data,
                    "sentence": text,
                    "duration": duration
                }
                data.append(output)
                break

# 保存为JSON文件
output_path = folder_path+".json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
