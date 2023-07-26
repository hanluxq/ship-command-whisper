import argparse
import functools

import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.utils import print_arguments, add_arguments
import pyaudio
import wave
import whisper

# 设置参数
output_file = './temp/tempAudio.wav'  # 音频文件保存路径
duration = 5  # 录制时长（秒）
sample_rate = 16000  # 采样率

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path", type=str, default="./temp/tempAudio.wav",              help="预测的音频路径")
#add_arg("model_path", type=str, default="../whisper-medium-finetune",  help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("model_path", type=str, default="../Whisper-Finetune/models/medium10",  help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("language",   type=str, default="Chinese",                       help="设置语言，可全称也可简写，如果为None则预测的是多语言")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("local_files_only", type=bool, default=True,  help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

# 录制语音
chunk = 1024
format = pyaudio.paInt16
channels = 1

p = pyaudio.PyAudio()

stream = p.open(format=format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk)

print("开始录音...")

frames = []

for i in range(0, int(sample_rate / chunk * duration)):
    data = stream.read(chunk)
    frames.append(data)

print("录音结束.")

stream.stop_stream()
stream.close()
p.terminate()

# 保存音频文件
wf = wave.open(output_file, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(format))
wf.setframerate(sample_rate)
wf.writeframes(b''.join(frames))
wf.close()

print("音频文件保存成功:", output_file)

# 获取Whisper的特征提取器、编码器和解码器
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only).half()
model.eval()

# 读取音频
sample, sr = librosa.load(args.audio_path, sr=16000)
duration = sample.shape[-1]/sr
assert duration < 30, f"本程序只适合推理小于30秒的音频，当前音频{duration}秒，请使用其他推理程序!"
# 预处理音频
input_features = processor(sample, sampling_rate=sr, return_tensors="pt", do_normalize=True).input_features.cuda().half()
# 开始识别
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256)
# 解码结果
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("开始识别输入.")
print(f"识别结果：{transcription}")
