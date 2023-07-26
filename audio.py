import pyaudio
import wave
import whisper
import torch



# 设置参数
output_file = './temp/tempAudio.wav'  # 音频文件保存路径
duration = 5  # 录制时长（秒）
sample_rate = 16000  # 采样率

# 模型地址
# C:\Users\HANLU\.cache\whisper
finetune_model = "./whisper-medium-finetune/pytorch_model.bin"

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
print("开始识别输入.")
model = whisper.load_model("medium")
result = model.transcribe(output_file,language="zh")
print(result["text"])
