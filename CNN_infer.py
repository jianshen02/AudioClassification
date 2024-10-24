import concurrent.futures
import librosa
import numpy as np
import tensorflow as tf
import time
import os

# 加载训练好的模型
model = tf.keras.models.load_model("./model/CNN_model_82.h5")

# 提取Mel频谱的函数
def extract_mel_spectrogram(y, sr, n_mels=128, n_fft=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

# 识别语音文件并输出结果及概率
def recognize_audio(file_path):
    try:
        # 加载音频文件
        y, sr = librosa.load(file_path, sr=22050)

        # 提取Mel频谱
        log_mel_spectrogram = extract_mel_spectrogram(y, sr)

        # 规范化大小
        log_mel_spectrogram_resized = librosa.util.fix_length(log_mel_spectrogram, size=128, axis=1)

        # 确保输入数据的形状为 (128, 128, 1)
        input_data = log_mel_spectrogram_resized[..., np.newaxis]
        input_data = np.expand_dims(input_data, axis=0)

        # 进行预测
        predictions = model.predict(input_data)

        # 输出预测结果及概率
        predicted_class = np.argmax(predictions, axis=1)
        probability = np.max(predictions)
        return predicted_class[0], probability
    except Exception as e:
        return None, str(e)

# 多线程或多进程并行处理
def parallel_recognition(file_paths, num_workers=4):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(recognize_audio, file_path): file_path for file_path in file_paths}
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append((file_path, result))
            except Exception as exc:
                print(f'{file_path} generated an exception: {exc}')
    return results

# 获取fold1目录下所有音频文件
def get_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    return audio_files

# fold1目录路径
audio_directory = "/home/ubuntu/Desktop/workspace/pythonProjects/AudioClassification-Tensorflow/dataset/UrbanSound8K/audio/fold1"

# 获取所有音频文件路径
audio_files = get_audio_files(audio_directory)

# 并行处理音频文件
results = parallel_recognition(audio_files, num_workers=8)

# 输出每个文件的识别结果
for file, result in results:
    if result[0] is not None:
        print(f"文件: {file} -> 识别结果: {result[0]}, 概率: {result[1]:.2f}")
    else:
        print(f"文件: {file} -> 错误: {result[1]}")
