import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import time

# 加载模型
model_path = "./models/YAMNet_model_873.h5"
model = tf.keras.models.load_model(model_path)

# 加载 YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# 预处理和特征提取函数
def preprocess_and_extract_features(file_path, target_sr=16000, target_length=16000):
    waveform, sr = librosa.load(file_path, sr=target_sr)
    if len(waveform) > target_length:
        waveform = waveform[:target_length]
    else:
        padding = target_length - len(waveform)
        waveform = np.pad(waveform, (0, padding), 'constant')
    scores, embeddings, spectrogram = yamnet_model(waveform)
    embeddings_mean = np.mean(embeddings, axis=0)
    return embeddings_mean

# 预测函数，返回预测类别和对应的概率
def predict_audio_class_with_probability(file_path):
    features = preprocess_and_extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # 增加批次维度
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_probability = np.max(predictions, axis=1)
    return predicted_class[0], predicted_probability[0]

# 文件夹路径
folder_path = '/home/ubuntu/Desktop/workspace/pythonProjects/AudioClassification-Tensorflow/dataset/UrbanSound8K/audio/fold1'

start_time = time.time()
# 处理文件夹中的所有音频文件
results = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if file_path.endswith(".wav"):
        predicted_class, predicted_probability = predict_audio_class_with_probability(file_path)
        results.append((filename, predicted_class, predicted_probability))
        print(f"File: {filename} - Predicted class ID: {predicted_class}, Probability: {predicted_probability:.4f}")

end_time = time.time()
print(f"串行代码执行时间: {end_time - start_time:.4f} 秒")

# 可选：保存结果到文件
import pandas as pd
results_df = pd.DataFrame(results, columns=['Filename', 'PredictedClass', 'Probability'])
results_df.to_csv('predictions.csv', index=False)
