import os
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

# UrbanSound8K 数据集路径
dataset_path = "/home/ubuntu/Desktop/workspace/pythonProjects/AudioClassification-Tensorflow/dataset/UrbanSound8K"
metadata_path = os.path.join(dataset_path, "metadata/UrbanSound8K.csv")

# 加载元数据
metadata = pd.read_csv(metadata_path)
print("======已经加载元数据======")


# 获取音频文件路径
def get_audio_path(fold, file_name):
    return os.path.join(dataset_path, "audio", f"fold{fold}", file_name)


# 固定的波形长度 (例如：16000 表示 1 秒钟的音频)
TARGET_LENGTH = 16000  # 1 秒的音频，采样率为 16000


# 提取音频波形，并标准化长度
def extract_waveform(file_path, target_sr=16000, target_length=TARGET_LENGTH):
    try:
        waveform, sr = librosa.load(file_path, sr=target_sr)
        if len(waveform) > target_length:
            # 如果波形长度大于目标长度，进行截断
            waveform = waveform[:target_length]
        else:
            # 如果波形长度小于目标长度，进行填充
            padding = target_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), 'constant')
        return waveform
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


# 加载并预处理数据集
def load_dataset():
    waveforms, labels = [], []
    for index, row in metadata.iterrows():
        file_path = get_audio_path(row["fold"], row["slice_file_name"])
        label = row["classID"]
        waveform = extract_waveform(file_path)
        if waveform is not None:
            waveforms.append(waveform)
            labels.append(label)
    return np.array(waveforms), np.array(labels)


# 加载数据
X, y = load_dataset()
print("======已经加载数据======")

# 加载 YAMNet 模型
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("======已经加载YAMNet模型======")


# 使用 YAMNet 提取特征，并对帧取平均值
def extract_features_from_waveforms(waveforms):
    embeddings_list = []
    for waveform in waveforms:
        scores, embeddings, spectrogram = yamnet_model(waveform)
        # 对时间帧取平均值，得到一个 (1024,) 的特征向量
        embeddings_mean = np.mean(embeddings, axis=0)
        embeddings_list.append(embeddings_mean)
    return np.array(embeddings_list)


# 提取训练集的特征
X_features = extract_features_from_waveforms(X)
print("======已经提取数据集特征======")

# 确保数据形状正确
X_features = np.squeeze(X_features)


# 构建分类模型，使用 LeakyReLU 作为激活函数
def build_model(input_shape):
    model = Sequential([
        Dense(2048, kernel_regularizer=regularizers.l2(0.0001), input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),

        Dense(1024, kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),

        Dense(512, kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),

        Dense(256, kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),

        Dense(128, kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),

        Dense(64, kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),

        Dense(32, kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),

        # 输出层，10 个类别
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 设置 K-fold 交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_no = 1
for train_index, val_index in kf.split(X_features):
    print(f"====== 正在训练第 {fold_no} 折 ======")

    # 训练集和验证集的分割
    X_train, X_val = X_features[train_index], X_features[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 创建模型
    input_shape = (1024,)
    model = build_model(input_shape)

    # 训练模型
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val),
              callbacks=[early_stopping, reduce_lr])

    # 评估模型
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy for fold {fold_no}: {accuracy:.4f}")

    # 增加折数
    fold_no += 1

# 保存模型
model.save("YAMNet_with_LeakyReLU_kfold.h5")
