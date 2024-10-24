import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 1. 数据预处理
# 1.1 加载音频文件
def load_audio_file(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

# 1.2 提取Mel频谱
def extract_mel_spectrogram(y, sr, n_mels=128, n_fft=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

# 1.3 数据增强
# 添加随机时间平移
def time_shift(y):
    shift = np.random.randint(int(0.1 * len(y)))
    direction = np.random.choice(['left', 'right'])
    if direction == 'right':
        y_shifted = np.pad(y, (shift, 0), mode='constant')[:len(y)]
    else:
        y_shifted = np.pad(y, (0, shift), mode='constant')[shift:]
    return y_shifted

# 改变音调
def pitch_shift(y, sr):
    n_steps = np.random.uniform(-2, 2)  # 随机选择音调变化幅度，范围可以是 -2 到 2 个半音
    y_shifted = librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=n_steps)
    return y_shifted

# 动态范围压缩
def dynamic_range_compression(y):
    return librosa.effects.percussive(y, margin=3.0)

# 随机剪切
def random_crop(y, crop_len):
    if len(y) > crop_len:
        start = np.random.randint(0, len(y) - crop_len)
        y = y[start:start + crop_len]
    else:
        y = librosa.util.fix_length(y, size=crop_len)
    return y

# 数据增强
def augment_audio(y, sr):
    # 定义一个应用增强的概率
    augmentation_prob = 0.8  # 80%的概率应用增强

    if np.random.rand() < augmentation_prob:
        # 添加随机噪声
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise

        # 随机选择一种或多种增强方式
        augmentations = [
            lambda y: time_shift(y),
            lambda y: pitch_shift(y, sr),
            lambda y: dynamic_range_compression(y),
            lambda y: random_crop(y, len(y))
        ]

        num_augmentations = np.random.randint(1, len(augmentations) + 1)
        selected_augmentations = np.random.choice(augmentations, num_augmentations, replace=False)

        for augment in selected_augmentations:
            y = augment(y)

        # 改变速度
        speed_factor = np.random.uniform(0.9, 1.1)
        if len(y) > sr:
            y = librosa.effects.time_stretch(y.astype(np.float32), rate=speed_factor)

    return y

# 1.4 数据集准备
# 假设数据集的元数据位于 "dataset/UrbanSound8K/metadata/UrbanSound8K.csv"
metadata = pd.read_csv("dataset/UrbanSound8K/metadata/UrbanSound8K.csv")

features = []
labels = []

for index, row in metadata.iterrows():
    file_path = os.path.join("dataset/UrbanSound8K/audio", f"fold{row['fold']}", row['slice_file_name'])
    y, sr = load_audio_file(file_path)

    # 数据增强
    y = augment_audio(y, sr)

    log_mel_spectrogram = extract_mel_spectrogram(y, sr)

    # 规范化大小，例如将频谱图的尺寸调整为 (128, 128)
    log_mel_spectrogram_resized = librosa.util.fix_length(log_mel_spectrogram, size=128, axis=1)

    features.append(log_mel_spectrogram_resized)
    labels.append(row['classID'])

# 分割数据集
X = np.array(features)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 确保输入数据的形状为 (128, 128, 1)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# 改进后的模型架构
inputs = layers.Input(shape=(128, 128, 1))

# 残差块定义
def residual_block(x, filters):
    shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)  # 调整 shortcut 的形状
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])  # 残差连接
    x = layers.Activation('relu')(x)
    return x

x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.1)(x)

# 添加残差块
x = residual_block(x, 64)
x = layers.MaxPooling2D((2, 2))(x)

x = residual_block(x, 128)
x = layers.MaxPooling2D((2, 2))(x)

x = residual_block(x, 256)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)

# 编译模型
learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 设置学习率调度器和早停
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型
history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr, early_stopping])

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("\n测试集准确率:", test_acc)

# 可视化训练结果
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='lower right', prop={'size': 10})
plt.show()

# 保存模型
model.save("trained_model.h5")