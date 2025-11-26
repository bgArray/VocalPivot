import os
from Constant import *
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 加载数据
def load_data(file_path):
    """加载.npy数据，提取0/1/2标签并确保特征与标签长度一致"""
    data = []
    for filename in os.listdir(file_path):
        if filename.endswith('.npy'):
            file_full_path = os.path.join(file_path, filename)
            data.extend(np.load(file_full_path, allow_pickle=True))

    X = []  # 存储mel特征序列（每个元素是一个样本的序列）
    y = []  # 存储标签序列（每个元素是一个样本的标签序列）

    for item in data:
        mel_spectrogram, tags = item

        # 确保mel和标签的时间步数量一致
        if len(mel_spectrogram) != len(tags):
            print(f"警告：mel长度({len(mel_spectrogram)})与标签长度({len(tags)})不匹配，已跳过该样本")
            continue

        # 处理每个时间步的标签：只保留0/1/2，取第一个有效标签（无则标记为-1）
        valid_tags = []
        for tag_group in tags:
            filtered = [t for t in tag_group if t in {"0", "1", "2"}]
            valid_tags.append(filtered[0] if filtered else "-1")  # 用-1表示无有效标签

        # 保留整个序列（包含所有时间步，包括无标签的）
        X.append(mel_spectrogram)
        y.append(valid_tags)

    return X, y


# 数据预处理
def preprocess_data(X, y):
    """预处理序列数据，适配LSTM输入"""
    # 1. 处理标签：将所有标签（包括-1）编码
    # 先将嵌套的标签序列扁平化为一维列表，用于训练编码器
    all_tags = [tag for seq in y for tag in seq]
    le = LabelEncoder()
    le.fit(all_tags)  # 拟合所有可能的标签（0/1/2/-1）

    # 将每个样本的标签序列编码为数字
    y_encoded = [le.transform(seq) for seq in y]

    # 2. 处理特征序列：统一长度
    max_length = max(len(seq) for seq in X)  # 所有样本的最大序列长度

    # 填充特征序列（后填充0）
    X_padded = sequence.pad_sequences(
        X,
        maxlen=max_length,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )

    # 填充标签序列（后填充-1对应的编码）
    y_padded = sequence.pad_sequences(
        y_encoded,
        maxlen=max_length,
        padding='post',
        truncating='post',
        value=le.transform(["-1"])[0]  # 用-1的编码填充
    )

    # 转换为独热编码（类别数=4：0/1/2/-1）
    num_classes = len(le.classes_)
    y_onehot = to_categorical(y_padded, num_classes=num_classes)

    return X_padded, y_onehot, max_length, le


# 构建LSTM模型（适配序列标注任务）
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),  # 忽略填充的0值
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),  # 返回所有时间步的输出
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),  # 保持序列结构
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # 每个时间步预测一个类别
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 主函数
def main(data_path):
    # 加载数据
    X, y = load_data(data_path)
    print(f"加载数据完成，样本数: {len(X)}, 每个样本为序列数据")

    # 预处理
    X_padded, y_onehot, max_length, le = preprocess_data(X, y)
    # 输入形状：(样本数, 时间步, 特征数)
    input_shape = (max_length, X_padded.shape[2])
    num_classes = y_onehot.shape[2]  # 独热编码的类别数

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_onehot, test_size=0.2, random_state=42
    )

    # 构建模型
    model = build_lstm_model(input_shape, num_classes)
    model.summary()

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        shuffle=True
    )

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {accuracy:.4f}")

    # 保存模型和编码器
    model.save('lstm_speech_classifier.h5')
    np.save('label_encoder.npy', le.classes_)  # 保存编码器类别
    print("模型和标签编码器保存完成")


if __name__ == "__main__":
    # DATASET_PATH = "your_dataset_path"  # 替换为你的数据路径
    main(DATASET_PATH)
