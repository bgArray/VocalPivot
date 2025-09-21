# 文件流
import os
import sys
import time
from Constant import *
from io import StringIO

# ML
import numpy as np
from keras import layers, models, callbacks
from sklearn.model_selection import train_test_split


# 加载数据集
def load_dataset(data_dir: str = DATASET_PATH) -> tuple:
    """
    数据预处理
    :param data_dir: .npy数据集文件的存放位置
    :return: 返回处理好的频谱和标签
    """
    X = []
    y = []

    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            data = np.load(os.path.join(data_dir, file), allow_pickle=True)
            for item in data:
                mel_gram, tag = item  # 数据结构[[梅尔频谱], (int)声音分类标签]
                # 确保标签合法（过滤无效标签）
                if tag in [-1, 0, 1, 2]:
                    X.append(mel_gram)
                    y.append(tag)

    # 转换为numpy数组并标准化
    X = np.array(X)
    y = np.array(y)

    # 为梅尔频谱添加通道维度 (样本数, 高度, 宽度, 通道数)
    X = X[..., np.newaxis]

    # 标签映射为0开始的索引（-1 -> 3）
    y = np.where(y == -1, 3, y)

    return X, y


# 构建CNN模型
def build_cnn_model(input_shape):
    """
    构建CNN
    激活用ReLU
    3*3卷积核
    2*2池化
    3组卷积，卷积核数量：32-64-128
    :param input_shape: 图像尺寸
    :return: 模型
    """
    model_CNN = models.Sequential(
        [
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            # 第二个卷积块
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            # 第三个卷积块
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            # 全连接层
            layers.Flatten(),  # 二维特征图展平
            layers.Dense(128, activation="relu"),  # 全连接
            layers.Dropout(0.5),  # 丢掉50%的神经元，防止过拟合
            layers.Dense(4, activation="softmax"),  # 4类分类（-1/0/1/2）
        ]
    )

    # 编译模型
    model_CNN.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # 多分类问题整数标签
        metrics=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    )

    return model_CNN


# 训练模型
def train_model():
    # 加载数据
    X, y = load_dataset()
    print(f"数据集规模: {X.shape}, 标签数量: {y.shape}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 获取输入形状 (高度, 宽度, 通道数)
    input_shape = X_train[0].shape

    # 构建模型
    model_r = build_cnn_model(input_shape)
    # 将 summary 转为字符串
    summary_str = StringIO()
    sys.stdout = summary_str  # 重定向标准输出到 StringIO
    model_r.summary()
    sys.stdout = sys.__stdout__  # 恢复标准输出

    # 训练模型
    history_r = model_r.fit(
        X_train,
        y_train,
        epochs=30,  # 训练轮次
        batch_size=32,  # 每一组的样本数
        validation_split=0.2,
        callbacks=[
            callbacks.EarlyStopping(
                patience=5, restore_best_weights=True
            ),  # 5轮没有提升提前终止 返回最优解
            callbacks.ReduceLROnPlateau(
                factor=0.5, patience=3
            ),  # 指标停滞时 学习率变为原来的一半 连续 3 轮没有提升就调整学习率
        ],
    )

    # 评估模型，会返回损失和所有指定的指标
    test_loss, test_acc, test_precision, test_recall, test_f1 = model.evaluate(
        X_test, y_test
    )

    # 打印所有指标
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集召回率: {test_recall:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")

    # 保存模型
    model_r.save("vocal_tech_cnn.h5")
    print("模型已保存为 vocal_tech_cnn.h5")

    # 实验结果简报
    with open(REPORTS_PATH + "训练数据（{0}）.txt".format(time.time()), "w") as f:
        f.write(
            "模型结构：\n"
            "{0}\n"
            "结果指标：\n"
            "测试集损失: {1}\n"
            "测试集准确率: {2}\n"
            "测试集精确率: {3}\n"
            "测试集召回率: {4}\n"
            "测试集F1分数: {5}\n".format(
                summary_str.getvalue(),
                test_loss,
                test_acc,
                test_precision,
                test_recall,
                test_f1,
            )
        )

    return model_r, history_r


if __name__ == "__main__":
    model, history = train_model()
