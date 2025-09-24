# 文件流
import os
import sys
import time
from Constant import *
from io import StringIO

# ML
import numpy as np
from keras import layers, models, callbacks, regularizers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)  # 导入sklearn的指标函数  可能每个版本不一样
import tensorflow as tf

t1 = time.time()  # 总时间开始


# 加载数据集
def load_dataset_M1(data_dir: str = DATASET_PATH, count: int = -1) -> tuple:
    """
    数据预处理

    这个函数是用来处理M1真假混声 再加上说话的数据集的
    后面要改的话不要动这个函数
    :param count: 要加载多少个数据文件
    :param data_dir: .npy数据集文件的存放位置
    :return: 返回处理好的频谱和标签
    """
    X = []
    y = []
    counter = 0
    # if count != -1:

    for file in os.listdir(data_dir):
        if counter == count:
            break
        if file.endswith(".npy"):
            data = np.load(os.path.join(data_dir, file), allow_pickle=True)
            for item in data:
                mel_gram, tag = item  # 数据结构[[梅尔频谱], (int)声音分类标签]
                # 确保标签合法（过滤无效标签）
                if tag in [-1, 0, 1, 2]:  # 这里是映射 -1说话 0真声 1混声 2假声
                    X.append(mel_gram)
                    y.append(tag)
        counter += 1

    # 转换为numpy数组并标准化
    X = np.array(X)
    y = np.array(y)

    # 为梅尔频谱添加通道维度 (样本数, 高度, 宽度, 通道数)
    X = X[..., np.newaxis]

    # 标签映射为0开始的索引（-1 -> 3）
    y = np.where(y == -1, 3, y)

    return X, y


def load_dataset_M1_normalization(
    data_dir: str = DATASET_PATH, count: int = -1
) -> tuple:
    """

    :param data_dir:
    :param count:
    :return:
    """
    # X = []
    y = []
    counter = 0

    # 第一遍遍历：收集所有数据计算全局统计量
    all_data = []
    for file in os.listdir(data_dir):
        if counter == count:
            break
        if file.endswith(".npy"):
            data = np.load(os.path.join(data_dir, file), allow_pickle=True)
            for item in data:
                mel_gram, tag = item
                if tag in [-1, 0, 1, 2]:
                    all_data.append(mel_gram)
                    y.append(tag)
        counter += 1

    # 计算全局均值和标准差
    all_data_np = np.array(all_data)
    global_mean = np.mean(all_data_np)
    global_std = np.std(all_data_np)
    epsilon = 1e-8  # 防止除零

    # 归一化处理
    X = [(mel - global_mean) / (global_std + epsilon) for mel in all_data]
    X = np.array(X)
    y = np.array(y)

    # 添加通道维度并处理标签
    X = X[..., np.newaxis]
    y = np.where(y == -1, 3, y)

    print(f"数据归一化完成 - 均值: {global_mean:.4f}, 标准差: {global_std:.4f}")
    return X, y


# 构建CNN模型
# 自注意力层实现
# 修复后的自注意力层实现
class SelfAttention(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.filters = filters
        # 卷积层用于生成查询、键、值
        self.query_conv = layers.Conv2D(filters // 8, kernel_size=1, padding="same")
        self.key_conv = layers.Conv2D(filters // 8, kernel_size=1, padding="same")
        self.value_conv = layers.Conv2D(filters, kernel_size=1, padding="same")
        self.gamma = self.add_weight(
            name="gamma", shape=[1], initializer="zeros", trainable=True
        )
        self.built = False  # 标记是否已构建

    def build(self, input_shape):
        # 显式构建方法，确保层正确初始化
        super(SelfAttention, self).build(input_shape)
        self.built = True

    def call(self, x):
        # 使用Keras的layers.Reshape代替直接调用.reshape()
        # x shape: (batch_size, height, width, channels)
        batch_size = layers.Lambda(lambda x: tf.shape(x)[0])(x)
        height = x.shape[1]
        width = x.shape[2]
        channels = x.shape[3]
        spatial_size = height * width

        # 生成查询、键、值并使用Keras层进行reshape
        query = self.query_conv(x)
        query = layers.Reshape((spatial_size, -1))(query)

        key = self.key_conv(x)
        key = layers.Reshape((spatial_size, -1))(key)

        value = self.value_conv(x)
        value = layers.Reshape((spatial_size, -1))(value)

        # 计算注意力权重 (batch_size, spatial_size, spatial_size)
        attention = layers.Dot(axes=-1)([query, key])
        attention = layers.Softmax(axis=-1)(attention)

        # 应用注意力权重 (batch_size, spatial_size, channels)
        out = layers.Dot(axes=-2)([attention, value])
        out = layers.Reshape((height, width, channels))(out)  # 恢复空间维度

        # 残差连接
        out = layers.Lambda(lambda x: self.gamma * x[0] + x[1])([out, x])
        return out

    def compute_output_shape(self, input_shape):
        # 显式定义输出形状，解决Keras无法推断的问题
        return input_shape


def build_cnn_model_1():
    pass


def build_cnn_model_2():
    pass


def build_cnn_model_3(input_shape):
    print("调用模型3")
    model_CNN = models.Sequential(
        [
            # 第一个卷积块（调整卷积核数量、BN位置）
            layers.Conv2D(16, (3, 3), input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # 第二个卷积块
            layers.Conv2D(32, (3, 3)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # 第三个卷积块（非对称卷积核 强化频域空间）
            layers.Conv2D(64, (5, 3)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # 第四个卷积块（去掉池化）
            layers.Conv2D(128, (3, 3)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.2),
            # 全连接层简化
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(4, activation="softmax"),
        ]
    )

    # 优化器调整初始学习率
    model_CNN.compile(
        optimizer=optimizers.Adam(),  # 学快一点  learning_rate=0.001 如果要调快最好batch size等比例
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model_CNN


def build_cnn_model_4(input_shape):
    """
    带有自注意力机制、优化drop，防止过拟合的新模型
            带有自注意力机制的优化模型
        增强过拟合优化：动态dropout、L2正则化、注意力机制辅助特征选择
    :param input_shape:
    :return:
    """
    print("调用模型4（带自注意力机制）")

    # 输入层
    inputs = layers.Input(shape=input_shape)

    # 第一个卷积块
    x = layers.Conv2D(
        32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)  # 空间dropout更适合卷积层

    # 第二个卷积块
    x = layers.Conv2D(
        64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # 第三个卷积块 + 自注意力
    x = layers.Conv2D(
        128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = SelfAttention(128)(x)  # 应用自注意力机制
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.3)(x)

    # 第四个卷积块
    x = layers.Conv2D(
        256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout2D(0.3)(x)

    # 特征融合与分类
    x = layers.GlobalAveragePooling2D()(x)  # 全局平均池化减少参数
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        x
    )
    x = layers.AlphaDropout(0.5)(x)  # AlphaDropout适合ReLU后的 dropout
    outputs = layers.Dense(4, activation="softmax")(x)

    # 构建模型
    model_CNN = models.Model(inputs=inputs, outputs=outputs)

    # 优化器设置（带学习率调度）
    optimizer = optimizers.Adam(learning_rate=0.001, decay=1e-5)  # 学习率衰减

    model_CNN.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model_CNN


model_list = [
    build_cnn_model_1,
    build_cnn_model_2,
    build_cnn_model_3,
    build_cnn_model_4,
]


# 训练模型（修改数据加载方式为归一化版本）
def train_model(
    model_in: int,
    learning_diminish: float,
    learning_patience: int = 3,
    epo: int = 50,
    b_size: int = 32,
):
    # 使用归一化后的数据加载
    X, y = load_dataset_M1_normalization()
    print(f"数据集规模: {X.shape}, 标签数量: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    input_shape = X_train[0].shape
    print(input_shape)

    model_r = model_list[model_in](input_shape)
    summary_str = StringIO()
    sys.stdout = summary_str
    model_r.summary()
    sys.stdout = sys.__stdout__

    # 增强过拟合控制的回调
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_f1",
            patience=8,
            restore_best_weights=True,  # 监控验证集F1分数
            mode="max",  # 明确指定模式为最大化，因为F1分数越高越好
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=learning_diminish,
            patience=learning_patience,
            min_lr=1e-6,  # 最小学习率限制
        ),
        callbacks.ModelCheckpoint(
            "best_model.h5", monitor="val_accuracy", save_best_only=True
        ),
    ]

    history_r = model_r.fit(
        X_train,
        y_train,
        epochs=epo,
        batch_size=b_size,
        validation_split=0.2,
        callbacks=callbacks_list,
    )

    # 评估部分保持不变
    test_loss, test_acc = model_r.evaluate(X_test, y_test)
    y_pred_probs = model_r.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_precision = precision_score(y_test, y_pred, average="macro")
    test_recall = recall_score(y_test, y_pred, average="macro")
    test_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集召回率: {test_recall:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")

    model_r.save("vocal_tech_cnn.h5")
    print("模型已保存为 vocal_tech_cnn.h5")

    actual_epochs = len(history_r.epoch)

    with open(
        REPORTS_PATH + "训练数据（{0}）.txt".format(time.time()), "w", encoding="utf-8"
    ) as f:
        f.write(
            "模型结构：\n"
            "{0}\n"
            "模型序号：{1}\n"
            "调整学习率系数：{2}\n"
            "数据集规模参数：{3}\n"
            "结果指标：\n"
            "测试集损失: {4:.4f}\n"
            "测试集准确率: {5:.4f}\n"
            "测试集精确率: {6:.4f}\n"
            "测试集召回率: {7:.4f}\n"
            "测试集F1分数: {8:.4f}\n"
            "实际训练轮次: {9} 轮\n"
            "训练时长： {10:.2f}秒\n".format(
                summary_str.getvalue(),
                model_in + 1,
                learning_diminish,
                X.shape,
                test_loss,
                test_acc,
                test_precision,
                test_recall,
                test_f1,
                actual_epochs,
                time.time() - t1,
            )
        )

    return model_r, history_r


if __name__ == "__main__":
    # 测试模型4
    model, history = train_model(3, 0.7, 5)
