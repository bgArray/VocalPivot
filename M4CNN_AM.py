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
from sklearn.metrics import precision_score, recall_score, f1_score  # 导入sklearn的指标函数  可能每个版本不一样
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
    # import matplotlib.pyplot as plt
    # plt.imshow(X[0, :, :, 0], cmap='viridis')  # 显示第一个样本的梅尔频谱
    # plt.show()
    return X, y


# def load_dataset_M1_normalization(
#     data_dir: str = DATASET_PATH, count: int = -1
# ) -> tuple:
#     """
#
#     :param data_dir:
#     :param count:
#     :return:
#     """
#     # X = []
#     y = []
#     counter = 0
#
#     # 第一遍遍历：收集所有数据计算全局统计量
#     all_data = []
#     for file in os.listdir(data_dir):
#         if counter == count:
#             break
#         if file.endswith(".npy"):
#             data = np.load(os.path.join(data_dir, file), allow_pickle=True)
#             for item in data:
#                 mel_gram, tag = item
#                 if tag in [-1, 0, 1, 2]:
#                     all_data.append(mel_gram)
#                     y.append(tag)
#         counter += 1
#
#     # 计算全局均值和标准差
#     all_data_np = np.array(all_data)
#     # global_mean = np.mean(all_data_np)
#     # global_std = np.std(all_data_np)
#     epsilon = 1e-8  # 防止除零
#     # 在 load_dataset_M1_normalization 中替换归一化方式
#     global_min = np.min(all_data_np)
#     global_max = np.max(all_data_np)
#     X = [(mel - global_min) / (global_max - global_min + epsilon) for mel in all_data]
#
#     # 归一化处理
#     # X = [(mel - global_mean) / (global_std + epsilon) for mel in all_data]
#     X = np.array(X)
#     y = np.array(y)
#
#     # 添加通道维度并处理标签
#     X = X[..., np.newaxis]
#     y = np.where(y == -1, 3, y)
#
#     # print(f"数据归一化完成 - 均值: {global_mean:.4f}, 标准差: {global_std:.4f}")
#     import matplotlib.pyplot as plt
#     plt.imshow(X[0, :, :, 0], cmap='viridis')  # 显示第一个样本的梅尔频谱
#     plt.show()
#     return X, y


# 构建CNN模型
def build_cnn_model_1(input_shape):
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

    # 编译模型（仅保留accuracy作为内置指标）
    model_CNN.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model_CNN


def build_cnn_model_2(input_shape):
    """
    构建CNN
    激活用ReLU
    3*3卷积核
    2*2池化
    4组卷积，卷积核数量：32-64-128-256
    :param input_shape: 图像尺寸
    :return: 模型
    """
    print("调用模型2")
    model_CNN = models.Sequential(
        [
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            # 第二个卷积块
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            # 第三个卷积块
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            # 第四个卷积块
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            # 过渡层
            layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            # 全连接层
            layers.Flatten(),  # 二维特征图展平
            layers.Dense(128, activation="relu"),  # 全连接
            layers.Dropout(0.5),  # 丢掉50%的神经元，防止过拟合
            layers.Dense(4, activation="softmax"),  # 4类分类（-1/0/1/2）
        ]
    )

    # 编译模型（仅保留accuracy作为内置指标）
    model_CNN.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model_CNN


def build_cnn_model_3(input_shape):
    print("调用模型3")
    model_CNN = models.Sequential([
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
    ])

    # 优化器调整初始学习率
    model_CNN.compile(
        optimizer=optimizers.Adam(),  # 学快一点  learning_rate=0.001 如果要调快最好batch size等比例
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model_CNN


def build_cnn_model_4(input_shape):
    """
    构建CNN
    激活用ReLU
    3*3卷积核
    2*2池化
    4组卷积，卷积核数量：32-64-128-256
    :param input_shape: 图像尺寸
    :return: 模型
    """
    print("调用模型4")
    model_CNN = models.Sequential(
        [
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # 第二个卷积块
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # 第三个卷积块
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # 第四个卷积块
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # 过渡层
            # 新增：全局平均池化（GAP），减少参数，缓解过拟合
            layers.GlobalAveragePooling2D(),  # 替代Conv2D(32, ...)和Flatten
            # layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            # 全连接层
            # layers.Flatten(),  # 二维特征图展平
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),  # 减少神经元数+L2正则
            layers.Dropout(0.6),  # 全连接层Dropout从0.5提至0.6
            layers.Dense(4, activation="softmax"),
        ]
    )

    # 编译模型（仅保留accuracy作为内置指标）
    model_CNN.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model_CNN


def build_cnn_model_5(input_shape):
    """
    基于模型2添加空间注意力机制的CNN模型
    保留模型2的4组卷积块结构，在特征提取后加入注意力机制
    """
    print("调用模型5（带注意力机制）")

    # 输入层
    inputs = layers.Input(shape=input_shape)

    # 第一个卷积块（模型2基础结构）
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # 第二个卷积块
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # 第三个卷积块
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # 第四个卷积块
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # 空间注意力机制（新增部分）
    # 1. 计算通道注意力
    channel_avg = layers.GlobalAveragePooling2D()(x)  # 通道平均池化 (batch, channels)
    channel_max = layers.GlobalMaxPooling2D()(x)  # 通道最大池化 (batch, channels)
    channel_att = layers.Concatenate()([channel_avg, channel_max])
    channel_att = layers.Dense(x.shape[-1] // 4, activation="relu")(channel_att)
    channel_att = layers.Dense(x.shape[-1], activation="sigmoid")(channel_att)
    channel_att = layers.Reshape((1, 1, x.shape[-1]))(channel_att)  # 调整形状用于广播
    x = layers.Multiply()([x, channel_att])  # 应用通道注意力

    # 2. 计算空间注意力
    spatial_avg = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)  # 空间平均池化
    spatial_max = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)  # 空间最大池化
    spatial_att = layers.Concatenate(axis=-1)([spatial_avg, spatial_max])
    spatial_att = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(spatial_att)  # 7x7卷积学习空间权重
    x = layers.Multiply()([x, spatial_att])  # 应用空间注意力

    # 过渡层（保留模型2的过渡层思路）
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)

    # 全连接层
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation="softmax")(x)

    # 构建模型
    model_CNN = models.Model(inputs=inputs, outputs=outputs)

    # 编译模型
    model_CNN.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model_CNN


model_list = [build_cnn_model_1, build_cnn_model_2, build_cnn_model_3, build_cnn_model_4, build_cnn_model_5]


# 训练模型
def train_model(model_in: int,
                learning_diminish: float,
                learning_patience: int = 3,
                epo: int = 50,
                b_size: int = 32):
    # 加载数据
    X, y = load_dataset_M1()
    print(f"数据集规模: {X.shape}, 标签数量: {y.shape}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 获取输入形状 (高度, 宽度, 通道数)
    input_shape = X_train[0].shape
    print(input_shape)

    # 构建模型
    # model_r = build_cnn_model_3(input_shape)
    model_r = model_list[model_in](input_shape)
    # 将 summary 转为字符串
    summary_str = StringIO()
    sys.stdout = summary_str  # 重定向标准输出到 StringIO
    model_r.summary()
    sys.stdout = sys.__stdout__  # 恢复标准输出

    # 训练模型
    history_r = model_r.fit(
        X_train,
        y_train,
        epochs=epo,  # 训练轮次
        batch_size=b_size,  # 每一组的样本数 32->64
        validation_split=0.2,
        callbacks=[
            callbacks.EarlyStopping(
                patience=8, restore_best_weights=True
            ),  # 5->8轮没有提升提前终止 返回最优解
            callbacks.ReduceLROnPlateau(
                factor=learning_diminish, patience=learning_patience
            ),  # 指标停滞时 学习率变为原来的一半 连续 3 轮没有提升就调整学习率
        ],
    )

    # 评估模型
    test_loss, test_acc = model_r.evaluate(X_test, y_test)

    # 预测测试集结果（用于计算额外指标）
    y_pred_probs = model_r.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)  # 将概率转换为类别标签

    # 使用sklearn计算精确率、召回率和F1分数（多分类用macro平均）
    # 原来那么写不行
    # 其他人复现如果这里有问题可以关注一下sklearn的版本之类的（12行）
    test_precision = precision_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    test_f1 = f1_score(y_test, y_pred, average='macro')

    # 打印所有指标
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集召回率: {test_recall:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")

    # 保存模型
    model_r.save("vocal_tech_cnn.h5")
    print("模型已保存为 vocal_tech_cnn.h5")

    actual_epochs = len(history_r.epoch)

    # 实验结果简报
    with open(REPORTS_PATH + "训练数据（{0}）.txt".format(time.time()), "w", encoding="utf-8") as f:
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
                time.time() - t1
            )
        )

    return model_r, history_r


if __name__ == "__main__":
    model, history = train_model(4, 0.7, 5)
