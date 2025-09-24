import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras import *
from keras import backend as K

import M4CNN_AM

# --------------------------
# 配置参数
# --------------------------
MODEL_PATH = "./models/vocal_tech_cnn_M5E4(0.1-0.3drop_nospeaking)_82.7.h5"  # 模型路径
TARGET_LABELS = [0, 1, 2]  # 要反向生成的目标标签
INPUT_SHAPE = (128, 94, 1)  # 模型输入形状 (height, width, channels)
SAVE_DIR = "inverted_images"  # 反向生成图像的保存目录
ITERATIONS = 200  # 优化迭代次数（越多越接近目标）
LEARNING_RATE = 0.01  # 优化学习率


# --------------------------
# 工具函数
# --------------------------
def invert_label(model, target_label, input_shape, iterations=200, lr=0.01):
    """
    从目标标签反向生成输入图像
    :param model: 训练好的模型
    :param target_label: 目标标签（如0、1、2、3）
    :param input_shape: 输入图像形状 (h, w, c)
    :param iterations: 优化迭代次数
    :param lr: 学习率
    :return: 反向生成的图像（归一化后）
    """
    # 初始化随机噪声作为起点（接近输入数据分布，这里假设输入在0~1范围）
    generated = Variable(np.random.uniform(
        low=0.0,
        high=1.0,
        size=(1,) + input_shape  # 形状：(1, height, width, channels)
    ).astype(np.float32))

    # 定义优化器
    optimizer = optimizers.Adam(learning_rate=lr)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            # 计算模型对生成图像的预测
            predictions = model(generated)
            # 目标：让模型对目标标签的预测概率最大化
            loss = -tf.reduce_mean(predictions[:, target_label])  # 负号表示最大化该标签概率

        # 计算梯度并更新生成图像
        grads = tape.gradient(loss, generated)
        optimizer.apply_gradients([(grads, generated)])

        # 约束生成图像在合理范围（根据你的输入数据分布调整，如0~1或-1~1）
        generated.assign(tf.clip_by_value(generated, 0.0, 1.0))

        # 打印进度
        if (i + 1) % 50 == 0:
            print(f"标签 {target_label} 迭代 {i+1}/{iterations}，损失: {loss.numpy():.4f}")

    # 转换为numpy数组并去除batch维度
    generated_img = generated.numpy()[0]
    return generated_img


def save_inverted_image(img, label, save_dir):
    """保存反向生成的图像"""
    os.makedirs(save_dir, exist_ok=True)
    # 处理通道维度（如果是单通道则转为灰度图）
    if img.shape[-1] == 1:
        img = img.squeeze(axis=-1)  # 去除通道维度

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="viridis")
    plt.title(f"Label {label} (Inverted)")
    plt.axis("off")
    save_path = os.path.join(save_dir, f"label_{label}_inverted.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存标签 {label} 的反向生成图像至 {save_path}")


# --------------------------
# 主函数
# --------------------------
def main():
    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    model = load_model(MODEL_PATH, safe_mode=False, custom_objects={"attention_loss": M4CNN_AM.reduce_mean_layer},)
    model.summary()

    # 2. 为每个目标标签反向生成图像
    for label in TARGET_LABELS:
        print(f"\n开始反向生成标签 {label} 的图像...")
        inverted_img = invert_label(
            model=model,
            target_label=label,
            input_shape=INPUT_SHAPE,
            iterations=ITERATIONS,
            lr=LEARNING_RATE
        )
        save_inverted_image(inverted_img, label, SAVE_DIR)

    print("\n所有标签的反向生成图像已保存！")


if __name__ == "__main__":
    main()