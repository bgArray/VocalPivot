import os
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from keras.models import *
# from keras import Input
# from keras.preprocessing import image
import M4CNN_AM

# --------------------------
# 配置参数（根据实际情况修改）
# --------------------------
MODEL_PATH = "./models/vocal_tech_cnn_M5E3(0.3drop)_85.3.h5"  # 训练好的模型路径
TEST_IMAGE_PATH = "./dataset_select/"  # 测试用的梅尔频谱.npy文件路径
LAYER_NAMES = [
    "conv2d",  # 第一个卷积层
    "conv2d_1",  # 第二个卷积层
    "conv2d_2",  # 第三个卷积层
    "conv2d_3",  # 第四个卷积层
    "multiply",  # 通道注意力输出层
    "multiply_1"  # 空间注意力输出层
]  # 要可视化的层名称（可通过model.summary()查看）
SAVE_DIR = "feature_maps"  # 特征图保存目录


# --------------------------
# 工具函数
# --------------------------
def load_test_data(npy_path):
    """加载测试用的梅尔频谱数据（与训练时格式一致）"""
    mel_gram = np.load(npy_path, allow_pickle=True)
    # 确保数据形状为 (height, width)，并添加批次和通道维度 (1, height, width, 1)
    if len(mel_gram.shape) == 2:
        mel_gram = mel_gram[np.newaxis, ..., np.newaxis]
    return mel_gram


def create_feature_extractor(model, layer_names):
    """创建提取指定层特征的模型"""
    return Model(
        inputs=model.input,
        outputs=[model.get_layer(name).output for name in layer_names]
    )


def visualize_feature_maps(feature_maps, layer_name, save_dir, n_cols=8):
    """可视化并保存指定层的特征图"""
    # 特征图形状：(batch_size, height, width, channels)
    batch_size, height, width, channels = feature_maps.shape
    n_rows = int(np.ceil(channels / n_cols))  # 计算行数

    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(min(channels, 32)):  # 最多显示32个通道（避免图过多）
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap="viridis")  # 取第一个样本
        plt.axis("off")
        plt.title(f"Ch {i + 1}", fontsize=8)

    plt.suptitle(f"Feature Maps - {layer_name}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 预留标题空间

    # 保存图像
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{layer_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存 {layer_name} 特征图至 {save_path}")


# --------------------------
# 主函数
# --------------------------
def main():
    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在")
    # # 根据你的数据实际形状修改（例如 (height, width, channels)）
    # # 错误提示中 Lambda 层输入是 (None, 6, 4, 256)，推测原始输入可能更小，可先尝试如下形状
    # input_shape = (6, 4, 1)  # 替换为你的梅尔频谱实际形状（不含 batch 维度）
    #
    # # 创建一个输入张量，帮助模型推断形状
    # dummy_input = Input(shape=input_shape)
    #
    # # 加载模型时传入 safe_mode=False，并通过 build 方法指定输入形状
    # model = load_model(MODEL_PATH, safe_mode=False)
    # model.build((None,) + input_shape)  # 显式指定输入形状（None 表示 batch 维度可变）
    # # model = load_model(MODEL_PATH, safe_mode=False)
    # model.summary()  # 查看模型结构，确认层名称是否正确
    model = M4CNN_AM.build_cnn_model_5((128, 94, 1))

    # 2. 加载并预处理测试数据
    # if not os.path.exists(TEST_IMAGE_PATH):
    #     raise FileNotFoundError(f"测试文件 {TEST_IMAGE_PATH} 不存在")
    # test_data = load_test_data(TEST_IMAGE_PATH)
    # print(f"测试数据形状: {test_data.shape}")
    # 2. 加载权重（而非直接加载整个模型）
    model.load_weights(MODEL_PATH)  # 只加载权重，不加载模型结构序列化信息

    # 3. 后续步骤不变（创建特征提取器、提取特征图）
    # feature_extractor = create_feature_extractor(model, LAYER_NAMES)
    # extracted_features = feature_extractor.predict(test_data)
    test_data, _ = M4CNN_AM.load_dataset_M1(TEST_IMAGE_PATH, count=1000)

    # 3. 创建特征提取器
    feature_extractor = create_feature_extractor(model, LAYER_NAMES)

    # 4. 提取特征图
    extracted_features = feature_extractor.predict(test_data)

    # 5. 可视化每个层的特征图
    for layer_name, feature_map in zip(LAYER_NAMES, extracted_features):
        visualize_feature_maps(feature_map, layer_name, SAVE_DIR)

    print("所有特征图可视化完成！")


if __name__ == "__main__":
    main()
