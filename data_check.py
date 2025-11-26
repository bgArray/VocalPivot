import numpy as np
import matplotlib.pyplot as plt


def inspect_npy_file(file_path, sample_index=0):
    """
    检查NPY文件中的数据有效性
    :param file_path: NPY文件路径
    :param sample_index: 要查看的样本索引（默认第一个）
    """
    try:
        # 加载NPY文件
        data = np.load(file_path, allow_pickle=True)
        print(f"成功加载文件: {file_path}")
        print(f"文件中包含 {len(data)} 个样本\n")

        # 检查样本索引是否有效
        if sample_index < 0 or sample_index >= len(data):
            print(f"无效的样本索引，可选范围: 0 到 {len(data) - 1}")
            return

        # 获取指定样本
        sample = data[sample_index]
        if len(sample) != 2:
            print("样本格式错误，应为 (mel_spectrogram, tags) 元组")
            return

        mel_spectrogram, tags = sample
        print(f"第 {sample_index} 个样本信息:")
        print(f"梅尔频谱形状: {mel_spectrogram.shape} (时间步 × 特征数)")
        print(f"标签序列长度: {len(tags)}")

        # 检查梅尔频谱与标签长度是否一致
        if len(mel_spectrogram) != len(tags):
            print(f"⚠️ 警告: 梅尔频谱长度 ({len(mel_spectrogram)}) 与标签长度 ({len(tags)}) 不匹配")
        else:
            print("梅尔频谱与标签长度匹配")

        # 打印前10个标签（避免过长）
        # print("\n前10个标签 (完整标签共{}个):".format(len(tags)))
        # for i, tag_group in enumerate(tags[:10]):
        #     print(f"时间步 {i}: {tag_group}")
        print(tags)

        # 可视化梅尔频谱（平铺显示）
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_spectrogram.T, aspect='auto', origin='lower')
        plt.title(f'第 {sample_index} 个样本的梅尔频谱')
        plt.xlabel('时间步')
        plt.ylabel('梅尔特征维度')
        plt.colorbar(label='幅度')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")


if __name__ == "__main__":
    # 替换为你的NPY文件路径
    npy_file_path = "dataset/ZH-Alto-1_一次就好_Falsetto_Group.npy"  # 例如: "dataset/sample_001.npy"
    inspect_npy_file(npy_file_path, sample_index=0)  # 查看第一个样本