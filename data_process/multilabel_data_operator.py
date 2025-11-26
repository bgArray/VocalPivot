# 多标签数据处理模块
# 用于生成多标签识别模型（M7MultiLabel_CNN_LSTM_pytorch.py）的训练数据
# 数据格式：每个样本为 [mel_spectrogram, tags]
#   - mel_spectrogram: (time_steps, 128) numpy数组
#   - tags: 长度为time_steps的列表，每个元素是一个标签列表（可能包含多个标签）

import os
import json
from typing import List, Dict, Tuple
from Constant import *

# 音频数据处理
import numpy as np
import librosa


def generate_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128) -> np.array:
    """
    生成音频的梅尔频谱（Mel Spectrogram）并转换为分贝刻度

    参数说明：
        y: 音频时间序列（librosa.load加载的音频数据，一维numpy数组）
        sr: 音频采样率（单位：Hz）
        n_fft: FFT窗口大小，默认2048
        hop_length: 帧移大小，默认512
        n_mels: 梅尔滤波器组数量，默认128

    返回值：
        np.array: 转换为分贝刻度的梅尔频谱，形状为 (time_steps, n_mels)
    """
    if y is None or sr is None:
        return None

    # 计算梅尔频谱
    # noinspection PyUnresolvedReferences
    mel_spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # 转换为分贝刻度并转置，使形状为 (time_steps, n_mels)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max).T
    return mel_spect_db


def parse_tech_labels(tech_str: str) -> List[str]:
    """
    解析技术标签字符串，支持多种格式

    参数：
        tech_str: 技术标签字符串，可能是 "0,1" 或 "Breathy,Vibrato" 等格式

    返回：
        标签列表，如 ["0", "1"] 或 ["Breathy", "Vibrato"]
    """
    if not tech_str or tech_str.strip() == "":
        return []

    # 按逗号分割并去除空白
    labels = [label.strip() for label in tech_str.split(",") if label.strip()]
    return labels


def process_single_file(
    json_path: str, wav_path: str, split_by_breath: bool = True
) -> List[Tuple[np.ndarray, List[List[str]]]]:
    """
    处理单个JSON和WAV文件，生成多标签数据

    参数：
        json_path: JSON文件路径
        wav_path: WAV音频文件路径
        split_by_breath: 是否按换气点（<AP>）分割成多个句子

    返回：
        数据列表，每个元素为 (mel_spectrogram, tags_list)
        - mel_spectrogram: (time_steps, 128) numpy数组
        - tags_list: 长度为time_steps的列表，每个元素是一个标签列表
    """
    if not os.path.exists(wav_path):
        print(f"警告：音频文件 {wav_path} 不存在，跳过")
        return []

    # 加载音频并生成梅尔频谱
    try:
        y, sr = librosa.load(wav_path, sr=None)
        mels = generate_mel_spectrogram(y, sr)
        if mels is None:
            return []
    except Exception as e:
        print(f"警告：处理音频 {wav_path} 时出错: {e}")
        return []

    # 计算每个mel tick对应的时间（秒）
    hop_length = 512
    absolute_time = hop_length / sr  # 一个mel tick对应的时间

    # 读取JSON数据
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            js_data: List[Dict] = json.load(f)
    except Exception as e:
        print(f"警告：读取JSON文件 {json_path} 时出错: {e}")
        return []

    # 处理标签，将时间对齐到mel tick
    # 初始化标签数组：为每个mel tick分配标签列表
    num_ticks = len(mels)
    tick_tags = [[] for _ in range(num_ticks)]  # 每个tick对应一个标签列表

    # 第一遍：为每个音素分配标签
    for item in js_data:
        ph_start = item.get("ph_start", [])
        ph_end = item.get("ph_end", [])

        if not ph_start or not ph_end:
            continue

        start_time = min(ph_start)
        end_time = max(ph_end)

        # 转换为mel tick索引
        start_tick = max(0, min(round(start_time / absolute_time), num_ticks - 1))
        end_tick = max(start_tick + 1, min(round(end_time / absolute_time), num_ticks))

        # 获取技术标签
        word = item.get("word", "")
        tech_str = item.get("tech", "")

        # 处理标签
        if word == "<AP>":  # 换气标记
            tags = ["-1"]
        else:
            tags = parse_tech_labels(tech_str)
            if not tags:
                # 如果没有标签，默认标记为真声（0）
                tags = ["0"]

        # 为当前时间段的所有mel tick分配标签
        for tick_idx in range(start_tick, end_tick):
            if tick_idx < num_ticks:
                # 合并标签（去重）
                existing_tags = set(tick_tags[tick_idx])
                new_tags = set(tags)
                tick_tags[tick_idx] = list(existing_tags | new_tags)

    # 第二遍：按换气点分割句子
    all_samples = []
    if split_by_breath:
        # 找到所有换气点（标签包含"-1"且只有"-1"的位置）
        breath_points = [0]  # 起始点
        for i, tags in enumerate(tick_tags):
            # 检查是否只有"-1"标签
            if len(tags) == 1 and tags[0] == "-1":
                breath_points.append(i)
        breath_points.append(num_ticks)  # 结束点

        # 去重并排序
        breath_points = sorted(list(set(breath_points)))

        # 按换气点分割
        for i in range(len(breath_points) - 1):
            start_idx = breath_points[i]
            end_idx = breath_points[i + 1]

            # 跳过太短的片段
            if end_idx - start_idx < 5:
                continue

            # 提取mel和标签
            sample_mels = mels[start_idx:end_idx]
            sample_tags = tick_tags[start_idx:end_idx]

            # 确保长度一致
            if len(sample_mels) == len(sample_tags):
                all_samples.append((sample_mels, sample_tags))
    else:
        # 不分割，返回整个音频作为一个样本
        all_samples.append((mels, tick_tags))

    return all_samples


def process_multilabel_dataset(
    data_path: str,
    output_path: str = None,
    split_by_breath: bool = True,
    target_folders: List[str] = None,
):
    """
    批量处理GTSinger格式的数据集，生成多标签训练数据

    参数：
        data_path: 数据集根目录（如 E:/文档_学习_高中/活动杂项/AiVocal/dataset）
        output_path: 输出npy文件的目录（默认使用DATASET_PATH）
        split_by_breath: 是否按换气点分割
        target_folders: 要处理的文件夹列表，如 ["Mixed_Voice_and_Falsetto", "Breathy"]
                        如果为None，则处理所有包含JSON和WAV的文件夹
    """
    if output_path is None:
        output_path = DATASET_PATH

    os.makedirs(output_path, exist_ok=True)

    # 如果指定了目标文件夹，只处理这些文件夹
    if target_folders is None:
        target_folders = []

    all_data = []
    processed_count = 0
    skipped_count = 0

    # 遍历数据集目录
    for root, dirs, files in os.walk(data_path):
        # 检查当前目录是否包含JSON和WAV文件
        json_files = [f for f in files if f.endswith(".json")]
        wav_files = [f for f in files if f.endswith(".wav")]

        if not json_files or not wav_files:
            continue

        # 如果指定了目标文件夹，检查当前路径是否包含这些文件夹
        if target_folders:
            path_parts = root.replace("\\", "/").split("/")
            if not any(folder in path_parts for folder in target_folders):
                continue

        # 处理每个JSON文件
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            base_name = os.path.splitext(json_file)[0]
            wav_path = os.path.join(root, f"{base_name}.wav")

            if not os.path.exists(wav_path):
                skipped_count += 1
                continue

            # 处理文件
            samples = process_single_file(json_path, wav_path, split_by_breath)

            if samples:
                all_data.extend(samples)
                processed_count += len(samples)
                print(f"处理完成: {json_path} -> {len(samples)} 个样本")
            else:
                skipped_count += 1

    # 保存数据
    if all_data:
        # 生成输出文件名
        output_filename = "multilabel_dataset.npy"
        if target_folders:
            output_filename = f"multilabel_dataset_{'_'.join(target_folders)}.npy"

        output_filepath = os.path.join(output_path, output_filename)

        # 保存为npy文件
        np.save(output_filepath, np.array(all_data, dtype=object))
        print(f"\n数据保存完成: {output_filepath}")
        print(f"总样本数: {len(all_data)}")
        print(f"成功处理: {processed_count} 个样本")
        print(f"跳过: {skipped_count} 个文件")

        # 打印一些统计信息
        print("\n标签统计:")
        all_labels = []
        for _, tags_list in all_data:
            for tags in tags_list:
                all_labels.extend(tags)

        from collections import Counter

        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common():
            print(f"  {label}: {count} 次")
    else:
        print("警告：没有生成任何数据！")


def process_singer_data(
    data_path: str,
    singer_name: str,
    output_path: str = None,
    split_by_breath: bool = True,
    target_folders: List[str] = None,
):
    """
    处理特定歌手的数据

    参数：
        data_path: 数据集根目录
        singer_name: 歌手文件夹名称，如 "Chinese/ZH-Alto-1"
        output_path: 输出目录
        split_by_breath: 是否按换气点分割
        target_folders: 目标文件夹列表
    """
    singer_path = os.path.join(data_path, singer_name)

    if not os.path.exists(singer_path):
        print(f"错误：歌手路径 {singer_path} 不存在")
        return

    if target_folders is None:
        target_folders = [
            "Breathy",
            "Glissando",
            "Mixed_Voice_and_Falsetto",
            "Pharyngeal",
            "Vibrato",
        ]

    all_data = []

    # 遍历目标文件夹
    for folder in target_folders:
        folder_path = os.path.join(singer_path, folder)
        if not os.path.exists(folder_path):
            continue

        # 遍历歌曲文件夹
        for song_name in os.listdir(folder_path):
            song_path = os.path.join(folder_path, song_name)
            if not os.path.isdir(song_path):
                continue

            # 遍历技术组文件夹（如 Control_Group, Falsetto_Group 等）
            for group_name in os.listdir(song_path):
                group_path = os.path.join(song_path, group_name)
                if not os.path.isdir(group_path):
                    continue

                # 处理该文件夹下的所有JSON和WAV文件
                for filename in os.listdir(group_path):
                    if filename.endswith(".json"):
                        json_path = os.path.join(group_path, filename)
                        base_name = os.path.splitext(filename)[0]
                        wav_path = os.path.join(group_path, f"{base_name}.wav")

                        if os.path.exists(wav_path):
                            samples = process_single_file(
                                json_path, wav_path, split_by_breath
                            )
                            if samples:
                                all_data.extend(samples)
                                print(
                                    f"处理: {singer_name}/{folder}/{song_name}/{group_name}/{filename} -> "
                                    f"{len(samples)} 个样本"
                                )

    # 保存数据
    if all_data:
        if output_path is None:
            output_path = DATASET_PATH

        os.makedirs(output_path, exist_ok=True)

        # 生成输出文件名
        singer_short_name = singer_name.replace("/", "_").replace("\\", "_")
        output_filename = f"multilabel_{singer_short_name}.npy"
        output_filepath = os.path.join(output_path, output_filename)

        np.save(output_filepath, np.array(all_data, dtype=object))
        print(f"\n数据保存完成: {output_filepath}")
        print(f"总样本数: {len(all_data)}")

        # 打印标签统计
        print("\n标签统计:")
        all_labels = []
        for _, tags_list in all_data:
            for tags in tags_list:
                all_labels.extend(tags)

        from collections import Counter

        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common():
            print(f"  {label}: {count} 次")
    else:
        print("警告：没有生成任何数据！")


if __name__ == "__main__":
    # 默认仅处理中文两位歌手（ZH-Alto-1、ZH-Tenor-1），并遍历 5 个技术文件夹
    chinese_singers = ["Chinese/ZH-Alto-1", "Chinese/ZH-Tenor-1"]
    target_folders = [
        "Breathy",
        "Glissando",
        "Mixed_Voice_and_Falsetto",
        "Pharyngeal",
        "Vibrato",
    ]

    for singer in chinese_singers:
        print(f"\n{'=' * 60}")
        print(f"处理歌手: {singer}")
        print(f"{'=' * 60}")
        process_singer_data(
            data_path=DATA_PATH,
            singer_name=singer,
            split_by_breath=True,
            target_folders=target_folders,
        )
