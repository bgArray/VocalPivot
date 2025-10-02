# 给LSTM写的
# 逻辑：直接遍历所有最次层音频，然后先转频谱，再根据频谱贴标签


# 文件流
import os
import json

# from pydub import AudioSegment
# import tempfile

# 程序模块
from typing import List, Dict  # , Union

from Constant import *

# 音频数据处理
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def generate_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128) -> np.array:
    """
    生成音频的梅尔频谱（Mel Spectrogram）并转换为分贝刻度

    参数说明：
        y: 音频时间序列（ librosa.load 加载的音频数据，一维numpy数组）
        sr: 音频采样率（单位：Hz）
        n_fft: FFT窗口大小，默认2048（决定频率分辨率，值越大频率细节越丰富但计算量越大）
        hop_length: 帧移大小，默认512（控制时间分辨率，值越小时间细节越丰富但数据量越大）
        n_mels: 梅尔滤波器组数量，默认128（决定梅尔频谱的频率轴维度）

    返回值：
        np.array: 转换为分贝刻度的梅尔频谱，形状为 (n_mels, t)，t为时间帧数量
    """
    if y is None or sr is None:
        return

    # 计算梅尔频谱
    # noinspection PyUnresolvedReferences
    mel_spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # 转换为分贝刻度
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max).T  # 倒置矩阵
    return mel_spect_db


# 显示梅尔频谱的函数
def plot_mel_spectrogram(mel_spect_db, sr, hop_length=512):
    """可视化梅尔频谱图"""
    plt.figure(figsize=(10, 4))  # 设置图像大小
    # 显示梅尔频谱（使用 librosa 的显示函数，自动处理频率轴）
    librosa.display.specshow(
        mel_spect_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",  # x轴为时间
        y_axis="mel",  # y轴为梅尔频率
    )
    plt.colorbar(format="%+2.0f dB")  # 颜色条表示分贝值
    plt.title("Mel Spectrogram")  # 标题
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图像


def GTS_LSTM_operator(target_path: str, is_show: bool = False):
    type_name: str = os.path.basename(target_path)
    song_name: str = os.path.basename(os.path.dirname(target_path))
    singer_name: str = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(target_path))))
    print(type_name, song_name)

    # 数据格式：
    # [ [mel_gram([[f1, f2, ..., f128](tick1), ...]), tag_list([[0, 6], [-1], [(-1 -6)], ...])], [第二句], ...]
    return_list: List = []
    for i in os.listdir(target_path):
        if i.endswith(".json"):
            # 获取文件完整路径
            json_path = os.path.join(target_path, i)
            base_name = os.path.splitext(json_path)[0]  # 纯名称不带文件格式
            wav_path = os.path.join(target_path, f"{base_name}.wav")  # 音频路径

            # 先处理音频 跟CNN的数据不一样
            y, sr = librosa.load(wav_path, sr=None)
            mels = generate_mel_spectrogram(y, sr)
            # time_tick = mels[0].__len__()  # mel tick 数量
            absolute_time = 512 / sr  # 一个mel tick 对应的时间
            if is_show:
                plot_mel_spectrogram(mels, sr)

            # 处理json
            with open(json_path, "r", encoding="utf-8") as f:
                js_data: List = json.load(f)

            sentence_start = 0
            tag_list: List = []
            for j in js_data:
                j: Dict
                start = j.get("ph_start", [])
                start = min(start)  # 辅音发音时间早
                end = j.get("ph_end", [])
                end = max(end)  # 元音晚
                tags = j.get("tech", "").split(",")
                # print(tags)
                if j.get("word") == "<AP>":  # 换气
                    tags = ["-1"]
                elif (
                    str(tags).find("1") == -1
                    and str(tags).find("2") == -1
                    and str(tags).find("0") == -1
                    and str(tags).find("-1") == -1
                ):
                    tags.append("0")

                start_tick = round(start / absolute_time)
                # if start_tick != 0:
                #     start_tick += 1
                end_tick = round(end / absolute_time)

                tag_ = [tags] * (end_tick - start_tick)
                tag_list.extend(tag_)
                if j.get("word") == "<AP>":
                    # print(mels[sentence_start : end_tick].__len__())
                    # print(tag_list.__len__())
                    sentence = [mels[sentence_start : end_tick], tag_list[:mels[sentence_start : end_tick].__len__()]]
                    tag_list = []
                    return_list.append(sentence)
                    sentence_start = end_tick + 1

    np.save("{0}{1}_{2}_{3}.npy".format(DATASET_PATH,
                                      singer_name,
                                      song_name,
                                      type_name), np.array(return_list, dtype=object))
