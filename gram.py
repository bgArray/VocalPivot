import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pydub import AudioSegment
import tempfile

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def process_audio_to_1s(audio_path, target_duration=1000):
    """
    将音频处理为指定时长（默认1秒=1000毫秒）
    超过则截断，不足则补零
    """
    try:
        # 使用pydub读取音频
        audio = AudioSegment.from_file(audio_path)

        # 处理音频长度
        if len(audio) > target_duration:
            # 超过1秒则截断
            audio = audio[:target_duration]
        elif len(audio) < target_duration:
            # 不足1秒则补静音
            silence = AudioSegment.silent(duration=target_duration - len(audio))
            audio = audio + silence

        # 保存为临时wav文件，便于librosa处理
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio.export(temp_file.name, format='wav')
            temp_filename = temp_file.name

        # 使用librosa读取处理后的音频
        y, sr = librosa.load(temp_filename, sr=None)

        # 删除临时文件
        os.unlink(temp_filename)

        return y, sr

    except Exception as e:
        print(f"处理音频文件 {audio_path} 时出错: {str(e)}")
        return None, None


def generate_mel_spectrogram(y, sr, audio_path, n_fft=2048, hop_length=512, n_mels=128):
    """生成并显示梅尔频谱"""
    if y is None or sr is None:
        return

    # 计算梅尔频谱
    # noinspection PyUnresolvedReferences
    mel_spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # 转换为分贝刻度
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    print(mel_spect_db)

    # 显示梅尔频谱
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spect_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'梅尔频谱: {os.path.basename(audio_path)}')
    plt.tight_layout()
    plt.show()


def process_audio_folder(folder_path):
    """遍历文件夹处理所有音频文件"""
    # 支持的音频文件格式
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

    # 遍历文件夹
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查是否为音频文件
        if os.path.isfile(file_path) and filename.lower().endswith(audio_extensions):
            print(f"处理文件: {filename}")

            # 处理音频为1秒长度
            y, sr = process_audio_to_1s(file_path)

            # 生成并显示梅尔频谱
            generate_mel_spectrogram(y, sr, file_path)


if __name__ == "__main__":
    # 替换为你的音频文件夹路径
    audio_folder = r"./data/一次就好_Control_Group_splits"

    if not os.path.isdir(audio_folder):
        print(f"错误: 文件夹 '{audio_folder}' 不存在")
    else:
        process_audio_folder(audio_folder)
