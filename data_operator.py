# 文件流
import os
import json
from pydub import AudioSegment
import tempfile

# 程序模块
from typing import List, Dict
from Constant import *

# 音频数据处理
import numpy as np
import librosa
import librosa.display


def GTS_spliter(obj_dir: str, out_dir: str):
    """

    Args:
        obj_dir: 目标的最次文件目录比如说：Chinese/ZH-Alto-1/Breathy/不再见/Breathy_Group
                 这下面就只有json以及谱面数据和wav文件
        out_dir: 目标输出文件夹，一定要填，会生成新的文件夹，所以是根目录

    Returns:

    """
    indexer: int = 0  # 有的字回重复，编码一下保留
    for filename in os.listdir(obj_dir):
        type_name: str = os.path.basename(obj_dir)
        song_name: str = os.path.basename(os.path.dirname(obj_dir))
        # print(song_name)
        if filename.endswith(".json"):
            # print(filename)
            # 获取文件完整路径
            json_path = os.path.join(obj_dir, filename)
            base_name = os.path.splitext(filename)[0]  # 纯名称不带文件格式
            wav_path = os.path.join(obj_dir, f"{base_name}.wav")  # 音频路径

            # 先读json
            with open(json_path, "r", encoding="utf-8") as f:
                data: List = json.load(f)
            time_list: List[List] = []
            name_meta: List = []  # 处理分割后的单个字的命名
            # 数据结构是外列表里套每一个字的dict
            for i in data:
                i: Dict
                start = i.get("ph_start", [])
                start = min(start)  # 辅音发音时间早
                end = i.get("ph_end", [])
                end = max(end)  # 元音晚
                if (
                    i.get("word") == "<AP>" or i.get("word") == "<SP>"
                ):  # 跳过换气/sp不知道是什么
                    continue
                else:
                    time_list.append([start, end])
                    name_meta.append(
                        "{0}_{1}_{2}".format(
                            song_name, i.get("word") + str(indexer), i.get("tech")
                        )
                    )
                    indexer += 1
            # 处理音频
            print(time_list)
            print(name_meta)
            # 加载音频文件
            audio = AudioSegment.from_wav(wav_path)

            # 创建输出目录
            output_dir = os.path.join(
                out_dir, "{0}_{1}_splits".format(song_name, type_name)
            )
            os.makedirs(output_dir, exist_ok=True)

            # 切割并保存每个片段
            for j in range(time_list.__len__()):
                # 转换时间单位（pydub使用毫秒）
                start_ms = time_list[j][0] * 1000
                end_ms = time_list[j][1] * 1000

                # 切割音频
                segment_audio = audio[start_ms:end_ms]

                # 保存片段
                output_path = os.path.join(output_dir, name_meta[j] + ".wav")
                segment_audio.export(output_path, format="wav")
            # return


def process_audio_to_1s(audio_path, target_duration=1000):
    """
    将音频处理为指定时长（默认1秒=1000毫秒）
    超过则截断，不足则补零
    便于CNN处理
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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio.export(temp_file.name, format="wav")
            temp_filename = temp_file.name

        # 使用librosa读取处理后的音频
        y, sr = librosa.load(temp_filename, sr=None)
        os.unlink(temp_filename)
        return y, sr

    except Exception as e:
        print(f"处理音频文件 {audio_path} 时出错: {str(e)}")
        return None, None


def generate_mel_spectrogram(
    y, sr, audio_path, n_fft=2048, hop_length=512, n_mels=128
) -> np.array:
    if y is None or sr is None:
        return

    # 计算梅尔频谱
    # noinspection PyUnresolvedReferences
    mel_spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # 转换为分贝刻度
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect_db


# 处理真假混声数据集
def M1_data_pre_operate(data_path: str, singer: str):
    # # 先找所有混声这一栏目录
    # result = []
    # target_folder = "Mixed_Voice_and_Falsetto"
    # # 使用os.walk遍历，只检查目录名是否完全匹配
    # for dirpath, dirnames, _ in os.walk(data_path):
    #     if target_folder in dirnames:
    #         # 直接拼接路径，避免额外检查
    #         full_path = os.path.join(dirpath, target_folder)
    #         result.append(full_path)
    #         # 如果不需要深入该目标文件夹内部，可以移除它以加速遍历
    #         dirnames.remove(target_folder)
    # print(result)
    target_path = data_path + "\\" + singer + r"\Mixed_Voice_and_Falsetto"
    folder = [
        "Control_Group",
        "Falsetto_Group",
        "Mixed_Voice_Group",
        "Paired_Speech_Group",
    ]
    for i in os.listdir(target_path):
        # 这是歌名
        path1 = target_path + "\\" + i
        for j in folder:
            path2 = path1 + "\\" + j
            print(path2)
            GTS_spliter(path2, USE_PATH)
        # break
    for i in os.listdir(USE_PATH):
        if i + ".npy" in os.listdir(DATASET_PATH):
            continue  # 避免重复操作
        folder_path = USE_PATH + "\\" + i
        datas = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 处理音频为1秒长度
            y, sr = process_audio_to_1s(file_path)
            # 生成梅尔频谱
            mel_gram = generate_mel_spectrogram(y, sr, file_path)
            # print(mel_gram)

            # 标签处理
            # print(filename)
            position = str(filename).rfind("_")  # 找前缀
            tech_tag = str(filename)[position + 1:].replace(".wav", "")
            tech_list: List
            if tech_tag.find(","):
                tech_list = tech_tag.split(",")
            else:
                tech_list.append(tech_tag)
            # print(tech_list)
            if "0" in tech_list:
                tag = 0
            elif "1" in tech_list:
                tag = 1
            elif "2" in tech_list:
                tag = 2
            elif "None" in tech_list:
                tag = -1  # 说话
            # print(tag)

            datas.append([mel_gram, tag])

        np.save(DATASET_PATH + i, np.array(datas, dtype=object))
        print("数据集生成成功：{0}".format(i))


def select_M2_data_pre_operate(path_list: list):  # 指定文件夹列表避免数据太差 会处理气声
    for i in path_list:  # path_list 里的路径应该为次级数据库路径，下函各类特征及对照组
        for j in os.listdir(i):
            path2 = i + "\\" + j
            print(path2)
            GTS_spliter(path2, USE_PATH)
    # target_path = data_path + "\\" + singer + r"\Mixed_Voice_and_Falsetto"
    # folder = [
    #     "Control_Group",
    #     "Falsetto_Group",
    #     "Mixed_Voice_Group",
    #     "Paired_Speech_Group",
    # ]
    # for i in os.listdir(target_path):
    #     # 这是歌名
    #     path1 = target_path + "\\" + i
    #     for j in folder:
    #         path2 = path1 + "\\" + j
    #         print(path2)
    #         GTS_spliter(path2, USE_PATH)
    #     # break
    for i in os.listdir(USE_PATH):
        if i + ".npy" in os.listdir(DATASET_PATH):
            continue  # 避免重复操作
        folder_path = USE_PATH + "\\" + i
        datas = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 处理音频为1秒长度
            y, sr = process_audio_to_1s(file_path)
            # 生成梅尔频谱
            mel_gram = generate_mel_spectrogram(y, sr, file_path)
            # print(mel_gram)

            # 标签处理
            # print(filename)
            position = str(filename).rfind("_")  # 找前缀
            tech_tag = str(filename)[position + 1:].replace(".wav", "")
            print(tech_tag)
            tech_list: List
            if tech_tag.find(","):
                tech_list = tech_tag.split(",")
            else:
                tech_list.append(tech_tag)
            # print(tech_list)
            if "0" in tech_list:
                tag = 0
            elif "1" in tech_list:
                tag = 1
            elif "2" in tech_list:
                tag = 2
            elif "3" in tech_list:
                tag = 3  # 气声

            elif "None" in tech_list:
                tag = -1  # 说话
            else:
                tag = 0  # 没说就是真声
            print(tag)

            datas.append([mel_gram, tag])

        np.save(DATASET_PATH + i, np.array(datas, dtype=object))
        print("数据集生成成功：{0}".format(i))


if __name__ == "__main__":
    select_M2_data_pre_operate([r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Soprano-1\Mixed_Voice_and_Falsetto\cap diamant",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Soprano-1\Mixed_Voice_and_Falsetto\celle de mes vingt ans",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Soprano-1\Mixed_Voice_and_Falsetto\chez moi",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Soprano-1\Mixed_Voice_and_Falsetto\comment deja",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Soprano-1\Mixed_Voice_and_Falsetto\eviter les roses",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Soprano-1\Mixed_Voice_and_Falsetto\je m'appelle helene",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Tenor-1\Mixed_Voice_and_Falsetto\a leve toi",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Tenor-1\Mixed_Voice_and_Falsetto\la fleur",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Tenor-1\Mixed_Voice_and_Falsetto\Le Toréador",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Tenor-1\Mixed_Voice_and_Falsetto\manon",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\French\FR-Tenor-1\Mixed_Voice_and_Falsetto\pourquoi",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Soprano-1\Mixed_Voice_and_Falsetto\Adio Adio",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Soprano-1\Mixed_Voice_and_Falsetto\Copilot",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Soprano-1\Mixed_Voice_and_Falsetto\Danke",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Soprano-1\Mixed_Voice_and_Falsetto\Mitten Im Paradies",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Soprano-1\Mixed_Voice_and_Falsetto\Ich Gehör Nur Mir",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Soprano-1\Mixed_Voice_and_Falsetto\Die Gedanken Sind Frei",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Tenor-1\Mixed_Voice_and_Falsetto\Halt",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Tenor-1\Mixed_Voice_and_Falsetto\Ich liebe dich",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Tenor-1\Mixed_Voice_and_Falsetto\Im Frühling",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Tenor-1\Mixed_Voice_and_Falsetto\Im wunderschönen Monat Mai",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\German\DE-Tenor-1\Mixed_Voice_and_Falsetto\In der Fremde",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Italian\IT-Bass-2\Mixed_Voice_and_Falsetto\Nina",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Italian\IT-Bass-2\Mixed_Voice_and_Falsetto\O cessate di piagarmi",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Italian\IT-Bass-1\Mixed_Voice_and_Falsetto\Tristezza",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Italian\IT-Soprano-1\Mixed_Voice_and_Falsetto\batti batti",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Italian\IT-Soprano-1\Mixed_Voice_and_Falsetto\bella ciao",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Italian\IT-Soprano-1\Mixed_Voice_and_Falsetto\caro mio ben",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Japanese\JA-Soprano-1\Mixed_Voice_and_Falsetto\あの頃～ジンジンバオヂュオニー～",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Japanese\JA-Soprano-1\Mixed_Voice_and_Falsetto\いつも何度でも",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Japanese\JA-Soprano-1\Mixed_Voice_and_Falsetto\さよならの夏～コクリコ坂から",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Japanese\JA-Tenor-1\Mixed_Voice_and_Falsetto\君をのせて",
                                r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Japanese\JA-Tenor-1\Mixed_Voice_and_Falsetto\ドラえもんのうた"])




    # M1_data_pre_operate(DATA_PATH, L_Singer[4])
    # M1_data_pre_operate(DATA_PATH, L_Singer[1])

    # for i in os.listdir(USE_PATH):
    #     # print(DATASET_PATH + i + ".npy")
    #     if i + ".npy" in os.listdir(DATASET_PATH):
    #         print(i)
    # print(os.listdir(DATASET_PATH))
