import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                               QVBoxLayout, QHBoxLayout, QFileDialog, QWidget,
                               QFrame, QMessageBox)
from PySide6.QtCore import Qt, Signal, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import soundfile as sf

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False


class AudioProcessor(QThread):
    """音频处理线程，避免UI卡顿"""
    # 修正信号参数数量，与实际发送的一致
    processing_complete = Signal(object, object, object, object, object, float)
    processing_error = Signal(str)

    def __init__(self, audio_path, model, le, max_length, target_dim):
        super().__init__()
        self.audio_path = audio_path
        self.model = model
        self.le = le
        self.max_length = max_length
        self.target_dim = target_dim

    def run(self):
        try:
            # 加载音频
            y, sr = librosa.load(self.audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            # 提取梅尔频谱（与训练时参数一致）
            n_fft = 2048
            hop_length = 512
            n_mels = 128  # 原始梅尔维度，后续会切片到74维

            mel_spectrogram = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )

            # 转换为分贝值
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            mel_sequence = mel_spectrogram_db.T  # 转置为 (时间步, 频域维度)

            # 频域切片到74维（与训练时一致）
            if mel_sequence.shape[1] >= self.target_dim:
                mel_sequence = mel_sequence[:, 0:self.target_dim]
            else:
                pad_width = self.target_dim - mel_sequence.shape[1]
                mel_sequence = np.pad(
                    mel_sequence,
                    pad_width=((0, 0), (0, pad_width)),
                    mode='constant',
                    constant_values=0.0
                )

            # 处理长音频：分块预测，确保覆盖全部音频
            total_frames = mel_sequence.shape[0]
            predicted_classes = []

            # 按模型最大长度分块处理
            for i in range(0, total_frames, self.max_length):
                end_idx = min(i + self.max_length, total_frames)
                chunk = mel_sequence[i:end_idx]

                # 对最后一块进行填充
                if len(chunk) < self.max_length:
                    pad_length = self.max_length - len(chunk)
                    chunk = np.pad(
                        chunk,
                        pad_width=((0, pad_length), (0, 0)),
                        mode='constant',
                        constant_values=0.0
                    )

                # 模型预测
                from keras.preprocessing import sequence
                X_input = sequence.pad_sequences(
                    [chunk],
                    maxlen=self.max_length,
                    dtype='float32',
                    padding='post',
                    truncating='post',
                    value=0.0
                )

                predictions = self.model.predict(X_input, verbose=0)[0]
                chunk_preds = np.argmax(predictions, axis=1)

                # 只保留实际有效部分的预测结果（去除填充部分）
                predicted_classes.extend(chunk_preds[:end_idx - i])

            predicted_classes = np.array(predicted_classes)

            # 生成时间轴（与实际帧数量匹配）
            frame_duration = hop_length / sr
            time_stamps = np.arange(len(predicted_classes)) * frame_duration
            time_stamps = np.minimum(time_stamps, duration)

            self.processing_complete.emit(
                y, sr, mel_spectrogram_db, predicted_classes,
                time_stamps, duration
            )
        except Exception as e:
            self.processing_error.emit(str(e))


class AudioClassifierVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.le = None
        self.max_length = None  # 模型输入序列长度
        self.target_dim = 74  # 与训练时一致的频域维度
        self.no_label_idx = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("音频逐帧分类可视化工具")
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 控制区
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)

        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model_files)
        control_layout.addWidget(self.load_model_btn)

        self.select_audio_btn = QPushButton("选择音频文件")
        self.select_audio_btn.clicked.connect(self.select_audio_file)
        self.select_audio_btn.setEnabled(False)
        control_layout.addWidget(self.select_audio_btn)

        self.status_label = QLabel("请先加载模型文件")
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        main_layout.addWidget(control_frame)

        # 音频信息区
        info_frame = QFrame()
        info_layout = QHBoxLayout(info_frame)
        self.audio_info_label = QLabel("未选择音频文件")
        info_layout.addWidget(self.audio_info_label)
        main_layout.addWidget(info_frame)

        # 可视化区域
        viz_frame = QFrame()
        viz_layout = QVBoxLayout(viz_frame)

        # 创建matplotlib图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                                      gridspec_kw={'height_ratios': [2, 1]})
        self.fig.tight_layout(pad=3)
        self.canvas = FigureCanvas(self.fig)
        viz_layout.addWidget(self.canvas)

        main_layout.addWidget(viz_frame, 1)

    def load_model_files(self):
        """加载模型和标签编码器"""
        # 选择模型文件
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "Keras模型 (*.keras);;所有文件 (*)"
        )
        if not model_path:
            return

        # 选择标签编码器文件
        le_path, _ = QFileDialog.getOpenFileName(
            self, "选择标签编码器文件", "", "Numpy文件 (*.npy);;所有文件 (*)"
        )
        if not le_path:
            return

        try:
            # 加载模型
            self.model = load_model(
                model_path,
                custom_objects={"loss": self.get_custom_loss()}
            )

            # 获取模型输入序列长度
            self.max_length = self.model.input_shape[1]

            # 加载标签编码器
            classes = np.load(le_path)
            self.le = LabelEncoder()
            self.le.classes_ = classes

            # 获取无标签索引
            self.no_label_idx = np.where(classes == "-1")[0][0]

            self.status_label.setText(f"模型加载成功 - 类别: {', '.join(classes)}")
            self.select_audio_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"模型加载错误: {str(e)}")
            self.status_label.setText("模型加载失败")

    def get_custom_loss(self):
        """创建与训练时一致的自定义损失函数"""
        from tensorflow import reduce_sum, maximum
        from keras import losses

        def ignore_no_label_loss(y_true, y_pred):
            mask = 1 - y_true[:, :, self.no_label_idx]
            ce_loss = losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
            masked_loss = ce_loss * mask
            valid_frame_count = maximum(reduce_sum(mask), 1e-8)
            return reduce_sum(masked_loss) / valid_frame_count

        return ignore_no_label_loss

    def select_audio_file(self):
        """选择并处理音频文件"""
        audio_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg);;所有文件 (*)"
        )

        if not audio_path:
            return

        self.status_label.setText("正在处理音频...")
        self.load_model_btn.setEnabled(False)
        self.select_audio_btn.setEnabled(False)

        # 创建并启动处理线程
        self.processor = AudioProcessor(
            audio_path,
            self.model,
            self.le,
            self.max_length,
            self.target_dim
        )
        self.processor.processing_complete.connect(self.on_processing_complete)
        self.processor.processing_error.connect(self.on_processing_error)
        self.processor.start()

        self.audio_info_label.setText(f"音频文件: {os.path.basename(audio_path)}")

    def on_processing_complete(self, y, sr, mel_spectrogram_db, predicted_classes, time_stamps, duration):
        """处理完成后更新可视化"""
        self.visualize_results(y, sr, mel_spectrogram_db, predicted_classes, time_stamps, duration)
        self.status_label.setText("音频处理完成，已生成可视化结果")
        self.load_model_btn.setEnabled(True)
        self.select_audio_btn.setEnabled(True)

    def on_processing_error(self, error_msg):
        """处理错误时显示消息"""
        QMessageBox.critical(self, "处理错误", f"音频处理失败: {error_msg}")
        self.status_label.setText("音频处理失败")
        self.load_model_btn.setEnabled(True)
        self.select_audio_btn.setEnabled(True)

    def visualize_results(self, y, sr, mel_spectrogram_db, predicted_classes, time_stamps, duration):
        """可视化梅尔频谱和分类结果"""
        # 清除之前的绘图
        self.ax1.clear()
        self.ax2.clear()

        # 绘制梅尔频谱
        librosa.display.specshow(
            mel_spectrogram_db,
            sr=sr,
            hop_length=512,
            x_axis='time',
            y_axis='mel',
            ax=self.ax1
        )
        self.ax1.set_title('梅尔频谱图')
        self.ax1.set_xlabel('时间 (秒)')
        self.ax1.set_ylabel('梅尔频率')
        self.fig.colorbar(self.ax1.collections[0], ax=self.ax1, format="%+2.f dB")

        # 绘制逐帧分类结果
        class_names = self.le.classes_
        unique_classes = np.unique(predicted_classes)

        # 创建颜色映射
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        color_map = {i: colors[i] for i in range(len(class_names))}

        # 绘制分类结果
        for cls in unique_classes:
            if class_names[cls] == "-1":  # 跳过无标签类别
                continue

            mask = (predicted_classes == cls)
            self.ax2.scatter(
                time_stamps[mask],
                np.full(sum(mask), cls),
                label=class_names[cls],
                color=color_map[cls],
                s=2,
                alpha=0.8
            )

        # 确保只显示一个图例且位置固定
        handles, labels = self.ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # 去重
        self.ax2.legend(by_label.values(), by_label.keys(), loc='upper right', markerscale=5)

        self.ax2.set_title('逐帧分类结果')
        self.ax2.set_xlabel('时间 (秒)')
        self.ax2.set_ylabel('类别')
        self.ax2.set_yticks(range(len(class_names)))
        self.ax2.set_yticklabels(class_names)
        self.ax2.set_xlim(0, duration)

        # 调整布局并刷新画布
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = AudioClassifierVisualizer()
    window.show()
    sys.exit(app.exec())