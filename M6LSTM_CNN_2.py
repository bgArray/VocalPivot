import os
import numpy as np
import numpy.typing as npt  # 导入NumPy类型注解模块
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Conv1D, BatchNormalization  # , MaxPooling1D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers, losses
from tensorflow import reduce_sum, maximum


# -------------------------- 1. 数据加载函数（保留核心逻辑，优化日志输出） --------------------------
def load_data(file_path):
    """加载.npy数据，提取0/1/2标签并确保特征与标签长度一致"""
    data = []
    # 统计有效/无效样本数，便于后续分析
    valid_count = 0
    invalid_count = 0

    for filename in os.listdir(file_path):
        if filename.endswith('.npy'):
            file_full_path = os.path.join(file_path, filename)
            try:
                # 捕获异常，避免单个损坏文件导致程序崩溃
                batch_data = np.load(file_full_path, allow_pickle=True)
                data.extend(batch_data)
                valid_count += len(batch_data)
            except Exception as e:
                print(f"警告：文件{filename}加载失败，错误信息：{str(e)}")
                invalid_count += 1

    X = []  # 存储mel特征序列
    y = []  # 存储标签序列
    mismatch_count = 0  # 统计长度不匹配样本

    for item in data:
        mel_spectrogram, tags = item
        # 确保mel和标签时间步一致
        if len(mel_spectrogram) != len(tags):
            mismatch_count += 1
            continue
        # 过滤无效标签，仅保留0/1/2，无有效标签标记为-1
        valid_tags = []
        for tag_group in tags:
            filtered = [t for t in tag_group if t in {"0", "1", "2"}]
            valid_tags.append(filtered[0] if filtered else "-1")
        X.append(mel_spectrogram)
        y.append(valid_tags)

    # 输出数据加载统计信息
    print(f"数据加载完成：")
    print(f"- 成功加载文件数：{len(os.listdir(file_path)) - invalid_count}")
    print(f"- 总样本数（含无效）：{valid_count}")
    print(f"- 长度不匹配样本数：{mismatch_count}")
    print(f"- 最终有效样本数：{len(X)}")
    return X, y


# -------------------------- 2. 数据增强函数（仅保留轻量级频谱增强，移除时序裁剪） --------------------------
def augment_mel_spectrogram(mel_seq):
    """轻量级频谱增强：仅保留频域偏移+幅度缩放，降低噪声干扰"""
    aug_seq = mel_seq.copy()
    # 固定随机种子确保增强稳定性（避免每次运行差异过大）
    np.random.seed(42)

    # 1. 微小频域偏移（±1个梅尔系数，减少关键频带破坏）
    shift = np.random.randint(-1, 2)  # 偏移范围从±2→±1
    if shift != 0:
        aug_seq = np.roll(aug_seq, shift, axis=1)
        # 空白处填充原序列边缘均值，避免硬填充0导致特征突变
        edge_mean = aug_seq[:, shift:shift + 1].mean() if shift > 0 else aug_seq[:, shift - 1:shift].mean()
        if shift > 0:
            aug_seq[:, :shift] = edge_mean
        else:
            aug_seq[:, shift:] = edge_mean

    # 2. 温和幅度缩放（0.9-1.1倍，减少力度变化干扰）
    scale = np.random.uniform(0.9, 1.1)  # 缩放范围从0.8-1.2→0.9-1.1
    aug_seq = aug_seq * scale

    return aug_seq


# -------------------------- 3. 数据预处理函数（削减增强比例，优化数据分布） --------------------------
def preprocess_data(
    X: list[npt.NDArray[np.float32]],  # 明确X是float32数组的列表
    y: list[list[str]],  # 明确y是字符串标签列表的列表
    le: LabelEncoder | None = None,
    max_length: int | None = None,
    is_train: bool = True
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], int, LabelEncoder]:
    # 函数逻辑不变...
    """
    优化后预处理：轻量级增强+序列填充+标签编码
    关键修复：numpy.pad()参数格式与类型错误
    """
    # 1. 梅尔频谱频域切片：保留200Hz-5kHz关键频带（原128维→74维）
    X_sliced = []
    target_dim = 74  # 目标频域维度，明确变量避免魔法数字
    for mel_seq in X:
        # 确保mel_seq是numpy数组（避免输入类型异常）
        mel_seq = np.asarray(mel_seq, dtype=np.float32)
        current_dim = mel_seq.shape[1]  # 当前频域维度

        if current_dim >= target_dim:
            # 维度足够：直接切片取前74维
            sliced = mel_seq[:, 0:target_dim]
        else:
            # 维度不足：计算需要填充的长度（确保是整数）
            pad_width = target_dim - current_dim
            # 修复pad_width格式：必须是"(轴1填充, 轴2填充)"，且填充长度为非负整数
            sliced = np.pad(
                mel_seq,
                pad_width=((0, 0), (0, pad_width)),  # 轴0（时间步）不填充，轴1（频域）后填充
                mode='constant',  # 填充模式：常数填充
                constant_values=0.0  # 填充值：0.0（明确指定，避免默认值类型问题）
            )
        X_sliced.append(sliced)
    X = X_sliced

    # 2. 训练集轻量级增强（仅50%样本增强，避免噪声泛滥）
    if is_train:
        X_augmented = []
        y_augmented = []
        for mel_seq, tag_seq in zip(X, y):
            # 保留原始样本
            X_augmented.append(mel_seq)
            y_augmented.append(tag_seq)
            # 50%概率生成增强样本，平衡数据量与噪声
            if np.random.random() < 0.5:
                aug_mel = augment_mel_spectrogram(mel_seq)
                X_augmented.append(aug_mel)
                y_augmented.append(tag_seq.copy())
        X, y = X_augmented, y_augmented
        print(f"增强后训练集规模：{len(X)}（原始样本+50%增强样本）")

    # 3. 标签编码（训练集拟合，测试集复用）
    if is_train:
        # 扁平化标签并统计分布，便于分析数据平衡性
        all_tags = [tag for seq in y for tag in seq]
        tag_count = {tag: all_tags.count(tag) for tag in set(all_tags)}
        print(f"训练集标签分布：{tag_count}")
        # 拟合标签编码器
        le = LabelEncoder()
        le.fit(all_tags)

    # 4. 标签序列编码与填充
    y_encoded = [le.transform(seq) for seq in y]
    # 确定最大长度（训练集计算，测试集复用）
    if is_train:
        max_length = max(len(seq) for seq in X)
        print(f"序列最大长度：{max_length}（用于后续填充）")

    # 5. 特征与标签填充（后填充0，保持时序完整性）
    # 特征填充（float32类型，适配模型输入）
    X_padded = sequence.pad_sequences(
        X, maxlen=max_length, dtype='float32', padding='post', truncating='post', value=0.0
    )
    # 标签填充（填充-1的编码，后续通过掩码忽略）
    no_label_code = le.transform(["-1"])[0]
    y_padded = sequence.pad_sequences(
        y_encoded, maxlen=max_length, padding='post', truncating='post', value=no_label_code
    )

    # 6. 标签独热编码（适配多分类任务）
    num_classes = len(le.classes_)
    y_onehot = to_categorical(y_padded, num_classes=num_classes)
    print(f"预处理完成：特征形状{X_padded.shape}，标签形状{y_onehot.shape}，类别数{num_classes}")

    return X_padded, y_onehot, max_length, le


# -------------------------- 4. 自定义掩码损失函数（忽略无标签帧，优化数值稳定性） --------------------------
def ignore_no_label_loss(no_label_idx):
    """
    闭包实现损失函数：仅计算有效标签（0/1/2）的损失，忽略无标签帧（-1）
    输入：no_label_idx - 无标签帧（-1）的独热编码索引
    """

    def loss(y_true, y_pred):
        # 生成掩码：有效帧（非-1）为1，无标签帧为0
        mask = 1 - y_true[:, :, no_label_idx]
        # 计算交叉熵损失（避免数值溢出，使用数值稳定版）
        ce_loss = losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        # 应用掩码，仅优化有效帧
        masked_loss = ce_loss * mask
        # 有效帧数量归一化（添加1e-8避免除以0，提升数值稳定性）
        valid_frame_count = maximum(reduce_sum(mask), 1e-8)
        return reduce_sum(masked_loss) / valid_frame_count

    return loss


# -------------------------- 5. 简化版CNN-LSTM模型（降低复杂度，抑制过拟合） --------------------------
# def build_cnn_lstm_model(input_shape, num_classes, no_label_idx):
#     """
#     优化后模型结构：轻量级CNN提取频域特征 + 精简LSTM捕捉时序关联
#     关键修改：减少参数量、增加正则化、恢复池化层
#     """
#     model = Sequential([
#         # 1. 1D-CNN层：提取局部频域特征（32滤波器，核大小5，避免过拟合）
#         Conv1D(
#             filters=32,
#             kernel_size=5,
#             activation='relu',
#             input_shape=input_shape,
#             padding='same'  # 保持时间步一致
#         ),
#         BatchNormalization(),  # 批量归一化，加速收敛
#         MaxPooling1D(pool_size=2, padding='same'),  # 池化压缩时间步，保留关键特征
#
#         # 2. Masking层：过滤填充的0值，避免干扰训练
#         Masking(mask_value=0.0),
#
#         # 3. LSTM层：精简单元数（128→64），提升正则化力度
#         LSTM(
#             units=64,
#             return_sequences=True,  # 逐帧输出，适配时序分类
#             dropout=0.4,  # 输入 dropout 提升泛化性
#             recurrent_dropout=0.3  # 循环 dropout 抑制时序过拟合
#         ),
#
#         # 4. 全连接层：精简单元数（64→32），加强正则化
#         Dense(32, activation='relu'),
#         Dropout(0.5),  # 提高dropout比例，抑制全连接层过拟合
#
#         # 5. 输出层：多分类softmax，输出每帧类别概率
#         Dense(num_classes, activation='softmax')
#     ])
#
#     # 编译模型：优化器+自定义掩码损失
#     model.compile(
#         optimizer=optimizers.Adam(learning_rate=0.0008),  # 降低初始学习率，避免震荡
#         loss=ignore_no_label_loss(no_label_idx),  # 传入无标签帧索引
#         metrics=['accuracy']  # 监控准确率，直观评估模型性能
#     )
#
#     # 打印模型结构与参数量（便于确认复杂度）
#     model.summary()
#     return model
def build_cnn_lstm_model(input_shape, num_classes, no_label_idx):
    model = Sequential([
        # 1. 1D-CNN层：提取频域特征，padding='same'保持时间步不变
        Conv1D(
            filters=32,
            kernel_size=5,
            activation='relu',
            input_shape=input_shape,
            padding='same'  # 关键：确保时间步不被CNN改变
        ),
        BatchNormalization(),  # 保留批量归一化，稳定训练
        # 👇 删除这行 MaxPooling1D，避免时间步压缩
        # MaxPooling1D(pool_size=2, padding='same'),

        # 2. Masking层：过滤填充的0值
        Masking(mask_value=0.0),

        # 3. LSTM层：保持return_sequences=True，逐帧输出（时间步与输入一致）
        LSTM(
            units=64,
            return_sequences=True,
            dropout=0.4,
            recurrent_dropout=0.3
        ),

        # 4. 全连接层与输出层
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # 输出时间步=894，与标签一致
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0008),
        loss=ignore_no_label_loss(no_label_idx),
        metrics=['accuracy']
    )
    model.summary()  # 编译前打印结构，确认输出形状为(None, 894, 4)
    return model

# -------------------------- 6. 主函数（整合数据流程与训练逻辑，优化参数） --------------------------
def main(data_path):
    # 1. 加载原始数据
    print("=" * 50)
    print("1. 开始加载数据...")
    X_raw, y_raw = load_data(data_path)
    if len(X_raw) == 0:
        print("错误：未加载到有效样本，请检查数据集路径与文件格式！")
        return

    # 2. 划分训练集与测试集（8:2，固定随机种子确保可复现）
    print("\n" + "=" * 50)
    print("2. 划分训练集与测试集...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"训练集原始样本数：{len(X_train_raw)}，测试集原始样本数：{len(X_test_raw)}")

    # 3. 数据预处理（训练集单独处理，测试集复用参数）
    print("\n" + "=" * 50)
    print("3. 预处理训练集...")
    X_train, y_train, max_length, le = preprocess_data(
        X_train_raw, y_train_raw, is_train=True
    )

    print("\n" + "=" * 50)
    print("3. 预处理测试集...")
    X_test, y_test, _, _ = preprocess_data(
        X_test_raw, y_test_raw, le=le, max_length=max_length, is_train=False
    )

    # 4. 计算无标签帧索引（用于损失函数）
    no_label_idx = np.where(le.classes_ == "-1")[0][0]
    print(f"\n无标签帧（-1）的独热编码索引：{no_label_idx}")

    # 5. 构建模型
    print("\n" + "=" * 50)
    print("4. 构建CNN-LSTM模型...")
    input_shape = (max_length, X_train.shape[2])  # (时间步, 频域维度)
    num_classes = y_train.shape[-1]
    model = build_cnn_lstm_model(input_shape, num_classes, no_label_idx)

    # 6. 定义训练回调（优化早停、模型保存、学习率调度）
    print("\n" + "=" * 50)
    print("5. 配置训练回调...")
    callbacks = [
        # 早停：监控验证准确率，耐心3轮，恢复最优权重（避免过拟合）
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1,
            mode='max'  # 准确率需最大化
        ),
        # 模型保存：仅保存验证准确率最优的模型（移除多余的save_format参数）
        ModelCheckpoint(
            'best_cnn_lstm_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
            # 此处删除 save_format='keras'，因Keras 2.15.0不支持该参数
        ),
        # 学习率调度：验证损失3轮不降则降率，避免震荡
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,  # 降率比例：0.4→原学习率*0.4
            patience=2,
            min_lr=1e-7,  # 最低学习率，避免停滞
            verbose=1
        )
    ]

    # 7. 模型训练（优化batch_size与epochs，适配CPU）
    print("\n" + "=" * 50)
    print("6. 开始模型训练...")
    history = model.fit(
        X_train, y_train,
        epochs=30,  # 减少epochs，避免过度训练
        batch_size=12,  # 小batch提升泛化性，适配CPU内存
        validation_split=0.1,  # 训练集10%作为验证集，监控过拟合
        shuffle=True,  # 每轮打乱数据，提升泛化性
        callbacks=callbacks,
        verbose=1  # 显示进度条，便于实时监控
    )

    # 8. 模型评估（测试集验证泛化能力）
    print("\n" + "=" * 50)
    print("7. 测试集评估模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"测试集最终结果：准确率={test_acc:.4f}，损失={test_loss:.4f}")

    # 9. 保存最终模型与标签编码器（便于后续推理）
    print("\n" + "=" * 50)
    print("8. 保存模型与编码器...")
    model.save('final_cnn_lstm_model.keras')
    np.save('label_encoder.npy', le.classes_)
    print("保存完成：")
    print("- 最终模型：final_cnn_lstm_model.keras")
    print("- 标签编码器：label_encoder.npy")
    print("=" * 50)


# -------------------------- 7. 程序入口（适配路径导入，增加错误处理） --------------------------
if __name__ == "__main__":
    # 加载数据集路径（优先从Constant.py导入，无则手动输入）
    try:
        from Constant import DATASET_PATH

        print(f"从Constant.py加载数据集路径：{DATASET_PATH}")
    except ImportError:
        print("未找到Constant.py，需手动输入数据集路径！")
        DATASET_PATH = input("请输入.npy数据集文件夹路径（例：D:/vocal_dataset）：").strip()

    # 路径有效性检查
    if not os.path.exists(DATASET_PATH):
        print(f"错误：路径{DATASET_PATH}不存在，请检查路径是否正确！")
    elif not any(f.endswith('.npy') for f in os.listdir(DATASET_PATH)):
        print(f"错误：路径{DATASET_PATH}下无.npy文件，请确认数据集格式！")
    else:
        # 启动主流程
        main(DATASET_PATH)
