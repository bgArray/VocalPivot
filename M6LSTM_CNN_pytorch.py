import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import copy


# -------------------------- 1. 数据加载函数（保留原有逻辑） --------------------------
def load_data(file_path):
    """加载.npy数据，提取0/1/2标签并确保特征与标签长度一致"""
    data = []
    for filename in os.listdir(file_path):
        if filename.endswith('.npy'):
            file_full_path = os.path.join(file_path, filename)
            data.extend(np.load(file_full_path, allow_pickle=True))

    X = []  # 存储mel特征序列（每个元素是一个样本的序列）
    y = []  # 存储标签序列（每个元素是一个样本的标签序列）
    for item in data:
        mel_spectrogram, tags = item
        # 确保mel和标签的时间步数量一致
        if len(mel_spectrogram) != len(tags):
            print(f"警告：mel长度({len(mel_spectrogram)})与标签长度({len(tags)})不匹配，已跳过该样本")
            continue

        # 处理每个时间步的标签：只保留0/1/2，无有效标签标记为-1
        valid_tags = []
        for tag_group in tags:
            filtered = [t for t in tag_group if t in {"0", "1", "2"}]
            valid_tags.append(filtered[0] if filtered else "-1")

        X.append(mel_spectrogram)
        y.append(valid_tags)

    return X, y


# -------------------------- 2. 数据增强函数（整合轻量级频谱增强） --------------------------
def augment_mel_spectrogram(mel_seq):
    """对单条梅尔频谱序列进行轻量级增强：频域偏移+幅度缩放+高斯噪声"""
    aug_seq = mel_seq.copy()
    np.random.seed(42)  # 保证增强可复现

    # 2.1 微小频域偏移（±2个梅尔系数，避免跨关键频带）
    shift = np.random.randint(-2, 3)
    if shift != 0:
        aug_seq = np.roll(aug_seq, shift, axis=1)  # 沿频域轴滚动
        # 偏移后空白处填充0
        if shift > 0:
            aug_seq[:, :shift] = 0.0
        else:
            aug_seq[:, shift:] = 0.0

    # 2.2 随机幅度缩放（0.8-1.2倍，模拟演唱力度变化）
    scale = np.random.uniform(0.8, 1.2)
    aug_seq = aug_seq * scale

    # 2.3 叠加高斯噪声（强度为原序列标准差的5%，避免掩盖特征）
    noise = np.random.normal(0, aug_seq.std() * 0.05, aug_seq.shape)
    aug_seq = aug_seq + noise

    return aug_seq


def augment_temporal(mel_seq, tag_seq):
    """时序裁剪增强（可选）：随机截取序列80%-90%长度，保持帧-标签对应"""
    seq_len = len(mel_seq)
    if seq_len < 10:  # 避免过短序列裁剪后无效
        return mel_seq, tag_seq
    # 截取长度与起始位置
    crop_len = np.random.randint(int(seq_len * 0.8), int(seq_len * 0.9) + 1)
    start_idx = np.random.randint(0, seq_len - crop_len + 1)
    return mel_seq[start_idx:start_idx + crop_len], tag_seq[start_idx:start_idx + crop_len]


# -------------------------- 3. 数据预处理函数（整合增强与标签处理） --------------------------
def preprocess_data(X, y, le=None, max_length=None, is_train=True):
    """
    数据预处理：频域切片+增强（训练集）+序列填充+标签编码
    :param X: 原始mel特征序列列表
    :param y: 原始标签序列列表
    :param le: 标签编码器（训练集为None，测试集传入训练集的le）
    :param max_length: 序列最大长度（训练集为None，测试集传入训练集的max_length）
    :param is_train: 是否为训练集（控制是否启用增强）
    :return: 预处理后的特征、独热编码标签、最大长度、标签编码器
    """
    # 3.1 梅尔频谱频域切片：保留200Hz-5kHz关键频带（原128维→74维）
    X_sliced = []
    for mel_seq in X:
        sliced = mel_seq[:, 0:74]  # 需根据实际梅尔滤波器组参数调整，确保覆盖目标频带
        X_sliced.append(sliced)
    X = X_sliced

    # 3.2 训练集数据增强（频谱增强+可选时序增强）
    if is_train:
        X_augmented = []
        y_augmented = []
        for mel_seq, tag_seq in zip(X, y):
            # 原始样本保留
            X_augmented.append(mel_seq)
            y_augmented.append(tag_seq)
            # 频谱增强样本
            aug_mel_spect = augment_mel_spectrogram(mel_seq)
            X_augmented.append(aug_mel_spect)
            y_augmented.append(tag_seq.copy())  # 标签与原样本一致
            # 可选：时序增强样本（进一步提升鲁棒性）
            aug_mel_temp, aug_tag_temp = augment_temporal(mel_seq, tag_seq)
            X_augmented.append(aug_mel_temp)
            y_augmented.append(aug_tag_temp)
        # 更新为增强后的数据集
        X, y = X_augmented, y_augmented

    # 3.3 标签编码（训练集拟合编码器，测试集复用）
    if is_train:
        # 扁平化所有标签用于拟合编码器
        all_tags = [tag for seq in y for tag in seq]
        le = LabelEncoder()
        le.fit(all_tags)
    # 将标签序列编码为数字
    y_encoded = [le.transform(seq) for seq in y]

    # 3.4 序列统一长度（训练集计算最大长度，测试集复用）
    if is_train:
        max_length = max(len(seq) for seq in X)
    # 特征序列填充（后填充0）
    X_padded = []
    for seq in X:
        if len(seq) < max_length:
            pad_width = max_length - len(seq)
            padded = np.pad(seq, ((0, pad_width), (0, 0)), mode='constant', constant_values=0.0)
        else:
            padded = seq[:max_length]
        X_padded.append(padded.astype('float32'))
    X_padded = np.array(X_padded)

    # 标签序列填充（后填充-1的编码）
    no_label_code = le.transform(["-1"])[0]
    y_padded = []
    for seq in y_encoded:
        if len(seq) < max_length:
            pad_width = max_length - len(seq)
            padded = np.pad(seq, (0, pad_width), mode='constant', constant_values=no_label_code)
        else:
            padded = seq[:max_length]
        y_padded.append(padded)
    y_padded = np.array(y_padded)

    # 3.5 标签独热编码（适配多分类）
    num_classes = len(le.classes_)
    y_onehot = np.zeros((len(y_padded), max_length, num_classes), dtype='float32')
    for i, seq in enumerate(y_padded):
        for j, label_idx in enumerate(seq):
            y_onehot[i, j, label_idx] = 1.0

    return X_padded, y_onehot, max_length, le


# -------------------------- 4. PyTorch Dataset类 --------------------------
class VocalDataset(Dataset):
    """PyTorch Dataset类，用于加载预处理后的数据"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------- 5. 自定义损失函数（忽略无标签帧） --------------------------
class IgnoreNoLabelLoss(nn.Module):
    """自定义损失：忽略标签为-1的帧，仅优化有效帧（0/1/2）"""
    def __init__(self, no_label_idx):
        super(IgnoreNoLabelLoss, self).__init__()
        self.no_label_idx = no_label_idx
    
    def forward(self, y_pred, y_true):
        """
        :param y_pred: (batch_size, seq_len, num_classes) 模型预测输出（已softmax）
        :param y_true: (batch_size, seq_len, num_classes) 独热编码标签
        :return: 掩码后的交叉熵损失
        """
        # 生成掩码：非-1帧为1，-1帧为0
        mask = 1 - y_true[:, :, self.no_label_idx]  # (batch_size, seq_len)
        
        # 计算交叉熵损失（逐帧计算）
        # y_pred: (batch_size, seq_len, num_classes)
        # y_true: (batch_size, seq_len, num_classes)
        # 需要转换为 (batch_size * seq_len, num_classes) 来计算损失
        batch_size, seq_len, num_classes = y_pred.shape
        y_pred_flat = y_pred.view(-1, num_classes)
        y_true_flat = y_true.view(-1, num_classes)
        mask_flat = mask.view(-1)
        
        # 计算交叉熵损失
        log_probs = torch.log(y_pred_flat + 1e-8)  # 避免log(0)
        ce_loss = -torch.sum(y_true_flat * log_probs, dim=1)  # (batch_size * seq_len,)
        
        # 应用掩码
        masked_loss = ce_loss * mask_flat
        
        # 用有效帧数量归一化（避免除以0）
        valid_frames = torch.sum(mask_flat)
        if valid_frames > 0:
            return torch.sum(masked_loss) / valid_frames
        else:
            return torch.tensor(0.0, device=y_pred.device)


# -------------------------- 6. CNN-LSTM模型定义 --------------------------
class CNNLSTMModel(nn.Module):
    """CNN-LSTM串联模型：1D-CNN提取单帧频域特征 + LSTM捕捉时序关联"""
    def __init__(self, input_dim, num_classes, max_length):
        super(CNNLSTMModel, self).__init__()
        self.max_length = max_length
        
        # 第一层CNN：提取局部频域特征
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=5,
            padding=2  # padding='same' 等价于 padding=2 (kernel_size=5)
        )
        self.bn1 = nn.BatchNorm1d(32)
        
        # 第二层CNN：深度卷积层（不改变时间步）
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1  # padding='same' 等价于 padding=1 (kernel_size=3)
        )
        self.bn2 = nn.BatchNorm1d(64)
        
        # LSTM层：增加单元数提升时序建模能力
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=256,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )
        
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        :param x: (batch_size, seq_len, input_dim)
        :return: (batch_size, seq_len, num_classes)
        """
        # 转换维度：(batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 第一层CNN + BatchNorm + ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第二层CNN + BatchNorm + ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 转换回：(batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Masking：过滤填充的0值（通过LSTM的pack_padded_sequence实现）
        # 但为了保持与Keras版本一致，这里先简单处理
        # 注意：PyTorch的LSTM会自动处理，但我们需要手动mask输出
        
        # LSTM层
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # 全连接层
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Softmax激活
        output = self.softmax(x)
        
        return output


# -------------------------- 7. 训练和评估函数 --------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, no_label_idx):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total_valid = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        y_pred = model(X_batch)
        
        # 计算损失
        loss = criterion(y_pred, y_batch)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 计算准确率（只统计有效帧）
        with torch.no_grad():
            # 获取预测类别
            pred_classes = torch.argmax(y_pred, dim=-1)  # (batch_size, seq_len)
            true_classes = torch.argmax(y_batch, dim=-1)  # (batch_size, seq_len)
            
            # 生成掩码（排除无标签帧）
            mask = (true_classes != no_label_idx).float()  # (batch_size, seq_len)
            
            # 计算正确预测数
            correct += ((pred_classes == true_classes) * mask).sum().item()
            total_valid += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_valid if total_valid > 0 else 0.0
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, no_label_idx):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_valid = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 前向传播
            y_pred = model(X_batch)
            
            # 计算损失
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            
            # 计算准确率（只统计有效帧）
            pred_classes = torch.argmax(y_pred, dim=-1)  # (batch_size, seq_len)
            true_classes = torch.argmax(y_batch, dim=-1)  # (batch_size, seq_len)
            
            # 生成掩码（排除无标签帧）
            mask = (true_classes != no_label_idx).float()  # (batch_size, seq_len)
            
            # 计算正确预测数
            correct += ((pred_classes == true_classes) * mask).sum().item()
            total_valid += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_valid if total_valid > 0 else 0.0
    
    return avg_loss, accuracy


# -------------------------- 8. 主函数（整合数据流程与模型训练） --------------------------
def main(data_path):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 8.1 加载原始数据
    print("开始加载数据...")
    X_raw, y_raw = load_data(data_path)
    print(f"数据加载完成，原始样本数: {len(X_raw)}")
    
    # 8.2 划分训练集与测试集（8:2）
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"训练集样本数: {len(X_train_raw)}, 测试集样本数: {len(X_test_raw)}")
    
    # 8.3 数据预处理（训练集单独预处理，测试集复用训练集参数）
    print("开始预处理训练集...")
    X_train, y_train, max_length, le = preprocess_data(
        X_train_raw, y_train_raw, is_train=True
    )
    print("开始预处理测试集...")
    X_test, y_test, _, _ = preprocess_data(
        X_test_raw, y_test_raw, le=le, max_length=max_length, is_train=False
    )
    print(f"预处理完成：训练集形状{X_train.shape}, 测试集形状{X_test.shape}, 类别数{y_train.shape[-1]}")
    
    # 8.4 创建Dataset和DataLoader
    train_dataset = VocalDataset(X_train, y_train)
    test_dataset = VocalDataset(X_test, y_test)
    
    # 从训练集中划分验证集（10%）
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 8.5 定义模型
    input_dim = X_train.shape[2]  # 74
    num_classes = y_train.shape[-1]  # 4
    model = CNNLSTMModel(input_dim, num_classes, max_length).to(device)
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 8.6 定义损失函数、优化器和学习率调度器
    no_label_idx = np.where(le.classes_ == "-1")[0][0]
    criterion = IgnoreNoLabelLoss(no_label_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )
    
    # 8.7 训练模型（包含早停机制）
    print("\n开始模型训练...")
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(50):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, no_label_idx
        )
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, no_label_idx)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印进度
        print(f"Epoch {epoch+1}/50 - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            # 保存最佳模型
            torch.save(best_model_state, 'best_cnn_lstm_model.pth')
            print(f"  -> 保存最佳模型 (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发，在第 {epoch+1} 轮停止训练")
            print(f"最佳验证损失: {best_val_loss:.4f}, 最佳验证准确率: {best_val_accuracy:.4f}")
            break
    
    # 8.8 加载最佳模型并评估测试集
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("\n开始模型评估...")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, no_label_idx)
    print(f"测试集最终准确率: {test_accuracy:.4f}, 测试集损失: {test_loss:.4f}")
    
    # 8.9 保存最终模型与标签编码器
    torch.save(model.state_dict(), 'final_cnn_lstm_model.pth')
    np.save('label_encoder.npy', le.classes_)
    print("最终模型（final_cnn_lstm_model.pth）与标签编码器（label_encoder.npy）保存完成")


# -------------------------- 9. 程序入口 --------------------------
if __name__ == "__main__":
    # 从Constant.py导入数据集路径（确保该文件存在且路径正确）
    try:
        from Constant import DATASET_PATH
    except ImportError:
        # 若Constant.py不存在，可直接指定路径（示例："D:/vocal_dataset"）
        DATASET_PATH = input("请输入数据集文件夹路径（如D:/vocal_dataset）：")
    
    # 检查路径有效性
    if not os.path.exists(DATASET_PATH):
        print(f"错误：数据集路径{DATASET_PATH}不存在，请检查路径是否正确！")
    else:
        main(DATASET_PATH)

