import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import copy


# -------------------------- 1. 多标签数据加载函数 --------------------------
def load_multilabel_data(file_path):
    """
    加载.npy数据，支持多标签格式
    每个时间步可以有多个标签（如同时有真声和颤音）
    """
    data = []
    for filename in os.listdir(file_path):
        if filename.endswith('.npy'):
            file_full_path = os.path.join(file_path, filename)
            try:
                loaded_data = np.load(file_full_path, allow_pickle=True)
                data.extend(loaded_data)
            except Exception as e:
                print(f"警告：加载文件 {filename} 时出错: {e}")
                continue

    X = []  # 存储mel特征序列
    y = []  # 存储多标签序列（每个元素是一个标签列表，可能包含多个标签）
    
    for item in data:
        if len(item) != 2:
            print(f"警告：数据格式不正确，跳过该样本")
            continue
            
        mel_spectrogram, tags = item
        # 确保mel和标签的时间步数量一致
        if len(mel_spectrogram) != len(tags):
            print(f"警告：mel长度({len(mel_spectrogram)})与标签长度({len(tags)})不匹配，已跳过该样本")
            continue

        # 处理每个时间步的标签：保留所有有效标签
        valid_tags = []
        for tag_group in tags:
            if isinstance(tag_group, list):
                # 如果已经是列表，过滤有效标签
                filtered = [str(t).strip() for t in tag_group if str(t).strip() not in ["", "None", "nan"]]
                if not filtered:
                    filtered = ["-1"]  # 无有效标签
            elif isinstance(tag_group, str):
                # 如果是字符串，按逗号分割
                if tag_group.strip() in ["", "None", "nan"]:
                    filtered = ["-1"]
                else:
                    filtered = [t.strip() for t in tag_group.split(",") if t.strip() not in ["", "None", "nan"]]
                    if not filtered:
                        filtered = ["-1"]
            else:
                # 其他情况，转换为字符串
                tag_str = str(tag_group).strip()
                if tag_str in ["", "None", "nan"]:
                    filtered = ["-1"]
                else:
                    filtered = [t.strip() for t in tag_str.split(",") if t.strip() not in ["", "None", "nan"]]
                    if not filtered:
                        filtered = ["-1"]
            
            valid_tags.append(filtered)

        X.append(mel_spectrogram)
        y.append(valid_tags)

    return X, y


# -------------------------- 2. 数据增强函数（与原版相同） --------------------------
def augment_mel_spectrogram(mel_seq):
    """对单条梅尔频谱序列进行轻量级增强：频域偏移+幅度缩放+高斯噪声"""
    aug_seq = mel_seq.copy()
    np.random.seed(42)

    # 微小频域偏移（±2个梅尔系数）
    shift = np.random.randint(-2, 3)
    if shift != 0:
        aug_seq = np.roll(aug_seq, shift, axis=1)
        if shift > 0:
            aug_seq[:, :shift] = 0.0
        else:
            aug_seq[:, shift:] = 0.0

    # 随机幅度缩放（0.8-1.2倍）
    scale = np.random.uniform(0.8, 1.2)
    aug_seq = aug_seq * scale

    # 叠加高斯噪声
    noise = np.random.normal(0, aug_seq.std() * 0.05, aug_seq.shape)
    aug_seq = aug_seq + noise

    return aug_seq


def augment_temporal(mel_seq, tag_seq):
    """时序裁剪增强：随机截取序列80%-90%长度"""
    seq_len = len(mel_seq)
    if seq_len < 10:
        return mel_seq, tag_seq
    crop_len = np.random.randint(int(seq_len * 0.8), int(seq_len * 0.9) + 1)
    start_idx = np.random.randint(0, seq_len - crop_len + 1)
    return mel_seq[start_idx:start_idx + crop_len], tag_seq[start_idx:start_idx + crop_len]


# -------------------------- 3. 多标签数据预处理函数 --------------------------
def preprocess_multilabel_data(X, y, label_to_idx=None, max_length=None, is_train=True):
    """
    多标签数据预处理：频域切片+增强+序列填充+多标签编码
    :param X: 原始mel特征序列列表
    :param y: 原始多标签序列列表（每个元素是标签列表）
    :param label_to_idx: 标签到索引的映射（训练集为None，测试集传入训练集的映射）
    :param max_length: 序列最大长度
    :param is_train: 是否为训练集
    :return: 预处理后的特征、多标签编码、最大长度、标签映射
    """
    # 3.1 梅尔频谱频域切片：保留200Hz-5kHz关键频带（原128维→74维）
    X_sliced = []
    for mel_seq in X:
        if mel_seq.shape[1] >= 74:
            sliced = mel_seq[:, 0:74]
        else:
            # 如果维度不足，进行填充
            pad_width = 74 - mel_seq.shape[1]
            sliced = np.pad(mel_seq, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
        X_sliced.append(sliced)
    X = X_sliced

    # 3.2 训练集数据增强
    if is_train:
        X_augmented = []
        y_augmented = []
        for mel_seq, tag_seq in zip(X, y):
            # 原始样本
            X_augmented.append(mel_seq)
            y_augmented.append(tag_seq)
            # 频谱增强样本
            aug_mel_spect = augment_mel_spectrogram(mel_seq)
            X_augmented.append(aug_mel_spect)
            y_augmented.append([tags.copy() for tags in tag_seq])  # 深拷贝标签
            # 时序增强样本
            aug_mel_temp, aug_tag_temp = augment_temporal(mel_seq, tag_seq)
            X_augmented.append(aug_mel_temp)
            y_augmented.append(aug_tag_temp)
        X, y = X_augmented, y_augmented

    # 3.3 构建标签到索引的映射（训练集构建，测试集复用）
    if is_train:
        # 收集所有出现的标签
        all_labels = set()
        for tag_seq in y:
            for tag_list in tag_seq:
                all_labels.update(tag_list)
        
        # 排序标签以确保一致性（-1放在最后）
        sorted_labels = sorted([l for l in all_labels if l != "-1"]) + ["-1"]
        label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        num_labels = len(label_to_idx)
    else:
        num_labels = len(label_to_idx)

    # 3.4 序列统一长度
    if is_train:
        max_length = max(len(seq) for seq in X)
    
    # 特征序列填充
    X_padded = []
    for seq in X:
        if len(seq) < max_length:
            pad_width = max_length - len(seq)
            padded = np.pad(seq, ((0, pad_width), (0, 0)), mode='constant', constant_values=0.0)
        else:
            padded = seq[:max_length]
        X_padded.append(padded.astype('float32'))
    X_padded = np.array(X_padded)

    # 多标签编码（multi-hot encoding）
    y_multihot = np.zeros((len(y), max_length, num_labels), dtype='float32')
    for i, tag_seq in enumerate(y):
        for j, tag_list in enumerate(tag_seq):
            if j >= max_length:
                break
            # 为每个标签设置1
            for tag in tag_list:
                if tag in label_to_idx:
                    y_multihot[i, j, label_to_idx[tag]] = 1.0

    return X_padded, y_multihot, max_length, label_to_idx


# -------------------------- 4. PyTorch Dataset类 --------------------------
class MultiLabelVocalDataset(Dataset):
    """PyTorch Dataset类，用于加载多标签数据"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------- 5. 多标签损失函数（忽略无标签帧） --------------------------
class MultiLabelIgnoreNoLabelLoss(nn.Module):
    """多标签损失：忽略标签为-1的帧，仅优化有效帧"""
    def __init__(self, no_label_idx):
        super(MultiLabelIgnoreNoLabelLoss, self).__init__()
        self.no_label_idx = no_label_idx
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, y_pred, y_true):
        """
        :param y_pred: (batch_size, seq_len, num_labels) 模型预测输出（logits）
        :param y_true: (batch_size, seq_len, num_labels) 多标签编码
        :return: 掩码后的BCE损失
        """
        # 生成掩码：非-1帧为1，-1帧为0
        # 如果-1标签存在，则检查该位置是否为1
        mask = 1 - y_true[:, :, self.no_label_idx]  # (batch_size, seq_len)
        
        # 计算BCE损失（逐帧逐标签）
        batch_size, seq_len, num_labels = y_pred.shape
        y_pred_flat = y_pred.view(-1, num_labels)
        y_true_flat = y_true.view(-1, num_labels)
        mask_flat = mask.view(-1, 1)  # (batch_size * seq_len, 1)
        
        # 计算BCE损失
        bce_loss = self.bce_loss(y_pred_flat, y_true_flat)  # (batch_size * seq_len, num_labels)
        
        # 应用掩码（只对有效帧计算损失）
        masked_loss = bce_loss * mask_flat
        
        # 用有效帧数量归一化
        valid_frames = torch.sum(mask_flat)
        if valid_frames > 0:
            return torch.sum(masked_loss) / valid_frames
        else:
            return torch.tensor(0.0, device=y_pred.device)


# -------------------------- 6. 轻量化多标签CNN-LSTM模型 --------------------------
class LightweightMultiLabelCNNLSTM(nn.Module):
    """
    轻量化多标签CNN-LSTM模型
    设计思路：
    1. 使用深度可分离卷积减少参数量
    2. 共享特征提取头，然后分支到不同的标签分类器
    3. 使用注意力机制聚焦关键帧
    """
    def __init__(self, input_dim, num_labels, max_length):
        super(LightweightMultiLabelCNNLSTM, self).__init__()
        self.max_length = max_length
        self.num_labels = num_labels
        
        # 共享特征提取头：轻量化CNN
        # 第一层：标准卷积
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(32)
        
        # 第二层：深度可分离卷积（减少参数量）
        self.depthwise_conv = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            groups=32  # 深度可分离卷积
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        
        # LSTM层：捕捉时序特征
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,  # 减少隐藏单元数以降低参数量
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )
        
        # 注意力机制：聚焦关键帧
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 共享全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # 多标签分类头：每个标签独立分类
        self.label_classifiers = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_labels)
        ])
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        :param x: (batch_size, seq_len, input_dim)
        :return: (batch_size, seq_len, num_labels) logits
        """
        # 转换维度：(batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 第一层CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 深度可分离卷积
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 转换回：(batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # 注意力机制（可选，这里简化处理）
        # 应用共享全连接层
        shared_features = self.shared_fc(lstm_out)  # (batch_size, seq_len, 64)
        
        # 多标签分类：每个标签独立预测
        outputs = []
        for classifier in self.label_classifiers:
            output = classifier(shared_features)  # (batch_size, seq_len, 1)
            outputs.append(output)
        
        # 拼接所有标签的输出
        multi_label_output = torch.cat(outputs, dim=-1)  # (batch_size, seq_len, num_labels)
        
        return multi_label_output


# -------------------------- 7. 训练和评估函数 --------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, no_label_idx):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_labels = 0
    
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
        
        # 计算准确率（多标签准确率：所有标签都预测正确才算正确）
        with torch.no_grad():
            # 应用sigmoid得到概率
            y_pred_proba = torch.sigmoid(y_pred)
            y_pred_binary = (y_pred_proba > 0.5).float()
            
            # 生成掩码（排除无标签帧）
            mask = (1 - y_batch[:, :, no_label_idx]).unsqueeze(-1)  # (batch_size, seq_len, 1)
            
            # 计算每个标签的准确率
            correct = (y_pred_binary == y_batch).float() * mask
            total_correct += correct.sum().item()
            total_labels += (mask.sum().item() * y_batch.shape[-1])
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_labels if total_labels > 0 else 0.0
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, no_label_idx):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_labels = 0
    
    # 用于计算每个标签的精确率、召回率、F1
    label_stats = {}
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 前向传播
            y_pred = model(X_batch)
            
            # 计算损失
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            
            # 计算准确率
            y_pred_proba = torch.sigmoid(y_pred)
            y_pred_binary = (y_pred_proba > 0.5).float()
            
            # 生成掩码
            mask = (1 - y_batch[:, :, no_label_idx]).unsqueeze(-1)
            
            # 计算准确率
            correct = (y_pred_binary == y_batch).float() * mask
            total_correct += correct.sum().item()
            total_labels += (mask.sum().item() * y_batch.shape[-1])
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_labels if total_labels > 0 else 0.0
    
    return avg_loss, accuracy


# -------------------------- 8. 主函数 --------------------------
def main(data_path):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 8.1 加载原始数据
    print("开始加载数据...")
    X_raw, y_raw = load_multilabel_data(data_path)
    print(f"数据加载完成，原始样本数: {len(X_raw)}")
    
    # 打印一些标签统计信息
    all_labels = []
    for tag_seq in y_raw:
        for tag_list in tag_seq:
            all_labels.extend(tag_list)
    label_counts = Counter(all_labels)
    print(f"\n标签统计:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} 次")
    
    # 8.2 划分训练集与测试集（8:2）
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"\n训练集样本数: {len(X_train_raw)}, 测试集样本数: {len(X_test_raw)}")
    
    # 8.3 数据预处理
    print("\n开始预处理训练集...")
    X_train, y_train, max_length, label_to_idx = preprocess_multilabel_data(
        X_train_raw, y_train_raw, is_train=True
    )
    print("开始预处理测试集...")
    X_test, y_test, _, _ = preprocess_multilabel_data(
        X_test_raw, y_test_raw, label_to_idx=label_to_idx, max_length=max_length, is_train=False
    )
    print(f"预处理完成：训练集形状{X_train.shape}, 测试集形状{X_test.shape}, 标签数{len(label_to_idx)}")
    
    # 打印标签映射
    print(f"\n标签映射:")
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    for idx in sorted(idx_to_label.keys()):
        print(f"  {idx}: {idx_to_label[idx]}")
    
    # 8.4 创建Dataset和DataLoader
    train_dataset = MultiLabelVocalDataset(X_train, y_train)
    test_dataset = MultiLabelVocalDataset(X_test, y_test)
    
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
    num_labels = len(label_to_idx)
    model = LightweightMultiLabelCNNLSTM(input_dim, num_labels, max_length).to(device)
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 8.6 定义损失函数、优化器和学习率调度器
    no_label_idx = label_to_idx.get("-1", -1)
    if no_label_idx == -1:
        print("警告：未找到-1标签，将使用所有帧计算损失")
        no_label_idx = num_labels  # 设置为不存在的索引，掩码将全为1
    
    criterion = MultiLabelIgnoreNoLabelLoss(no_label_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )
    
    # 8.7 训练模型
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
            torch.save(best_model_state, 'best_multilabel_cnn_lstm_model.pth')
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
    
    # 8.9 保存最终模型与标签映射
    torch.save(model.state_dict(), 'final_multilabel_cnn_lstm_model.pth')
    # 保存标签映射（使用numpy保存字典）
    np.save('label_to_idx.npy', np.array([label_to_idx], dtype=object))
    # 同时保存为JSON格式以便读取
    import json
    with open('label_to_idx.json', 'w', encoding='utf-8') as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)
    print("最终模型（final_multilabel_cnn_lstm_model.pth）与标签映射（label_to_idx.npy, label_to_idx.json）保存完成")


# -------------------------- 9. 程序入口 --------------------------
if __name__ == "__main__":
    # 从Constant.py导入数据集路径
    try:
        from Constant import DATASET_PATH
    except ImportError:
        DATASET_PATH = input("请输入数据集文件夹路径（如./dataset/）：")
    
    # 检查路径有效性
    if not os.path.exists(DATASET_PATH):
        print(f"错误：数据集路径{DATASET_PATH}不存在，请检查路径是否正确！")
    else:
        main(DATASET_PATH)

