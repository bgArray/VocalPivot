import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score  # 导入混淆矩阵函数
# import seaborn as sns
from keras.models import load_model
from sklearn.model_selection import train_test_split
from M1.M2CNN import load_dataset_M1  # 导入你的数据加载函数

# ---------------------- 1. 加载模型和测试数据 ----------------------
# 加载训练好的h5模型（替换为你的模型实际路径）
model = load_model("./models/vocal_tech_cnn_M2E3_85.4.h5")
print("模型加载完成")

# 加载数据集（适配你的load_dataset_M1参数，无count则删除count=10）
try:
    X, y = load_dataset_M1()  # 若你的函数需要count参数
except TypeError:
    X, y = load_dataset_M1()  # 若你的函数不需要额外参数

# 划分测试集（与训练时完全一致的参数，确保数据匹配）
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"测试集规模：X_test={X_test.shape}, y_test={y_test.shape}")

# ---------------------- 2. 计算模型预测结果 ----------------------
# 预测测试集概率并转换为类别标签
y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)  # 0=真声，1=混声，2=假声，3=说话（与原代码标签一致）

# ---------------------- 3. 先计算混淆矩阵cm（关键：这步要在使用cm之前） ----------------------
label_names = ["真声", "混声", "假声", "说话"]  # 标签映射（必须与原代码一致）
cm = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵，此时cm变量正式定义

# ---------------------- 4. 计算平均准确率（基于已定义的cm） ----------------------
# 1. 各类别准确率（对角线元素=正确预测数，每行总和=该类真实样本数）
class_sample_count = cm.sum(axis=1)  # 各类别真实样本数量
class_acc = cm.diagonal() / class_sample_count  # 各类别准确率（处理分母为0的情况）
class_acc = np.nan_to_num(class_acc)  # 若某类无样本（分母为0），准确率设为0

# 2. 宏观平均准确率（平等对待每类，取算术平均）
macro_acc = np.mean(class_acc)

# 3. 加权平均准确率（按各类别样本量加权，贴合数据分布）
weighted_acc = np.sum(class_acc * class_sample_count) / np.sum(class_sample_count)

# 4. 总体准确率（验证与训练日志一致性）
overall_acc = accuracy_score(y_test, y_pred)

# 打印所有准确率指标
print("\n" + "=" * 50)
print("准确率指标汇总：")
print(f"总体准确率：{overall_acc:.4f}")
print(f"宏观平均准确率（平等对待每类）：{macro_acc:.4f}")
print(f"加权平均准确率（按样本量加权）：{weighted_acc:.4f}")
print("\n各类别准确率：")
for i, (label, acc, count) in enumerate(zip(label_names, class_acc, class_sample_count)):
    print(f"{label}：{acc:.4f}（真实样本数：{count}）")
print("=" * 50 + "\n")


# ---------------------- 5. 绘制混淆矩阵：按类别独立上色（仅修改图例部分） ----------------------
def custom_cmap_for_classes(cm, label_names):
    """为每个真实类别（行）生成独立色系：真声蓝、混声绿、假声橙、说话紫"""
    n_classes = len(label_names)
    rgb_matrix = np.zeros((n_classes, n_classes, 3))  # RGB颜色矩阵（行×列×3通道）

    # 1. 预先定义完整的类别基础色（直接在这里定义，后续图例也用这个列表）
    global base_colors  # 设为全局变量，方便图例调用
    base_colors = [
        [0.2, 0.4, 0.8],  # 真声：蓝色
        [0.2, 0.8, 0.4],  # 混声：绿色
        [0.8, 0.6, 0.2],  # 假声：橙色
        [0.6, 0.2, 0.8]  # 说话：紫色
    ]

    # 每类单独归一化，按数值深浅调整颜色亮度
    for true_class in range(n_classes):
        class_data = cm[true_class]  # 当前类的所有预测结果
        if np.max(class_data) == 0:
            continue  # 无样本时跳过
        normalized_data = (class_data / np.max(class_data)) * 0.8  # 归一化到0-0.8（留亮度基底）
        for pred_class in range(n_classes):
            brightness = 1 - normalized_data[pred_class]  # 数值越大→亮度越低→颜色越深
            rgb_matrix[true_class, pred_class] = [
                base_colors[true_class][0] * brightness,
                base_colors[true_class][1] * brightness,
                base_colors[true_class][2] * brightness
            ]
    return rgb_matrix


# 生成自定义颜色矩阵
custom_rgb = custom_cmap_for_classes(cm, label_names)

# 绘制混淆矩阵（其他代码不变，仅修改图例生成部分）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(12, 10))

# 渲染自定义颜色背景
ax.imshow(custom_rgb, aspect='auto')

# 添加数值标注（白色粗体，确保在深色背景上清晰）
for i in range(len(label_names)):
    for j in range(len(label_names)):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center", color="white", fontsize=11, fontweight='bold'
        )

# 设置坐标轴和标题（这部分不变）
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names, fontsize=12)
ax.set_yticklabels(label_names, fontsize=12)
ax.set_xlabel('预测标签', fontsize=14, fontweight='bold')
ax.set_ylabel('真实标签', fontsize=14, fontweight='bold')
ax.set_title(
    f'声乐技巧分类混淆矩阵\n（总体准确率：{overall_acc:.2%} | 宏观平均准确率：{macro_acc:.2%}）',
    fontsize=16, fontweight='bold', pad=20
)

# ---------------------- 关键修改：正确生成图例 ----------------------
# 直接遍历base_colors列表（每个元素是完整的RGB值），不再拆分
legend_elements = []
for i in range(len(label_names)):
    # 用base_colors[i]（完整RGB列表）作为facecolor参数
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=base_colors[i], label=label_names[i]))
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

# 调整布局并保存（这部分不变）
plt.tight_layout()
plt.savefig(
    "confusion_matrix_custom_color.png",
    dpi=300, bbox_inches="tight", facecolor='white'
)
plt.show()
