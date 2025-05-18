'''
MNIST Static HyperNetwork Example (PyTorch Annotated Version)
Based on the paper "Hypernetworks" by David Ha, Andrew Dai, and Quoc V. Le.

此程序实现了论文中的"静态超网络"概念，展示了如何将超网络应用于MNIST图像分类任务。
相比标准CNN，超网络生成模式允许更灵活和紧凑的参数表示。
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Configure numpy output format
np.set_printoptions(precision=5, edgeitems=8, linewidth=200)

# 正交初始化是论文中用于超网络的关键初始化方法之一
# 论文第4节"Weight Matrix Parameterization"中提到了初始化的重要性
def orthogonal_init(tensor, gain=1):
    """
    为权重矩阵实现正交初始化，这对于超网络的稳定训练非常重要
    
    论文中建议通过特殊初始化方法稳定超网络训练，正交初始化是其中一种推荐方法
    它能保证初始权重矩阵的行（或列）彼此正交，有助于防止梯度消失/爆炸
    """
    if isinstance(tensor, torch.nn.Parameter):
        orthogonal_init(tensor.data, gain)
        return
    if tensor.ndimension() < 2:
        return
    
    # 对于卷积层这样的多维张量，需要先展平成2D矩阵进行SVD分解
    original_shape = tensor.shape
    num_rows = original_shape[0]
    num_cols = tensor.numel() // num_rows
    
    flat_tensor = tensor.new(num_rows, num_cols).normal_(0, 1)
    
    # 利用SVD分解构造正交矩阵
    u, _, v = torch.linalg.svd(flat_tensor, full_matrices=False)
    q = u if u.shape[0] == num_rows else v
    q = q[:num_rows, :num_cols]
    
    # 应用正交化矩阵
    with torch.no_grad():
        tensor.view_as(flat_tensor).copy_(q)
        tensor.mul_(gain)
    return tensor

# 标准CNN模型类 - 作为对照组，不使用超网络
class StandardCNN(nn.Module):
    def __init__(self, f_size=7, in_size=16, out_size=16):
        """
        标准CNN模型使用常规方式定义的卷积层
        
        论文中将此作为基线模型，用于与超网络模型进行比较
        在Section 5.1中，作者使用了类似的架构设计评估静态超网络
        """
        super(StandardCNN, self).__init__()
        # 第一个卷积层 - 标准实现
        self.conv1 = nn.Conv2d(1, in_size, kernel_size=f_size, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积层 - 标准实现，这一层在超网络版本中将被替换为生成的权重
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size=f_size, padding='same')
        # 计算池化后的输出尺寸: 28x28 -> 7x7(经过两次2x2最大池化)
        self.fc = nn.Linear(out_size * 7 * 7, 10)
        
        # 使用截断正态分布初始化权重，与论文中的实现一致
        nn.init.normal_(self.conv1.weight, std=0.01)
        self.conv1.bias.data.fill_(0.0)
        nn.init.normal_(self.conv2.weight, std=0.01)
        self.conv2.bias.data.fill_(0.0)
        
    def forward(self, x):
        """前向传播逻辑"""
        # 第一个卷积+池化块
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积+池化块
        x = self.pool(F.relu(self.conv2(x)))
        # 展平并连接到全连接层
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_conv2_weights(self):
        """返回第二个卷积层的权重，用于可视化和比较"""
        return self.conv2.weight.detach().cpu().numpy()

# 超网络CNN模型 - 实现了论文中的静态超网络概念
class HyperCNN(nn.Module):
    def __init__(self, f_size=7, in_size=16, out_size=16, z_dim=4):
        """
        超网络CNN模型，使用一个小型网络生成第二个卷积层的权重
        
        关键参数:
        - f_size: 卷积核大小，论文中使用3×3至7×7不等
        - in_size/out_size: 输入/输出通道数
        - z_dim: 潜在向量维度，控制超网络的表达能力
        
        论文3.1节"Static Hypernetwork"描述了此类静态超网络的设计
        """
        super(HyperCNN, self).__init__()
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.z_dim = z_dim
        
        # 第一个卷积层（标准实现，不由超网络生成）
        self.conv1 = nn.Conv2d(1, in_size, kernel_size=f_size, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        
        # 超网络部分 - 用于生成第二个卷积层的权重
        # -------------------------------------------
        
        # 1. z_signal - 超网络的输入信号
        # 论文3.1节提到，静态超网络的输入是一个固定的潜在向量z
        self.z_signal = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        
        # 2. 第一层变换 - z → 中间表示
        # 论文中的实现使用了多层感知机作为超网络，这里使用矩阵W2和偏置b2
        self.w2 = nn.Parameter(torch.randn(z_dim, in_size * z_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(in_size * z_dim))
        
        # 3. 第二层变换 - 中间表示 → 最终权重
        # 生成卷积核权重的最后一层映射
        self.w1 = nn.Parameter(torch.randn(z_dim, out_size * f_size * f_size) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(out_size * f_size * f_size))
        
        # 第二个卷积层的偏置（不由超网络生成）
        self.conv2_bias = nn.Parameter(torch.zeros(out_size))
        
        # 全连接层
        self.fc = nn.Linear(out_size * 7 * 7, 10)
        
        # 初始化权重
        nn.init.normal_(self.conv1.weight, std=0.01)
        self.conv1.bias.data.fill_(0.0)
        nn.init.normal_(self.w1, std=0.01)
        nn.init.normal_(self.w2, std=0.01)
        
    def generate_conv2_weights(self):
        """
        使用超网络生成第二个卷积层的权重
        
        这实现了论文3.1节描述的静态超网络的核心功能:
        1. 从输入z开始
        2. 通过网络层进行变换
        3. 将输出重塑为目标网络(主网络)所需的权重形状
        
        在静态情况下，z是可学习的参数而非网络的输入
        """
        # 步骤1: z_signal通过第一层变换
        h_in = torch.matmul(self.z_signal, self.w2) + self.b2
        h_in = h_in.reshape(self.in_size, self.z_dim)
        
        # 步骤2: 中间表示通过第二层变换
        h_final = torch.matmul(h_in, self.w1) + self.b1
        
        # 步骤3: 重塑为卷积核格式
        # 将一维向量重塑为四维卷积核 [out_channels, in_channels, height, width]
        kernel = h_final.reshape(self.out_size, self.in_size, self.f_size, self.f_size)
        return kernel
        
    def forward(self, x):
        """
        前向传播函数，结合了标准层和由超网络生成的层
        
        论文第3节描述了如何将超网络与主网络结合使用:
        1. 超网络生成主网络的部分权重
        2. 主网络使用这些权重执行正常的前向传播
        """
        # 第一个卷积+池化块 (标准)
        x = self.pool(F.relu(self.conv1(x)))
        
        # 动态生成第二个卷积层的权重
        conv2_weights = self.generate_conv2_weights()
        
        # 第二个卷积+池化块 (使用超网络生成的权重)
        x = F.conv2d(x, conv2_weights, bias=self.conv2_bias, padding='same')
        x = self.pool(F.relu(x))
        
        # 展平并通过全连接层
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
    
    def get_conv2_weights(self):
        """返回由超网络生成的第二个卷积层权重，用于可视化和分析"""
        return self.generate_conv2_weights().detach().cpu().numpy()

# 训练函数 - 实现模型训练的一个完整epoch
def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    """
    训练一个epoch
    
    论文中的训练细节在实验部分(第5节)有详细描述，包括学习率、批大小等
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # 学习率调度器 - 论文中提到使用学习率衰减
    lr_scheduler = None
    if hasattr(optimizer, "param_groups"):
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # 梯度裁剪 - 论文中提到这对超网络训练很重要
        # 论文4.2节指出"梯度裁剪对超网络的训练至关重要"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        
        # 跟踪训练指标
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0:
            print(f'Batch: {batch_idx}/{len(train_loader)} '
                  f'Loss: {loss.item():.4f} '
                  f'Err: {1-correct/total:.4f}')
    
    # 更新学习率
    if lr_scheduler:
        lr_scheduler.step()
    
    train_loss /= len(train_loader)
    train_acc = correct / total
    train_err = 1.0 - train_acc
    
    return train_loss, train_err, optimizer.param_groups[0]['lr']

# 评估函数 - 在验证集或测试集上评估模型
def evaluate(model, device, data_loader):
    """评估模型在测试/验证集上的性能"""
    model.eval()
    loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    loss /= total
    acc = correct / total
    err = 1.0 - acc
    
    return loss, err

# 卷积滤波器统计函数 - 用于分析生成的权重
def show_filter_stats(conv_filter, title="Filter Stats"):
    """
    显示卷积滤波器的统计信息
    
    在论文第5.2节和图2中，作者分析了由超网络生成的滤波器的统计属性，
    并将其与标准CNN权重进行了比较
    """
    print(f"\n{title}:")
    print(f"Filter shape: {conv_filter.shape}")
    print(f"mean = {np.mean(conv_filter):.5f}")
    print(f"stddev = {np.std(conv_filter):.5f}")
    print(f"max = {np.max(conv_filter):.5f}")
    print(f"min = {np.min(conv_filter):.5f}")
    print(f"median = {np.median(conv_filter):.5f}")

# 参数计数函数 - 对比标准CNN与超网络CNN的参数数量
def count_parameters(model):
    """
    计算并打印模型的可训练参数数量
    
    论文4.2节讨论了超网络可以大幅减少参数数量，这个函数用于验证这一点
    """
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {param.shape}, {num_params}")
    
    print(f"Total trainable parameters: {total_params}")
    return total_params

# 卷积滤波器可视化函数
def visualize_filters(conv_filter, save_path):
    """
    可视化卷积滤波器并保存为图像文件
    
    论文图2展示了超网络生成的滤波器与标准CNN权重的比较，
    本函数复现了类似的可视化方法
    """
    f_size = conv_filter.shape[2]
    in_dim = conv_filter.shape[1]
    out_dim = conv_filter.shape[0]
    
    # 创建可视化画布
    canvas = np.zeros(((f_size+1)*out_dim, (f_size+1)*in_dim))
    
    # 绘制每个滤波器
    for i in range(out_dim):
        for j in range(in_dim):
            canvas[i*(f_size+1):i*(f_size+1)+f_size, j*(f_size+1):j*(f_size+1)+f_size] = conv_filter[i, j]
    
    # 添加边框
    canvas_fixed = np.zeros((canvas.shape[0]+1, canvas.shape[1]+1))
    canvas_fixed[1:, 1:] = canvas
    
    # 绘制并保存
    plt.figure(figsize=(16, 16))
    plt.imshow(canvas_fixed.T, cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def main():
    """
    主函数，实现整个实验流程
    
    论文第5节描述了作者进行的各种实验，包括在MNIST上评估静态超网络的表现
    """
    print("Starting MNIST Hypernetwork Example (PyTorch Version)...")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查是否有CUDA可用
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 超参数设置 - 与论文中的实验设置一致
    # 论文5.1节描述了各种超参数选择
    batch_size = 1000  # 匹配原始实现
    test_batch_size = 1000
    epochs = 10  # 原始实现为50，这里减少了训练轮数以加快演示
    lr = 0.005    # 初始学习率，由论文5.1节指定
    min_lr = 0.0001
    f_size = 7    # 卷积核大小，论文中使用3x3至7x7不等
    in_size = 16  # 输入通道数
    out_size = 16 # 输出通道数
    z_dim = 4     # 超网络潜在向量维度，控制表达能力
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载MNIST数据集
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # MNIST测试集有10,000个样本
    # 我们将使用全部测试集，因为训练集已有60,000个样本
    test_dataset_full = test_dataset
    test_dataset = test_dataset_full
    
    # 从训练数据中划分出验证集
    train_size = len(train_dataset) - 10000  # 使用10,000个样本作为验证集
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, 10000]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    
    # 训练标准CNN模型
    # 论文5.1节将标准CNN作为基线与超网络CNN进行比较
    print("\nTraining standard CNN model...")
    model_std = StandardCNN(f_size=f_size, in_size=in_size, out_size=out_size).to(device)
    optimizer_std = optim.Adam(model_std.parameters(), lr=lr)
    
    best_val_err = 1.0
    best_val_loss = float('inf')
    
    # 计算模型参数
    count_parameters(model_std)
    
    # 训练循环 - 标准CNN
    for epoch in range(epochs):
        # 训练并获取指标
        train_loss, train_err, current_lr = train(model_std, device, train_loader, optimizer_std, epoch)
        
        # 在验证集上评估
        val_loss, val_err = evaluate(model_std, device, val_loader)
        
        # 追踪最佳模型
        test_err = 0.0
        if val_err <= best_val_err:
            best_val_err = val_err
            best_val_loss = val_loss
            
            # 在测试集上评估 - 仅在特定epoch进行以节省时间
            if epoch == 0 or epoch % 3 == 0 or epoch == epochs-1:
                test_loss, test_err = evaluate(model_std, device, test_loader)
        
        # 打印epoch结果
        print(f"Epoch: {epoch}, "
              f"train_loss={train_loss:.4f}, "
              f"train_err={train_err:.4f}, "
              f"val_err={val_err:.4f}, "
              f"best_val_err={best_val_err:.4f}, "
              f"test_err={test_err:.4f}, "
              f"lr={current_lr:.6f}")
    
    # 可视化标准CNN滤波器
    # 论文图2比较了标准CNN与超网络CNN的滤波器
    conv_filter_std = model_std.get_conv2_weights()
    show_filter_stats(conv_filter_std, "Standard CNN Filter Stats")
    visualize_filters(conv_filter_std, "pytorch_standard_cnn_filter.png")
    
    # 训练超网络CNN模型
    # 论文5.1节详细描述了超网络CNN的训练和表现
    print("\nTraining hypernetwork CNN model...")
    model_hyper = HyperCNN(f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim).to(device)
    optimizer_hyper = optim.Adam(model_hyper.parameters(), lr=lr)
    
    best_val_err = 1.0
    best_val_loss = float('inf')
    
    # 计算模型参数
    # 论文4.2节讨论了超网络如何减少参数数量
    count_parameters(model_hyper)
    
    # 训练循环 - 超网络CNN
    for epoch in range(epochs):
        # 训练并获取指标
        train_loss, train_err, current_lr = train(model_hyper, device, train_loader, optimizer_hyper, epoch)
        
        # 在验证集上评估
        val_loss, val_err = evaluate(model_hyper, device, val_loader)
        
        # 追踪最佳模型
        test_err = 0.0
        if val_err <= best_val_err:
            best_val_err = val_err
            best_val_loss = val_loss
            
            # 在测试集上评估 - 仅在特定epoch进行以节省时间
            if epoch == 0 or epoch % 3 == 0 or epoch == epochs-1:
                test_loss, test_err = evaluate(model_hyper, device, test_loader)
        
        # 打印epoch结果
        print(f"Epoch: {epoch}, "
              f"train_loss={train_loss:.4f}, "
              f"train_err={train_err:.4f}, "
              f"val_err={val_err:.4f}, "
              f"best_val_err={best_val_err:.4f}, "
              f"test_err={test_err:.4f}, "
              f"lr={current_lr:.6f}")
    
    # 可视化超网络CNN滤波器
    # 论文图2展示了超网络生成的滤波器，我们在此复现
    conv_filter_hyper = model_hyper.get_conv2_weights()
    show_filter_stats(conv_filter_hyper, "Hypernetwork CNN Filter Stats")
    visualize_filters(conv_filter_hyper, "pytorch_hypernetwork_cnn_filter.png")
    
    print("\nExecution complete. Filter visualizations saved as PNG files.")

if __name__ == "__main__":
    main()
