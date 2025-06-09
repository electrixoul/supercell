'''
CIFAR-100 HyperNetwork ALL LAYERS VERSION (Fully Annotated)
============================================================
基于cifar100_hypernetwork_fixed.py改进的全层HyperNetwork实现
所有卷积层和全连接层的权重都由独立的HyperNetwork动态生成

核心思想：
- 不直接学习网络权重，而是学习生成权重的"超网络"
- 每层有一个低维Z信号(8维)，通过线性变换生成该层的所有权重
- 大幅减少需要学习的参数数量，提高参数效率
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

def set_device():
    """
    设备检测和配置函数
    优先级：MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

class HyperCNNCIFAR_AllLayers(nn.Module):
    """
    全层HyperNetwork CIFAR-100分类器
    
    架构特点：
    - 3个卷积层 + 2个全连接层
    - 每层权重都由独立的HyperNetwork生成
    - 5个8维Z信号分别控制5层的权重生成
    - 总参数约6M，其中99.99%为HyperNetwork相关参数
    """
    
    def __init__(self, f_size=5, in_size=32, out_size=64, z_dim=8):
        """
        初始化HyperNetwork架构
        
        参数：
        - f_size: 卷积核尺寸 (Conv1,Conv2用5x5, Conv3用3x3)
        - in_size: Conv1输出通道数 (32)
        - out_size: Conv2输出通道数 (64)  
        - z_dim: Z信号维度 (8维)
        """
        super(HyperCNNCIFAR_AllLayers, self).__init__()
        
        # 保存架构参数
        self.f_size = f_size      # 卷积核大小
        self.in_size = in_size    # 第一层卷积输出通道数
        self.out_size = out_size  # 第二层卷积输出通道数
        self.z_dim = z_dim        # Z信号维度
        
        # 固定的网络组件（不由HyperNetwork生成）
        self.pool = nn.MaxPool2d(2, 2)     # 2x2最大池化，步长2
        self.dropout = nn.Dropout(0.5)     # 50%的dropout，防止过拟合
        
        # ========== Z信号参数 ==========
        # 每层一个独立的Z信号，用于生成该层的权重
        # Z信号是可学习参数，通过反向传播优化
        # 初始化为小的随机值，避免训练初期权重过大
        self.z_signal_conv1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)  # Conv1的Z信号
        self.z_signal_conv2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)  # Conv2的Z信号
        self.z_signal_conv3 = nn.Parameter(torch.randn(1, z_dim) * 0.01)  # Conv3的Z信号
        self.z_signal_fc1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)    # FC1的Z信号
        self.z_signal_fc2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)    # FC2的Z信号
        
        # ========== HyperNetwork生成器 ==========
        # 每个HyperNetwork是一个简单的线性变换：Z信号 -> 权重
        
        # Conv1 HyperNetwork: 生成 3->32 的5x5卷积权重
        conv1_weight_size = in_size * 3 * f_size * f_size  # 32 * 3 * 5 * 5 = 2,400
        self.hyper_conv1 = nn.Linear(z_dim, conv1_weight_size)
        self.conv1_bias = nn.Parameter(torch.zeros(in_size))  # Conv1偏置（独立学习）
        
        # Conv2 HyperNetwork: 生成 32->64 的5x5卷积权重
        conv2_weight_size = out_size * in_size * f_size * f_size  # 64 * 32 * 5 * 5 = 51,200
        self.hyper_conv2 = nn.Linear(z_dim, conv2_weight_size)
        self.conv2_bias = nn.Parameter(torch.zeros(out_size))  # Conv2偏置
        
        # Conv3 HyperNetwork: 生成 64->128 的3x3卷积权重
        conv3_weight_size = 128 * out_size * 3 * 3  # 128 * 64 * 3 * 3 = 73,728
        self.hyper_conv3 = nn.Linear(z_dim, conv3_weight_size)
        self.conv3_bias = nn.Parameter(torch.zeros(128))  # Conv3偏置
        
        # FC1 HyperNetwork: 生成 2048->256 的全连接权重
        # 输入尺寸计算：经过3次2x2池化后，32x32 -> 4x4，128通道，所以是128*4*4=2048
        fc1_input_size = 128 * 4 * 4  # 2048
        fc1_weight_size = 256 * fc1_input_size  # 256 * 2048 = 524,288
        self.hyper_fc1 = nn.Linear(z_dim, fc1_weight_size)
        self.fc1_bias = nn.Parameter(torch.zeros(256))  # FC1偏置
        
        # FC2 HyperNetwork: 生成 256->100 的输出层权重
        fc2_weight_size = 100 * 256  # 100 * 256 = 25,600
        self.hyper_fc2 = nn.Linear(z_dim, fc2_weight_size)
        self.fc2_bias = nn.Parameter(torch.zeros(100))  # FC2偏置（100个CIFAR-100类别）
        
        # 初始化所有HyperNetwork组件
        self._init_hypernetworks()
        
        # 打印架构信息
        print(f"ALL LAYERS HyperNetwork Architecture:")
        print(f"  Conv1: z_dim={z_dim} -> {conv1_weight_size} params (32x3x5x5)")
        print(f"  Conv2: z_dim={z_dim} -> {conv2_weight_size} params (64x32x5x5)")
        print(f"  Conv3: z_dim={z_dim} -> {conv3_weight_size} params (128x64x3x3)")
        print(f"  FC1: z_dim={z_dim} -> {fc1_weight_size} params (256x2048)")
        print(f"  FC2: z_dim={z_dim} -> {fc2_weight_size} params (100x256)")
        
    def _init_hypernetworks(self):
        """
        初始化所有HyperNetwork组件
        
        初始化策略：
        - 权重：小的正态分布（std=0.01），避免初期权重过大
        - 偏置：零初始化，标准做法
        """
        hypernetworks = [
            self.hyper_conv1, self.hyper_conv2, self.hyper_conv3,
            self.hyper_fc1, self.hyper_fc2
        ]
        
        for hyper_net in hypernetworks:
            # 权重：小的正态分布初始化
            nn.init.normal_(hyper_net.weight, std=0.01)
            # 偏置：零初始化
            nn.init.constant_(hyper_net.bias, 0.0)
    
    def generate_conv1_weights(self):
        """
        生成Conv1层权重
        
        过程：
        1. Z信号(1x8) -> HyperNetwork -> 扁平权重(1x2400)
        2. 重塑为卷积权重形状(32, 3, 5, 5)
        
        返回：Conv1权重张量，形状为(out_channels, in_channels, h, w)
        """
        weights_flat = self.hyper_conv1(self.z_signal_conv1)  # (1, 2400)
        return weights_flat.reshape(self.in_size, 3, self.f_size, self.f_size)  # (32, 3, 5, 5)
    
    def generate_conv2_weights(self):
        """
        生成Conv2层权重
        
        过程：Z信号(1x8) -> 扁平权重(1x51200) -> 重塑(64, 32, 5, 5)
        """
        weights_flat = self.hyper_conv2(self.z_signal_conv2)
        return weights_flat.reshape(self.out_size, self.in_size, self.f_size, self.f_size)
    
    def generate_conv3_weights(self):
        """
        生成Conv3层权重
        
        过程：Z信号(1x8) -> 扁平权重(1x73728) -> 重塑(128, 64, 3, 3)
        注意：Conv3使用3x3卷积核
        """
        weights_flat = self.hyper_conv3(self.z_signal_conv3)
        return weights_flat.reshape(128, self.out_size, 3, 3)
    
    def generate_fc1_weights(self):
        """
        生成FC1层权重
        
        过程：Z信号(1x8) -> 扁平权重(1x524288) -> 重塑(256, 2048)
        """
        weights_flat = self.hyper_fc1(self.z_signal_fc1)
        return weights_flat.reshape(256, 128 * 4 * 4)
    
    def generate_fc2_weights(self):
        """
        生成FC2层权重（输出层）
        
        过程：Z信号(1x8) -> 扁平权重(1x25600) -> 重塑(100, 256)
        """
        weights_flat = self.hyper_fc2(self.z_signal_fc2)
        return weights_flat.reshape(100, 256)
        
    def forward(self, x):
        """
        前向传播过程
        
        网络结构：
        输入(3,32,32) -> Conv1+Pool -> Conv2+Pool -> Conv3+Pool -> FC1+Dropout -> FC2 -> 输出(100)
        
        特点：每层权重都是实时生成的，不是预存储的
        """
        
        # ========== Conv1块：3->32通道，32x32->16x16 ==========
        conv1_weights = self.generate_conv1_weights()  # 实时生成Conv1权重
        x = F.conv2d(x, conv1_weights, bias=self.conv1_bias, padding='same')  # 卷积操作
        x = self.pool(F.relu(x))  # ReLU激活 + 2x2池化，尺寸从32x32变为16x16
        
        # ========== Conv2块：32->64通道，16x16->8x8 ==========
        conv2_weights = self.generate_conv2_weights()  # 实时生成Conv2权重
        x = F.conv2d(x, conv2_weights, bias=self.conv2_bias, padding='same')
        x = self.pool(F.relu(x))  # 尺寸从16x16变为8x8
        
        # ========== Conv3块：64->128通道，8x8->4x4 ==========
        conv3_weights = self.generate_conv3_weights()  # 实时生成Conv3权重
        x = F.conv2d(x, conv3_weights, bias=self.conv3_bias, padding='same')
        x = self.pool(F.relu(x))  # 尺寸从8x8变为4x4
        
        # ========== 扁平化：从(batch, 128, 4, 4)到(batch, 2048) ==========
        x = x.view(x.size(0), -1)  # 展平为1D向量，128*4*4=2048维
        
        # ========== FC1块：2048->256 ==========
        fc1_weights = self.generate_fc1_weights()  # 实时生成FC1权重
        x = F.linear(x, fc1_weights, bias=self.fc1_bias)  # 线性变换
        x = F.relu(x)  # ReLU激活
        x = self.dropout(x)  # Dropout正则化，训练时随机置零50%的神经元
        
        # ========== FC2块：256->100（输出层） ==========
        fc2_weights = self.generate_fc2_weights()  # 实时生成FC2权重
        x = F.linear(x, fc2_weights, bias=self.fc2_bias)  # 最终线性变换到100个类别
        
        return x  # 返回未经softmax的logits

def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=50):
    """
    单个epoch的训练过程
    
    参数：
    - model: 要训练的模型
    - device: 计算设备(CPU/GPU)
    - train_loader: 训练数据加载器
    - optimizer: 优化器
    - epoch: 当前epoch编号
    - log_interval: 打印间隔
    
    返回：
    - train_loss: 平均训练损失
    - train_err: 训练错误率
    """
    model.train()  # 设置为训练模式，启用dropout和batch norm的训练行为
    train_loss = 0  # 累计损失
    correct = 0     # 正确预测数量
    total = 0       # 总样本数量
    
    # 遍历训练数据批次
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移动到指定设备，非阻塞传输提高效率
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # ========== 标准训练步骤 ==========
        optimizer.zero_grad()  # 清零梯度
        output = model(data)   # 前向传播，注意这里会实时生成所有层的权重
        loss = F.cross_entropy(output, target)  # 计算交叉熵损失
        loss.backward()        # 反向传播，计算梯度
        
        # 梯度裁剪：防止梯度爆炸，对HyperNetwork很重要
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()       # 更新参数（包括Z信号和HyperNetwork权重）
        
        # ========== 指标跟踪 ==========
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # 获取预测类别
        correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确预测
        total += target.size(0)
        
        # 定期打印训练进度和Z信号状态
        if batch_idx % log_interval == 0:
            current_acc = 100. * correct / total
            print(f'  Batch: {batch_idx:3d}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {current_acc:6.2f}%')
            
            # ========== Z信号监控 ==========
            # 监控Z信号的范数，检查是否过大或过小
            conv_norms = [
                model.z_signal_conv1.norm().item(),  # Conv1 Z信号的L2范数
                model.z_signal_conv2.norm().item(),  # Conv2 Z信号的L2范数
                model.z_signal_conv3.norm().item()   # Conv3 Z信号的L2范数
            ]
            fc_norms = [
                model.z_signal_fc1.norm().item(),    # FC1 Z信号的L2范数
                model.z_signal_fc2.norm().item()     # FC2 Z信号的L2范数
            ]
            print(f'    Conv Z-norms: {conv_norms[0]:.4f}, {conv_norms[1]:.4f}, {conv_norms[2]:.4f}')
            print(f'    FC Z-norms: {fc_norms[0]:.4f}, {fc_norms[1]:.4f}')
    
    # 计算epoch平均指标
    train_loss /= len(train_loader)
    train_acc = correct / total
    train_err = 1.0 - train_acc
    return train_loss, train_err

def evaluate(model, device, data_loader):
    """
    模型评估函数
    
    用于验证集和测试集评估，不更新模型参数
    
    返回：
    - loss: 平均损失
    - err: 错误率
    """
    model.eval()  # 设置为评估模式，禁用dropout
    loss = 0
    correct = 0
    total = 0
    
    # 禁用梯度计算，节省内存和计算
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)  # 前向传播，权重仍然是实时生成的
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    loss /= total
    acc = correct / total
    err = 1.0 - acc
    return loss, err

def count_parameters(model):
    """
    统计和分类模型参数
    
    将参数分为两类：
    1. HyperNetwork相关：Z信号 + HyperNetwork权重和偏置
    2. 标准参数：卷积和全连接的偏置参数
    
    返回：
    - total_params: 总参数数量
    - hyper_params: HyperNetwork参数数量
    """
    total_params = 0
    hyper_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只统计可训练参数
            num_params = param.numel()  # 参数数量
            total_params += num_params
            
            # 分类参数：HyperNetwork相关 vs 标准参数
            if 'z_signal' in name or 'hyper_' in name:
                hyper_params += num_params
                print(f"[HYPER] {name}: {param.shape}, {num_params:,}")
            else:
                print(f"[OTHER] {name}: {param.shape}, {num_params:,}")
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Hypernetwork parameters: {hyper_params:,}")
    print(f"Standard parameters: {total_params - hyper_params:,}")
    print(f"Hypernetwork ratio: {100.*hyper_params/total_params:.2f}%")
    return total_params, hyper_params

def main():
    """
    主训练函数
    
    完整的训练流程：
    1. 设备和随机种子设置
    2. 数据加载和预处理
    3. 模型创建和初始化
    4. 训练循环
    5. 最终评估和结果保存
    """
    print("Starting CIFAR-100 ALL LAYERS HyperNetwork...")
    print("ALL conv and fc layers generated by hypernetworks!")
    print("="*60)
    
    # ========== 可重复性设置 ==========
    torch.manual_seed(42)     # PyTorch随机种子
    np.random.seed(42)        # NumPy随机种子
    
    device = set_device()     # 自动选择最佳计算设备
    
    # ========== 超参数配置 ==========
    batch_size = 128          # 训练批次大小
    test_batch_size = 256     # 测试批次大小（更大，因为不需要反向传播）
    epochs = 50               # 训练轮数
    lr = 0.001                # 初始学习率
    f_size = 5                # 卷积核大小（Conv1,Conv2）
    in_size = 32              # Conv1输出通道数
    out_size = 64             # Conv2输出通道数
    z_dim = 8                 # Z信号维度
    
    print(f"Hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Filter size: {f_size}")
    print(f"  Z dimension: {z_dim}")
    print("="*60)
    
    # ========== 数据预处理 ==========
    # 训练集增强：提高泛化能力
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
        transforms.ToTensor(),                     # 转换为张量，范围[0,1]
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100标准化
    ])
    
    # 测试集：只做标准化，不做增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # ========== 数据集加载 ==========
    print("Loading CIFAR-100 dataset...")
    train_dataset = datasets.CIFAR100('data', train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR100('data', train=False, transform=transform_test)
    
    # 从训练集中分出验证集
    train_size = len(train_dataset) - 5000  # 45000个训练样本
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, 5000]    # 5000个验证样本
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True  # 2个工作进程，内存固定加速GPU传输
    )
    val_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, 
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, 
        num_workers=2, pin_memory=True
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print("="*60)
    
    # ========== 模型创建 ==========
    print("Training ALL LAYERS HyperNetwork CNN...")
    model = HyperCNNCIFAR_AllLayers(
        f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim
    ).to(device)  # 移动到计算设备
    
    # ========== 优化器和调度器 ==========
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)  # 指数衰减学习率
    
    # 打印模型参数统计
    print("\nALL LAYERS HyperNetwork CNN parameters:")
    count_parameters(model)
    print("="*60)
    
    # ========== 训练循环 ==========
    best_val_err = 1.0  # 最佳验证错误率
    for epoch in range(epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs} - ALL LAYERS HyperNetwork CNN")
        print("-" * 50)
        
        # 训练一个epoch
        train_loss, train_err = train_epoch(model, device, train_loader, optimizer, epoch)
        
        # 验证集评估
        val_loss, val_err = evaluate(model, device, val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_err < best_val_err:
            best_val_err = val_err
            torch.save(model.state_dict(), 'cifar100_all_layers_hyper_best.pt')
            print(f"    *** New best validation error: {100*val_err:.2f}% - Model saved! ***")
        
        # 打印epoch结果
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nResults: Train Err: {100*train_err:.2f}% | Val Err: {100*val_err:.2f}% | "
              f"Best Val Err: {100*best_val_err:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
    
    # ========== 最终评估 ==========
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('cifar100_all_layers_hyper_best.pt'))
    test_loss, test_err = evaluate(model, device, test_loader)
    test_acc = (1 - test_err) * 100
    
    # ========== 性能对比 ==========
    # 与之前实验结果对比
    std_test_acc = 16.23       # 标准CNN结果
    fixed_hyper_acc = 33.52    # 单层HyperNetwork结果
    full_hyper_acc = 54.16     # 复杂全量HyperNetwork结果
    
    print(f"Standard CNN - Test Accuracy: {std_test_acc:.2f}% (previous result)")
    print(f"Fixed HyperNetwork (1 layer) - Test Accuracy: {fixed_hyper_acc:.2f}% (previous result)")
    print(f"Full HyperNetwork (all layers) - Test Accuracy: {full_hyper_acc:.2f}% (previous result)")
    print(f"ALL LAYERS HyperNetwork - Test Accuracy: {test_acc:.2f}%")
    
    # 计算改进幅度
    improvement_vs_std = test_acc - std_test_acc
    improvement_vs_fixed = test_acc - fixed_hyper_acc
    improvement_vs_full = test_acc - full_hyper_acc
    
    print(f"Improvement vs Standard CNN: {improvement_vs_std:+.2f}%")
    print(f"Improvement vs Fixed HyperNet: {improvement_vs_fixed:+.2f}%")
    print(f"Improvement vs Full HyperNet: {improvement_vs_full:+.2f}%")
    
    # ========== Z信号最终状态 ==========
    # 打印训练后的Z信号，观察学习到的模式
    print(f"\nFinal hypernetwork signals:")
    print(f"z_signal_conv1: {model.z_signal_conv1.detach().cpu().numpy().flatten()}")
    print(f"z_signal_conv2: {model.z_signal_conv2.detach().cpu().numpy().flatten()}")
    print(f"z_signal_conv3: {model.z_signal_conv3.detach().cpu().numpy().flatten()}")
    print(f"z_signal_fc1: {model.z_signal_fc1.detach().cpu().numpy().flatten()}")
    print(f"z_signal_fc2: {model.z_signal_fc2.detach().cpu().numpy().flatten()}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
