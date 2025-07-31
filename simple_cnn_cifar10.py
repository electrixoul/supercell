"""
Simple Small-Scale CNN for CIFAR-10 Classification
将simple_mlp_cifar10.py中的MLP替换为小规模CNN进行对比实验

Architecture:
- Conv1: 3→32, 3×3, padding=1
- MaxPool: 2×2  
- Conv2: 32→64, 3×3, padding=1
- MaxPool: 2×2
- Conv3: 64→128, 3×3, padding=1
- MaxPool: 2×2
- FC1: 128×4×4→128
- FC2: 128→10

Hardware: MacBook Pro M4 Pro with MPS acceleration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def set_device():
    """设置M4 Pro的最优设备"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用MPS (Metal Performance Shaders) 加速 - M4 Pro")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ 使用CUDA加速")
    else:
        device = torch.device("cpu")
        print("⚠️  使用CPU")
    return device

class SimpleCNN(nn.Module):
    """小规模CNN网络"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层序列
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32x32x32
        self.pool1 = nn.MaxPool2d(2, 2)                          # 16x16x32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 16x16x64
        self.pool2 = nn.MaxPool2d(2, 2)                          # 8x8x64
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 8x8x128
        self.pool3 = nn.MaxPool2d(2, 2)                           # 4x4x128
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout防止过拟合  
        self.dropout = nn.Dropout(0.5)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # 第二个卷积块
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 第三个卷积块
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_epoch(model, device, train_loader, optimizer, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 打印进度
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):3.0f}%)] '
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def test_epoch(model, device, test_loader):
    """测试一个epoch"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    accuracy = 100. * correct / total
    
    return test_loss, accuracy

def plot_learning_curves(train_losses, train_accs, test_losses, test_accs, save_path='cnn_learning_curves.png'):
    """绘制学习曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='测试损失', linewidth=2)
    ax1.set_title('CNN损失曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='测试准确率', linewidth=2)
    ax2.set_title('CNN准确率曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 学习曲线已保存至: {save_path}")

def count_parameters(model):
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*50)
    print("CNN模型参数统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 详细分层统计
    print("\n各层参数详情：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape} -> {param.numel():,} 参数")
    
    print("="*50)
    
    return total_params, trainable_params

def visualize_feature_maps(model, device, sample_data, save_path='cnn_feature_maps.png'):
    """可视化卷积特征图"""
    model.eval()
    
    # 获取第一个样本
    if len(sample_data.shape) == 3:
        sample_data = sample_data.unsqueeze(0)
    
    sample_data = sample_data.to(device)
    
    # 定义hook函数来获取中间层输出
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册hook
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.conv3.register_forward_hook(get_activation('conv3'))
    
    # 前向传播
    with torch.no_grad():
        _ = model(sample_data)
    
    # 绘制特征图
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    fig.suptitle('CNN卷积特征图可视化', fontsize=16, fontweight='bold')
    
    layers = ['conv1', 'conv2', 'conv3']
    for i, layer_name in enumerate(layers):
        feature_maps = activations[layer_name][0]  # 取第一个样本
        
        for j in range(8):  # 显示前8个通道
            if j < feature_maps.shape[0]:
                feature_map = feature_maps[j].cpu().numpy()
                axes[i, j].imshow(feature_map, cmap='viridis')
                axes[i, j].set_title(f'{layer_name}[{j}]', fontsize=8)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 特征图可视化已保存至: {save_path}")

def main():
    print("🚀 开始训练小规模CNN在CIFAR-10数据集上")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = set_device()
    
    # 超参数
    batch_size = 128
    test_batch_size = 256
    epochs = 25  # CNN通常需要更多轮次
    learning_rate = 0.001
    
    print(f"📋 超参数设置:")
    print(f"  批量大小: {batch_size}")
    print(f"  测试批量: {test_batch_size}")
    print(f"  训练轮次: {epochs}")
    print(f"  学习率: {learning_rate}")
    print("="*60)
    
    # 数据预处理 - CNN版本使用更强的数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载并加载CIFAR-10数据集
    print("📥 下载CIFAR-10数据集...")
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                   download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                           shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"📊 数据集信息:")
    print(f"  训练样本: {len(train_dataset):,}")
    print(f"  测试样本: {len(test_dataset):,}")
    print(f"  类别数量: 10")
    print("="*60)
    
    # CIFAR-10类别名称
    classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')
    print(f"🏷️  类别: {', '.join(classes)}")
    print("="*60)
    
    # 创建模型
    model = SimpleCNN(num_classes=10).to(device)
    
    # 统计参数
    count_parameters(model)
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # 训练记录
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    best_test_acc = 0.
    
    # 开始训练
    print("🔥 开始训练...")
    total_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)
        
        # 测试
        test_loss, test_acc = test_epoch(model, device, test_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_cnn_cifar10.pt')
        
        epoch_time = time.time() - epoch_start_time
        
        # 打印epoch结果
        print(f'Epoch {epoch:2d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | '
              f'Best: {best_test_acc:6.2f}% | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.1f}s')
        print("-" * 80)
    
    total_time = time.time() - total_start_time
    
    # 训练完成总结
    print("="*60)
    print("🎉 训练完成!")
    print(f"📈 最佳测试准确率: {best_test_acc:.2f}%")
    print(f"⏱️  总训练时间: {total_time:.1f}s ({total_time/60:.1f}分钟)")
    print(f"📁 最佳模型已保存至: best_cnn_cifar10.pt")
    print("="*60)
    
    # 绘制学习曲线
    plot_learning_curves(train_losses, train_accs, test_losses, test_accs)
    
    # 特征图可视化
    sample_data, _ = next(iter(test_loader))
    visualize_feature_maps(model, device, sample_data[0])
    
    # 最终结果摘要
    print("\n📊 最终结果摘要:")
    print(f"  模型架构: 小规模CNN (3→32→64→128→FC)")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  最终训练准确率: {train_accs[-1]:.2f}%")
    print(f"  最终测试准确率: {test_accs[-1]:.2f}%")
    print(f"  最佳测试准确率: {best_test_acc:.2f}%")
    print(f"  训练轮次: {epochs}")
    print(f"  使用设备: {device}")
    
    # 与MLP对比分析
    print("\n🔍 与MLP对比分析:")
    print("  优势:")
    print("    • 利用了空间局部性 - 卷积操作保留了图像的空间结构")
    print("    • 平移不变性 - 对图像中对象的位置变化更鲁棒")
    print("    • 参数共享 - 相同的卷积核在整个图像上共享权重")
    print("    • 分层特征提取 - 从低级边缘到高级语义特征")
    print("  预期表现:")
    print("    • 收敛速度可能更快")
    print("    • 最终准确率应该显著高于MLP的52.01%")
    print("    • 对图像数据更适配的归纳偏置")

if __name__ == "__main__":
    main()
