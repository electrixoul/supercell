"""
Simple 3-Layer MLP for CIFAR-10 Classification
使用标准BP算法在CIFAR-10数据集上训练简单的多层感知机

Architecture:
- Input: 32x32x3 = 3072 features (flattened)  
- Hidden: 128 neurons with ReLU activation
- Output: 10 classes (softmax)

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

class SimpleMLP(nn.Module):
    """简单的三层MLP"""
    def __init__(self, input_size=3072, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        
        # 第一层：输入层到隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # 第二层：隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """标准权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 展平输入 (batch_size, 3, 32, 32) -> (batch_size, 3072)
        x = x.view(x.size(0), -1)
        
        # 第一层 + ReLU激活
        x = F.relu(self.fc1(x))
        
        # 第二层（输出层）
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
        
        # 反向传播 (BP算法)
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

def plot_learning_curves(train_losses, train_accs, test_losses, test_accs, save_path='learning_curves.png'):
    """绘制学习曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='测试损失', linewidth=2)
    ax1.set_title('损失曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='测试准确率', linewidth=2)
    ax2.set_title('准确率曲线', fontsize=14, fontweight='bold')
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
    print("模型参数统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print("="*50)
    
    return total_params, trainable_params

def main():
    print("🚀 开始训练简单MLP在CIFAR-10数据集上")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = set_device()
    
    # 超参数
    batch_size = 128
    test_batch_size = 256
    epochs = 20
    learning_rate = 0.001
    
    print(f"📋 超参数设置:")
    print(f"  批量大小: {batch_size}")
    print(f"  测试批量: {test_batch_size}")
    print(f"  训练轮次: {epochs}")
    print(f"  学习率: {learning_rate}")
    print("="*60)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])
    
    # 下载并加载CIFAR-10数据集
    print("📥 下载CIFAR-10数据集...")
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=transform)
    
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
    model = SimpleMLP(input_size=32*32*3, hidden_size=128, num_classes=10).to(device)
    
    # 统计参数
    count_parameters(model)
    
    # 优化器 (标准SGD或Adam都可以，这里用Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
        
        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_mlp_cifar10.pt')
        
        epoch_time = time.time() - epoch_start_time
        
        # 打印epoch结果
        print(f'Epoch {epoch:2d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | '
              f'Best: {best_test_acc:6.2f}% | Time: {epoch_time:.1f}s')
        print("-" * 80)
    
    total_time = time.time() - total_start_time
    
    # 训练完成总结
    print("="*60)
    print("🎉 训练完成!")
    print(f"📈 最佳测试准确率: {best_test_acc:.2f}%")
    print(f"⏱️  总训练时间: {total_time:.1f}s ({total_time/60:.1f}分钟)")
    print(f"📁 最佳模型已保存至: best_mlp_cifar10.pt")
    print("="*60)
    
    # 绘制学习曲线
    plot_learning_curves(train_losses, train_accs, test_losses, test_accs)
    
    # 最终结果摘要
    print("\n📊 最终结果摘要:")
    print(f"  模型架构: 3层MLP (3072→128→10)")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  最终训练准确率: {train_accs[-1]:.2f}%")
    print(f"  最终测试准确率: {test_accs[-1]:.2f}%")
    print(f"  最佳测试准确率: {best_test_acc:.2f}%")
    print(f"  训练轮次: {epochs}")
    print(f"  使用设备: {device}")

if __name__ == "__main__":
    main()
