"""
Simple 3-Layer MLP for CIFAR-10 Classification with Enhanced Data Augmentation
使用增强数据增强技术的MLP版本，与CNN版本使用相同的数据预处理策略

Architecture:
- Input: 32x32x3 = 3072 features (flattened)  
- Hidden: 128 neurons with ReLU activation
- Output: 10 classes (softmax)

Data Augmentation (从CNN版本复制):
- RandomHorizontalFlip(p=0.5)
- RandomCrop(32, padding=4)
- ToTensor()
- Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

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
    """简单的三层MLP - 与原版相同的架构"""
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

def plot_learning_curves(train_losses, train_accs, test_losses, test_accs, save_path='mlp_enhanced_learning_curves.png'):
    """绘制学习曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='测试损失', linewidth=2)
    ax1.set_title('MLP增强版损失曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='测试准确率', linewidth=2)
    ax2.set_title('MLP增强版准确率曲线', fontsize=14, fontweight='bold')
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
    print("MLP增强版模型参数统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 详细分层统计
    print("\n各层参数详情：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape} -> {param.numel():,} 参数")
    
    print("="*50)
    
    return total_params, trainable_params

def main():
    print("🚀 开始训练增强数据增强的MLP在CIFAR-10数据集上")
    print("="*60)
    print("📢 本版本使用与CNN相同的数据增强策略!")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = set_device()
    
    # 超参数 - 与原MLP版本保持一致
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
    
    # 📌 关键改进：使用与CNN版本相同的增强数据增强
    print("🔧 数据增强策略 (从CNN版本复制):")
    print("  • RandomHorizontalFlip(p=0.5) - 随机水平翻转")
    print("  • RandomCrop(32, padding=4) - 随机裁剪")
    print("  • ToTensor() - 转换为张量")
    print("  • Normalize(mean, std) - 标准化")
    print("="*60)
    
    # 数据预处理 - 从CNN版本复制的增强数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),        # 50%概率水平翻转
        transforms.RandomCrop(32, padding=4),          # 随机裁剪，先padding再裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])
    
    # 测试集保持简单预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载并加载CIFAR-10数据集
    print("📥 下载CIFAR-10数据集...")
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                   download=True, transform=train_transform)  # 使用增强变换
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=test_transform)   # 测试集使用简单变换
    
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
    
    # 创建模型 - 与原版MLP完全相同的架构
    model = SimpleMLP(input_size=32*32*3, hidden_size=128, num_classes=10).to(device)
    
    # 统计参数
    count_parameters(model)
    
    # 优化器 - 保持与原版一致
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
            torch.save(model.state_dict(), 'best_mlp_enhanced_cifar10.pt')
        
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
    print(f"📁 最佳模型已保存至: best_mlp_enhanced_cifar10.pt")
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
    
    # 数据增强效果分析
    print("\n🔍 数据增强效果分析:")
    print("  预期改进:")
    print("    • 随机水平翻转增加了数据的多样性")
    print("    • 随机裁剪提供了位置变化的鲁棒性")
    print("    • 应该能提升相对于原始MLP(52.01%)的性能")
    print("  理论分析:")
    print("    • MLP虽然丢失了空间结构信息，但数据增强仍能提供正则化效果")
    print("    • 增强的数据分布可能帮助MLP学习更鲁棒的特征表示")
    print("    • 预期性能提升：2-5个百分点")
    
    print(f"\n💡 与原始MLP对比:")
    print(f"  原始MLP最佳准确率: 52.01%")
    print(f"  增强MLP最佳准确率: {best_test_acc:.2f}%")
    if best_test_acc > 52.01:
        improvement = best_test_acc - 52.01
        print(f"  性能提升: +{improvement:.2f}%")
        print(f"  相对提升: {improvement/52.01*100:.1f}%")
    else:
        decline = 52.01 - best_test_acc
        print(f"  性能下降: -{decline:.2f}%")

if __name__ == "__main__":
    main()
