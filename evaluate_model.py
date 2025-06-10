"""
CIFAR-100 HyperNetwork Progressive Stable Model 评估程序
简单直接地测试训练好的模型在CIFAR-100测试集上的性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import argparse

class HyperCNNCIFAR_AllLayers_Progressive(nn.Module):
    """与训练时完全相同的模型架构"""
    
    def __init__(self, f_size=5, in_size=64, out_size=128, z_dim=10):
        super(HyperCNNCIFAR_AllLayers_Progressive, self).__init__()
        
        self.f_size = f_size
        self.in_size = in_size      
        self.out_size = out_size    
        self.z_dim = z_dim          
        
        # 固定组件
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)   
        self.dropout2 = nn.Dropout(0.2)   
        self.dropout3 = nn.Dropout(0.3)   
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(in_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn_fc1 = nn.BatchNorm1d(512)
        
        # Z信号参数
        self.z_signal_conv1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_conv2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_conv3 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_fc1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_fc2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        
        # HyperNetwork生成器
        conv1_weight_size = in_size * 3 * f_size * f_size
        self.hyper_conv1 = nn.Linear(z_dim, conv1_weight_size)
        self.conv1_bias = nn.Parameter(torch.zeros(in_size))
        
        conv2_weight_size = out_size * in_size * f_size * f_size
        self.hyper_conv2 = nn.Linear(z_dim, conv2_weight_size)
        self.conv2_bias = nn.Parameter(torch.zeros(out_size))
        
        conv3_weight_size = 256 * out_size * 3 * 3
        self.hyper_conv3 = nn.Linear(z_dim, conv3_weight_size)
        self.conv3_bias = nn.Parameter(torch.zeros(256))
        
        fc1_input_size = 256 * 4 * 4
        fc1_weight_size = 512 * fc1_input_size
        self.hyper_fc1 = nn.Linear(z_dim, fc1_weight_size)
        self.fc1_bias = nn.Parameter(torch.zeros(512))
        
        fc2_weight_size = 100 * 512
        self.hyper_fc2 = nn.Linear(z_dim, fc2_weight_size)
        self.fc2_bias = nn.Parameter(torch.zeros(100))
        
    def generate_conv1_weights(self):
        weights_flat = self.hyper_conv1(self.z_signal_conv1)
        return weights_flat.reshape(self.in_size, 3, self.f_size, self.f_size)
    
    def generate_conv2_weights(self):
        weights_flat = self.hyper_conv2(self.z_signal_conv2)
        return weights_flat.reshape(self.out_size, self.in_size, self.f_size, self.f_size)
    
    def generate_conv3_weights(self):
        weights_flat = self.hyper_conv3(self.z_signal_conv3)
        return weights_flat.reshape(256, self.out_size, 3, 3)
    
    def generate_fc1_weights(self):
        weights_flat = self.hyper_fc1(self.z_signal_fc1)
        return weights_flat.reshape(512, 256 * 4 * 4)
    
    def generate_fc2_weights(self):
        weights_flat = self.hyper_fc2(self.z_signal_fc2)
        return weights_flat.reshape(100, 512)
        
    def forward(self, x):
        # Conv1块
        conv1_weights = self.generate_conv1_weights()
        x = F.conv2d(x, conv1_weights, bias=self.conv1_bias, padding='same')
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Conv2块
        conv2_weights = self.generate_conv2_weights()
        x = F.conv2d(x, conv2_weights, bias=self.conv2_bias, padding='same')
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Conv3块
        conv3_weights = self.generate_conv3_weights()
        x = F.conv2d(x, conv3_weights, bias=self.conv3_bias, padding='same')
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        
        # 扁平化
        x = x.view(x.size(0), -1)
        
        # FC1块
        fc1_weights = self.generate_fc1_weights()
        x = F.linear(x, fc1_weights, bias=self.fc1_bias)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # FC2块
        fc2_weights = self.generate_fc2_weights()
        x = self.dropout3(x)
        x = F.linear(x, fc2_weights, bias=self.fc2_bias)
        
        return x

def evaluate_model(model, device, test_loader, verbose=True):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))
    
    if verbose:
        print("开始评估模型...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 统计每个类别的准确率
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
            if verbose and batch_idx % 20 == 0:
                current_acc = 100. * correct / total
                print(f'  Batch {batch_idx:3d}/{len(test_loader)} | Accuracy: {current_acc:.2f}%')
    
    overall_accuracy = 100. * correct / total
    
    if verbose:
        print(f'\n整体测试准确率: {overall_accuracy:.2f}% ({correct}/{total})')
        
        # 找出表现最好和最差的类别
        class_accuracies = []
        for i in range(100):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                class_accuracies.append((i, acc))
        
        class_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        print(f'\n表现最好的5个类别:')
        for i, (class_idx, acc) in enumerate(class_accuracies[:5]):
            print(f'  {i+1}. 类别 {class_idx}: {acc:.2f}%')
            
        print(f'\n表现最差的5个类别:')
        for i, (class_idx, acc) in enumerate(class_accuracies[-5:]):
            print(f'  {i+1}. 类别 {class_idx}: {acc:.2f}%')
    
    return overall_accuracy, class_accuracies

def main():
    parser = argparse.ArgumentParser(description='评估 Progressive Stable HyperNetwork 模型')
    parser.add_argument('--model_path', type=str, 
                        default='cifar100_all_layers_hyper_progressive_stable_local_best.pt',
                        help='模型权重文件路径')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='测试批次大小')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据预处理（与训练时的测试集预处理一致）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 加载CIFAR-100测试集
    print("加载CIFAR-100测试集...")
    test_dataset = datasets.CIFAR100('data', train=False, download=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f'测试集大小: {len(test_dataset)} 样本')
    
    # 创建模型
    print("创建模型...")
    model = HyperCNNCIFAR_AllLayers_Progressive(
        f_size=5, in_size=64, out_size=128, z_dim=10
    ).to(device)
    
    # 加载训练好的权重
    print(f"加载模型权重: {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print("✅ 模型权重加载成功!")
    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        return
    
    # 评估模型
    print("="*60)
    print("开始模型评估")
    print("="*60)
    
    overall_accuracy, class_accuracies = evaluate_model(model, device, test_loader, verbose=True)
    
    print("="*60)
    print("评估完成!")
    print(f"🎯 最终测试准确率: {overall_accuracy:.2f}%")
    print("="*60)
    
    # 与之前版本的性能对比
    print("\n📊 与其他版本性能对比:")
    comparisons = [
        ("Standard CNN", 16.23),
        ("Fixed HyperNetwork (1 layer)", 33.52),
        ("ALL LAYERS HyperNetwork (original)", 36.13),
        ("ALL LAYERS HyperNetwork (Multi-GPU)", 42.76),
        ("ALL LAYERS HyperNetwork (Large Multi-GPU)", 53.71),
        ("ALL LAYERS HyperNetwork (Data Enhancement)", 51.95),
        ("ALL LAYERS HyperNetwork (Progressive Stable)", overall_accuracy),
    ]
    
    for name, acc in comparisons:
        if "Progressive Stable" in name:
            print(f"  🔸 {name}: {acc:.2f}% ⭐")
        else:
            print(f"  • {name}: {acc:.2f}%")
    
    # 计算改进
    best_previous = 53.71  # Large Multi-GPU version
    improvement = overall_accuracy - best_previous
    print(f"\n📈 相比最佳先前版本的改进: {improvement:+.2f}%")

if __name__ == "__main__":
    main()
