'''
CIFAR-100 HyperNetwork ALL LAYERS VERSION - Large Batch Size for H100
=====================================================================
基于cifar100_hypernetwork_fixed_all_layers.py，针对H100优化的大批次版本
- 增大batch size充分利用GPU内存和计算能力
- 调整学习率适应大批次训练
- 优化数据加载以提高GPU利用率
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
    """设备检测和配置函数，针对H100优化"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

class HyperCNNCIFAR_AllLayers(nn.Module):
    """全层HyperNetwork CIFAR-100分类器（与原版相同）"""
    
    def __init__(self, f_size=5, in_size=32, out_size=64, z_dim=8):
        super(HyperCNNCIFAR_AllLayers, self).__init__()
        
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.z_dim = z_dim
        
        # 固定组件
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
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
        
        conv3_weight_size = 128 * out_size * 3 * 3
        self.hyper_conv3 = nn.Linear(z_dim, conv3_weight_size)
        self.conv3_bias = nn.Parameter(torch.zeros(128))
        
        fc1_input_size = 128 * 4 * 4
        fc1_weight_size = 256 * fc1_input_size
        self.hyper_fc1 = nn.Linear(z_dim, fc1_weight_size)
        self.fc1_bias = nn.Parameter(torch.zeros(256))
        
        fc2_weight_size = 100 * 256
        self.hyper_fc2 = nn.Linear(z_dim, fc2_weight_size)
        self.fc2_bias = nn.Parameter(torch.zeros(100))
        
        self._init_hypernetworks()
        
        print(f"ALL LAYERS HyperNetwork Architecture (Large Batch):")
        print(f"  Conv1: z_dim={z_dim} -> {conv1_weight_size} params")
        print(f"  Conv2: z_dim={z_dim} -> {conv2_weight_size} params")
        print(f"  Conv3: z_dim={z_dim} -> {conv3_weight_size} params")
        print(f"  FC1: z_dim={z_dim} -> {fc1_weight_size} params")
        print(f"  FC2: z_dim={z_dim} -> {fc2_weight_size} params")
        
    def _init_hypernetworks(self):
        hypernetworks = [
            self.hyper_conv1, self.hyper_conv2, self.hyper_conv3,
            self.hyper_fc1, self.hyper_fc2
        ]
        
        for hyper_net in hypernetworks:
            nn.init.normal_(hyper_net.weight, std=0.01)
            nn.init.constant_(hyper_net.bias, 0.0)
    
    def generate_conv1_weights(self):
        weights_flat = self.hyper_conv1(self.z_signal_conv1)
        return weights_flat.reshape(self.in_size, 3, self.f_size, self.f_size)
    
    def generate_conv2_weights(self):
        weights_flat = self.hyper_conv2(self.z_signal_conv2)
        return weights_flat.reshape(self.out_size, self.in_size, self.f_size, self.f_size)
    
    def generate_conv3_weights(self):
        weights_flat = self.hyper_conv3(self.z_signal_conv3)
        return weights_flat.reshape(128, self.out_size, 3, 3)
    
    def generate_fc1_weights(self):
        weights_flat = self.hyper_fc1(self.z_signal_fc1)
        return weights_flat.reshape(256, 128 * 4 * 4)
    
    def generate_fc2_weights(self):
        weights_flat = self.hyper_fc2(self.z_signal_fc2)
        return weights_flat.reshape(100, 256)
        
    def forward(self, x):
        # Conv1块
        conv1_weights = self.generate_conv1_weights()
        x = F.conv2d(x, conv1_weights, bias=self.conv1_bias, padding='same')
        x = self.pool(F.relu(x))
        
        # Conv2块
        conv2_weights = self.generate_conv2_weights()
        x = F.conv2d(x, conv2_weights, bias=self.conv2_bias, padding='same')
        x = self.pool(F.relu(x))
        
        # Conv3块
        conv3_weights = self.generate_conv3_weights()
        x = F.conv2d(x, conv3_weights, bias=self.conv3_bias, padding='same')
        x = self.pool(F.relu(x))
        
        # 扁平化
        x = x.view(x.size(0), -1)
        
        # FC1块
        fc1_weights = self.generate_fc1_weights()
        x = F.linear(x, fc1_weights, bias=self.fc1_bias)
        x = F.relu(x)
        x = self.dropout(x)
        
        # FC2块
        fc2_weights = self.generate_fc2_weights()
        x = F.linear(x, fc2_weights, bias=self.fc2_bias)
        
        return x

def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=25):
    """训练一个epoch，针对大批次优化打印频率"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 指标跟踪
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 大批次训练时减少打印频率
        if batch_idx % log_interval == 0:
            current_acc = 100. * correct / total
            print(f'  Batch: {batch_idx:3d}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {current_acc:6.2f}% | '
                  f'Samples/sec: {target.size(0) * (batch_idx + 1) / (time.time() - epoch_start_time):.0f}')
            
            # Z信号监控
            conv_norms = [
                model.z_signal_conv1.norm().item(),
                model.z_signal_conv2.norm().item(),
                model.z_signal_conv3.norm().item()
            ]
            fc_norms = [
                model.z_signal_fc1.norm().item(),
                model.z_signal_fc2.norm().item()
            ]
            print(f'    Conv Z-norms: {conv_norms[0]:.4f}, {conv_norms[1]:.4f}, {conv_norms[2]:.4f}')
            print(f'    FC Z-norms: {fc_norms[0]:.4f}, {fc_norms[1]:.4f}')
    
    train_loss /= len(train_loader)
    train_acc = correct / total
    train_err = 1.0 - train_acc
    return train_loss, train_err

def evaluate(model, device, data_loader):
    """模型评估函数"""
    model.eval()
    loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    loss /= total
    acc = correct / total
    err = 1.0 - acc
    return loss, err

def count_parameters(model):
    """统计模型参数"""
    total_params = 0
    hyper_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            
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

def calculate_optimal_batch_size(model, device, sample_data):
    """
    自动计算最优batch size
    基于GPU内存动态调整批次大小
    """
    model.train()
    batch_sizes = [128, 256, 512, 1024, 1536, 2048]  # H100可能支持的批次大小
    optimal_batch_size = 128  # 默认值
    
    print("Finding optimal batch size for H100...")
    
    for batch_size in batch_sizes:
        try:
            # 创建测试批次
            test_data = sample_data[:batch_size].to(device)
            test_target = torch.randint(0, 100, (batch_size,)).to(device)
            
            # 测试前向传播
            with torch.cuda.amp.autocast():  # 使用混合精度
                output = model(test_data)
                loss = F.cross_entropy(output, test_target)
            
            # 测试反向传播
            loss.backward()
            model.zero_grad()
            
            optimal_batch_size = batch_size
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Batch size {batch_size}: OK (Memory: {memory_used:.1f}GB)")
            
            # 清理内存
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch size {batch_size}: OOM")
                break
            else:
                raise e
    
    return optimal_batch_size

def main():
    """主训练函数 - H100大批次优化版本"""
    global epoch_start_time
    
    print("Starting CIFAR-100 ALL LAYERS HyperNetwork - H100 Large Batch Version")
    print("="*70)
    
    # 可重复性设置
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = set_device()
    
    # 启用混合精度训练和优化设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 优化卷积性能
        torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
        torch.backends.cudnn.allow_tf32 = True
    
    # 基础超参数
    epochs = 50
    lr = 0.001
    f_size = 5
    in_size = 32
    out_size = 64
    z_dim = 8
    
    # 数据预处理（更激进的增强策略）
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # 增加旋转增强
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 数据集加载
    print("Loading CIFAR-100 dataset...")
    train_dataset = datasets.CIFAR100('data', train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR100('data', train=False, transform=transform_test)
    
    train_size = len(train_dataset) - 5000
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, 5000]
    )
    
    # 创建模型进行batch size测试
    model = HyperCNNCIFAR_AllLayers(
        f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim
    ).to(device)
    
    # 自动计算最优batch size
    sample_data = torch.randn(2048, 3, 32, 32)  # 创建样本数据
    optimal_batch_size = calculate_optimal_batch_size(model, device, sample_data)
    
    # 根据batch size调整学习率（线性缩放规则）
    scale_factor = optimal_batch_size / 128  # 基准batch size为128
    adjusted_lr = lr * scale_factor
    
    print(f"\nOptimal Configuration:")
    print(f"  Batch size: {optimal_batch_size}")
    print(f"  Adjusted learning rate: {adjusted_lr:.6f} (scale factor: {scale_factor:.2f})")
    print("="*70)
    
    # 创建优化的数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=optimal_batch_size, shuffle=True,
        num_workers=8,  # 增加工作进程数
        pin_memory=True,
        persistent_workers=True,  # 保持工作进程活跃
        prefetch_factor=2  # 预取因子
    )
    val_loader = DataLoader(
        val_dataset, batch_size=optimal_batch_size * 2,  # 验证时可以用更大批次
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=optimal_batch_size * 2,
        num_workers=4, pin_memory=True
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("="*70)
    
    # 重新创建模型和优化器
    model = HyperCNNCIFAR_AllLayers(
        f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim
    ).to(device)
    
    # 使用带权重衰减的AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=adjusted_lr, weight_decay=1e-4)
    
    # 更激进的学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=adjusted_lr * 2, 
        epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print("\nALL LAYERS HyperNetwork CNN parameters:")
    count_parameters(model)
    print("="*70)
    
    # 训练循环
    best_val_err = 1.0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs} - Large Batch HyperNetwork CNN")
        print("-" * 50)
        
        # 训练
        train_loss, train_err = train_epoch(model, device, train_loader, optimizer, epoch)
        
        # 验证
        val_loss, val_err = evaluate(model, device, val_loader)
        
        # 保存最佳模型
        if val_err < best_val_err:
            best_val_err = val_err
            torch.save(model.state_dict(), 'cifar100_all_layers_hyper_large_batch_best.pt')
            print(f"    *** New best validation error: {100*val_err:.2f}% - Model saved! ***")
        
        # 打印结果
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        samples_per_sec = len(train_dataset) / epoch_time
        
        print(f"\nResults: Train Err: {100*train_err:.2f}% | Val Err: {100*val_err:.2f}% | "
              f"Best Val Err: {100*best_val_err:.2f}% | LR: {current_lr:.6f}")
        print(f"Time: {epoch_time:.1f}s | Samples/sec: {samples_per_sec:.0f}")
        
        # 更新学习率
        scheduler.step()
    
    # 最终评估
    print("\n" + "="*70)
    print("FINAL RESULTS - H100 Large Batch Version")
    print("="*70)
    
    model.load_state_dict(torch.load('cifar100_all_layers_hyper_large_batch_best.pt'))
    test_loss, test_err = evaluate(model, device, test_loader)
    test_acc = (1 - test_err) * 100
    
    # 性能对比
    std_test_acc = 16.23
    fixed_hyper_acc = 33.52
    full_hyper_acc = 54.16
    original_all_layers_acc = 36.13  # 原始小批次版本结果
    
    print(f"Standard CNN - Test Accuracy: {std_test_acc:.2f}%")
    print(f"Fixed HyperNetwork (1 layer) - Test Accuracy: {fixed_hyper_acc:.2f}%")
    print(f"Full HyperNetwork (complex) - Test Accuracy: {full_hyper_acc:.2f}%")
    print(f"ALL LAYERS HyperNetwork (original) - Test Accuracy: {original_all_layers_acc:.2f}%")
    print(f"ALL LAYERS HyperNetwork (Large Batch) - Test Accuracy: {test_acc:.2f}%")
    
    improvement_vs_original = test_acc - original_all_layers_acc
    print(f"Improvement vs Original ALL LAYERS: {improvement_vs_original:+.2f}%")
    
    # Z信号最终状态
    print(f"\nFinal hypernetwork signals:")
    print(f"z_signal_conv1: {model.z_signal_conv1.detach().cpu().numpy().flatten()}")
    print(f"z_signal_conv2: {model.z_signal_conv2.detach().cpu().numpy().flatten()}")
    print(f"z_signal_conv3: {model.z_signal_conv3.detach().cpu().numpy().flatten()}")
    print(f"z_signal_fc1: {model.z_signal_fc1.detach().cpu().numpy().flatten()}")
    print(f"z_signal_fc2: {model.z_signal_fc2.detach().cpu().numpy().flatten()}")
    
    print(f"\nOptimal batch size used: {optimal_batch_size}")
    print("Large batch training completed!")

if __name__ == "__main__":
    main()
