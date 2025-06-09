'''
CIFAR-100 HyperNetwork ALL LAYERS VERSION - Multi-GPU Training (Anti-Overfitting Optimized)
==========================================================================================
基于cifar100_hypernetwork_fixed_all_layers_multi_gpu_large.py的优化版本
主要改进：
1. 减少参数量防止过拟合：Z维度12->6，网络宽度减半
2. 增强正则化：更大dropout，weight_decay，Batch Normalization
3. 改进数据增强：更强的随机变换
4. 早停机制：防止过拟合
5. 改进学习率调度：Cosine Annealing + Warm Restart
'''

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import time
import math

def setup(rank, world_size):
    """初始化分布式训练进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'  # 避免端口冲突
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式训练"""
    dist.destroy_process_group()

def set_device(rank):
    """设置设备，rank对应GPU ID"""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
        if rank == 0:  # 只在主进程打印
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        if rank == 0:
            print("CUDA not available, using CPU")
    return device

class HyperCNNCIFAR_AllLayers_Optimized(nn.Module):
    """优化版全层HyperNetwork CIFAR-100分类器"""
    
    def __init__(self, f_size=5, in_size=32, out_size=64, z_dim=6):
        super(HyperCNNCIFAR_AllLayers_Optimized, self).__init__()
        
        self.f_size = f_size
        self.in_size = in_size      # 优化: 64->32 (减少参数)
        self.out_size = out_size    # 优化: 128->64 (减少参数)
        self.z_dim = z_dim          # 优化: 12->6 (大幅减少参数)
        
        # 固定组件 - 添加Batch Normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)  # 较小dropout for feature extraction
        self.dropout2 = nn.Dropout(0.5)   # 较大dropout for classifier
        self.dropout3 = nn.Dropout(0.7)   # 最大dropout for final layer
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(in_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn_fc1 = nn.BatchNorm1d(256)
        
        # Z信号参数 - 减少维度
        self.z_signal_conv1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_conv2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_conv3 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_fc1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_fc2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        
        # HyperNetwork生成器 - 大幅减少参数
        conv1_weight_size = in_size * 3 * f_size * f_size
        self.hyper_conv1 = nn.Linear(z_dim, conv1_weight_size)
        self.conv1_bias = nn.Parameter(torch.zeros(in_size))
        
        conv2_weight_size = out_size * in_size * f_size * f_size
        self.hyper_conv2 = nn.Linear(z_dim, conv2_weight_size)
        self.conv2_bias = nn.Parameter(torch.zeros(out_size))
        
        conv3_weight_size = 128 * out_size * 3 * 3  # 保持128
        self.hyper_conv3 = nn.Linear(z_dim, conv3_weight_size)
        self.conv3_bias = nn.Parameter(torch.zeros(128))
        
        fc1_input_size = 128 * 4 * 4  # 对应conv3输出
        fc1_weight_size = 256 * fc1_input_size  # 保持256
        self.hyper_fc1 = nn.Linear(z_dim, fc1_weight_size)
        self.fc1_bias = nn.Parameter(torch.zeros(256))
        
        fc2_weight_size = 100 * 256  # 对应fc1输出
        self.hyper_fc2 = nn.Linear(z_dim, fc2_weight_size)
        self.fc2_bias = nn.Parameter(torch.zeros(100))
        
        self._init_hypernetworks()
        
    def _init_hypernetworks(self):
        """改进的初始化方法"""
        hypernetworks = [
            self.hyper_conv1, self.hyper_conv2, self.hyper_conv3,
            self.hyper_fc1, self.hyper_fc2
        ]
        
        for hyper_net in hypernetworks:
            # 使用Xavier初始化
            nn.init.xavier_normal_(hyper_net.weight, gain=0.1)
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
        # Conv1块 + BN + Dropout
        conv1_weights = self.generate_conv1_weights()
        x = F.conv2d(x, conv1_weights, bias=self.conv1_bias, padding='same')
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Conv2块 + BN + Dropout
        conv2_weights = self.generate_conv2_weights()
        x = F.conv2d(x, conv2_weights, bias=self.conv2_bias, padding='same')
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Conv3块 + BN + Dropout
        conv3_weights = self.generate_conv3_weights()
        x = F.conv2d(x, conv3_weights, bias=self.conv3_bias, padding='same')
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        
        # 扁平化
        x = x.view(x.size(0), -1)
        
        # FC1块 + BN + Dropout
        fc1_weights = self.generate_fc1_weights()
        x = F.linear(x, fc1_weights, bias=self.fc1_bias)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # FC2块 + 最大Dropout
        fc2_weights = self.generate_fc2_weights()
        x = self.dropout3(x)  # 最大dropout
        x = F.linear(x, fc2_weights, bias=self.fc2_bias)
        
        return x

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_error):
        if self.best_score is None:
            self.best_score = val_error
        elif val_error < self.best_score - self.min_delta:
            self.best_score = val_error
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_epoch(model, device, train_loader, optimizer, epoch, rank, world_size, log_interval=20):
    """分布式训练一个epoch"""
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
        
        # 梯度裁剪 - 更严格
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # 指标跟踪
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 只在主进程打印日志
        if rank == 0 and batch_idx % log_interval == 0:
            current_acc = 100. * correct / total
            print(f'  Batch: {batch_idx:3d}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {current_acc:6.2f}% | '
                  f'GPU {rank}')
            
            # Z信号监控（只在主进程）
            conv_norms = [
                model.module.z_signal_conv1.norm().item(),
                model.module.z_signal_conv2.norm().item(),
                model.module.z_signal_conv3.norm().item()
            ]
            fc_norms = [
                model.module.z_signal_fc1.norm().item(),
                model.module.z_signal_fc2.norm().item()
            ]
            print(f'    Conv Z-norms: {conv_norms[0]:.4f}, {conv_norms[1]:.4f}, {conv_norms[2]:.4f}')
            print(f'    FC Z-norms: {fc_norms[0]:.4f}, {fc_norms[1]:.4f}')
    
    train_loss /= len(train_loader)
    train_acc = correct / total
    train_err = 1.0 - train_acc
    return train_loss, train_err

def evaluate(model, device, data_loader, rank):
    """分布式评估函数"""
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
    
    # 聚合所有GPU的结果
    loss_tensor = torch.tensor(loss).to(device)
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total).to(device)
    
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    total_loss = loss_tensor.item() / total_tensor.item()
    total_acc = correct_tensor.item() / total_tensor.item()
    total_err = 1.0 - total_acc
    
    return total_loss, total_err

def count_parameters(model, rank):
    """统计模型参数（只在主进程执行）"""
    if rank != 0:
        return 0, 0
        
    total_params = 0
    hyper_params = 0
    
    # 使用model.module访问DDP包装的模型
    actual_model = model.module if hasattr(model, 'module') else model
    
    for name, param in actual_model.named_parameters():
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

def train_worker(rank, world_size, epochs):
    """分布式训练工作进程"""
    
    # 设置分布式训练
    setup(rank, world_size)
    
    # 设置设备
    device = set_device(rank)
    
    # 随机种子设置（每个进程使用不同种子以获得数据多样性）
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    
    # 超参数配置 - 防过拟合优化
    batch_size = 128  # 保持合理大小
    test_batch_size = 256
    lr = 0.001  # 适中学习率
    f_size = 5
    in_size = 32      # 减少: 64->32
    out_size = 64     # 减少: 128->64
    z_dim = 6         # 大幅减少: 12->6
    weight_decay = 5e-4  # 增强L2正则化
    
    if rank == 0:
        print(f"Optimized Anti-Overfitting Multi-GPU Training Configuration:")
        print(f"  World size (H20 GPUs): {world_size}")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Effective batch size: {batch_size * world_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Z-dimension: {z_dim} (reduced from 12)")
        print(f"  Conv channels: {in_size}->{out_size}->128 (reduced from 64->128->256)")
        print(f"  Anti-overfitting: BN + Multi-level Dropout + Early Stopping")
        print("="*70)
    
    # 增强数据预处理 - 更强的数据增强
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # 新增: 随机旋转
        transforms.RandomCrop(32, padding=4),  # 新增: 随机裁剪
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 新增: 颜色变换
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 新增: 随机擦除
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 数据集加载
    train_dataset = datasets.CIFAR100('data', train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR100('data', train=False, transform=transform_test)
    
    # 验证集分割
    train_size = len(train_dataset) - 5000
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, 5000]
    )
    
    # 分布式采样器
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, sampler=val_sampler,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, sampler=test_sampler,
        num_workers=2, pin_memory=True
    )
    
    if rank == 0:
        print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        print(f"Batches per GPU per epoch: {len(train_loader)}")
        print("="*70)
    
    # 模型创建 - 使用优化参数
    model = HyperCNNCIFAR_AllLayers_Optimized(
        f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim
    ).to(device)
    
    # DDP包装
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # 优化器和调度器 - 改进版
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 早停机制
    early_stopping = EarlyStopping(patience=15, min_delta=0.001) if rank == 0 else None
    
    # 参数统计（只在主进程）
    if rank == 0:
        print("\nOptimized Anti-Overfitting ALL LAYERS HyperNetwork CNN:")
        count_parameters(model, rank)
        print("="*70)
    
    # 训练循环
    best_val_err = 1.0
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 设置采样器的epoch（确保每个epoch数据顺序不同）
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{epochs} - Optimized Anti-Overfitting HyperNetwork CNN")
            print("-" * 60)
        
        # 训练
        train_loss, train_err = train_epoch(
            model, device, train_loader, optimizer, epoch, rank, world_size
        )
        
        # 验证
        val_loss, val_err = evaluate(model, device, val_loader, rank)
        
        # 更新学习率
        scheduler.step()
        
        # 记录准确率（用于分析）
        train_acc = (1 - train_err) * 100
        val_acc = (1 - val_err) * 100
        if rank == 0:
            train_accs.append(train_acc)
            val_accs.append(val_acc)
        
        # 保存最佳模型（只在主进程）
        if rank == 0:
            if val_err < best_val_err:
                best_val_err = val_err
                torch.save(model.module.state_dict(), 'cifar100_all_layers_hyper_optimized_best.pt')
                print(f"    *** New best validation error: {100*val_err:.2f}% - Model saved! ***")
            
            # 打印结果
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            effective_samples_per_sec = len(train_dataset) * world_size / epoch_time
            
            print(f"\nResults: Train Err: {100*train_err:.2f}% | Val Err: {100*val_err:.2f}% | "
                  f"Best Val Err: {100*best_val_err:.2f}% | LR: {current_lr:.6f}")
            print(f"Train-Val Gap: {100*(train_err - val_err):+.2f}% | Time: {epoch_time:.1f}s | "
                  f"Samples/sec: {effective_samples_per_sec:.0f}")
            
            # 早停检查
            if early_stopping is not None:
                early_stopping(val_err)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
    
    # 最终评估
    if rank == 0:
        print("\n" + "="*70)
        print("FINAL RESULTS - Optimized Anti-Overfitting Version")
        print("="*70)
        
        # 加载最佳模型
        model.module.load_state_dict(torch.load('cifar100_all_layers_hyper_optimized_best.pt'))
        
    # 所有进程参与最终测试
    test_loss, test_err = evaluate(model, device, test_loader, rank)
    test_acc = (1 - test_err) * 100
    
    if rank == 0:
        # 性能对比
        std_test_acc = 16.23
        fixed_hyper_acc = 33.52
        full_hyper_acc = 54.16
        original_all_layers_acc = 36.13
        multi_gpu_acc = 42.76
        large_multi_gpu_acc = 53.71
        
        print(f"Standard CNN - Test Accuracy: {std_test_acc:.2f}%")
        print(f"Fixed HyperNetwork (1 layer) - Test Accuracy: {fixed_hyper_acc:.2f}%")
        print(f"Full HyperNetwork (complex) - Test Accuracy: {full_hyper_acc:.2f}%")
        print(f"ALL LAYERS HyperNetwork (original) - Test Accuracy: {original_all_layers_acc:.2f}%")
        print(f"ALL LAYERS HyperNetwork (Multi-GPU) - Test Accuracy: {multi_gpu_acc:.2f}%")
        print(f"ALL LAYERS HyperNetwork (Large Multi-GPU) - Test Accuracy: {large_multi_gpu_acc:.2f}%")
        print(f"ALL LAYERS HyperNetwork (Optimized Anti-Overfitting) - Test Accuracy: {test_acc:.2f}%")
        
        improvement_vs_large = test_acc - large_multi_gpu_acc
        improvement_vs_original = test_acc - original_all_layers_acc
        print(f"Improvement vs Large Multi-GPU: {improvement_vs_large:+.2f}%")
        print(f"Improvement vs Original: {improvement_vs_original:+.2f}%")
        
        # 过拟合分析
        final_train_acc = train_accs[-1] if train_accs else 0
        final_val_acc = val_accs[-1] if val_accs else 0
        train_test_gap = final_train_acc - test_acc
        print(f"\nOverfitting Analysis:")
        print(f"Final Train Accuracy: {final_train_acc:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Train-Test Gap: {train_test_gap:+.2f}% (lower is better)")
        
        # Z信号最终状态
        print(f"\nFinal hypernetwork signals:")
        print(f"z_signal_conv1: {model.module.z_signal_conv1.detach().cpu().numpy().flatten()}")
        print(f"z_signal_conv2: {model.module.z_signal_conv2.detach().cpu().numpy().flatten()}")
        print(f"z_signal_conv3: {model.module.z_signal_conv3.detach().cpu().numpy().flatten()}")
        print(f"z_signal_fc1: {model.module.z_signal_fc1.detach().cpu().numpy().flatten()}")
        print(f"z_signal_fc2: {model.module.z_signal_fc2.detach().cpu().numpy().flatten()}")
        
        print(f"\nUsed {world_size} GPUs for optimized anti-overfitting training")
        print("Optimized Anti-Overfitting Multi-GPU training completed!")
    
    # 清理
    cleanup()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIFAR-100 Optimized Anti-Overfitting Multi-GPU HyperNetwork Training')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs to train (default: 60)')
    return parser.parse_args()

def main():
    """主函数 - 启动多进程分布式训练"""
    
    # 解析命令行参数
    args = parse_args()
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("CUDA not available! Multi-GPU training requires CUDA.")
        return
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Only {world_size} GPU available. Multi-GPU training requires at least 2 GPUs.")
        print("Consider running the single GPU version instead.")
        return
    
    print("Starting CIFAR-100 ALL LAYERS HyperNetwork - Optimized Anti-Overfitting Multi-GPU Version")
    print(f"Using {world_size} GPUs: {[torch.cuda.get_device_name(i) for i in range(world_size)]}")
    print(f"Training for {args.epochs} epochs")
    print("="*70)
    
    # 启动多进程训练
    mp.spawn(train_worker, args=(world_size, args.epochs), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
