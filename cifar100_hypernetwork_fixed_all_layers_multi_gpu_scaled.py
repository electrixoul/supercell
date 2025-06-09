'''
CIFAR-100 HyperNetwork ALL LAYERS VERSION - Multi-GPU Training (Scaled Parameters)
==================================================================================
基于cifar100_hypernetwork_fixed_all_layers_multi_gpu.py的参数扩展版本
- 增加网络参数规模和batch size
- 添加命令行参数支持
- 使用DistributedDataParallel (DDP)实现多GPU训练
- 自动检测可用GPU数量并分配训练任务
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

def setup(rank, world_size):
    """初始化分布式训练进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # 避免端口冲突
    
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

class HyperCNNCIFAR_AllLayers_Scaled(nn.Module):
    """参数扩展版全层HyperNetwork CIFAR-100分类器"""
    
    def __init__(self, f_size=5, in_size=64, out_size=128, z_dim=16):
        super(HyperCNNCIFAR_AllLayers_Scaled, self).__init__()
        
        self.f_size = f_size
        self.in_size = in_size      # 从32扩展到64
        self.out_size = out_size    # 从64扩展到128
        self.z_dim = z_dim          # 从8扩展到16
        
        # 固定组件
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Z信号参数 - 扩展维度
        self.z_signal_conv1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_conv2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_conv3 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_fc1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_fc2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        
        # HyperNetwork生成器 - 参数规模扩展
        # Conv1: 3 -> in_size (64), kernel 5x5
        conv1_weight_size = in_size * 3 * f_size * f_size
        self.hyper_conv1 = nn.Linear(z_dim, conv1_weight_size)
        self.conv1_bias = nn.Parameter(torch.zeros(in_size))
        
        # Conv2: in_size -> out_size (64->128), kernel 5x5
        conv2_weight_size = out_size * in_size * f_size * f_size
        self.hyper_conv2 = nn.Linear(z_dim, conv2_weight_size)
        self.conv2_bias = nn.Parameter(torch.zeros(out_size))
        
        # Conv3: out_size -> 256 (128->256), kernel 3x3
        conv3_out_size = 256
        conv3_weight_size = conv3_out_size * out_size * 3 * 3
        self.hyper_conv3 = nn.Linear(z_dim, conv3_weight_size)
        self.conv3_bias = nn.Parameter(torch.zeros(conv3_out_size))
        
        # FC1: (256 * 4 * 4) -> 512 (扩展到512)
        fc1_input_size = conv3_out_size * 4 * 4  # 256 * 4 * 4
        fc1_output_size = 512
        fc1_weight_size = fc1_output_size * fc1_input_size
        self.hyper_fc1 = nn.Linear(z_dim, fc1_weight_size)
        self.fc1_bias = nn.Parameter(torch.zeros(fc1_output_size))
        
        # FC2: 512 -> 100
        fc2_weight_size = 100 * fc1_output_size
        self.hyper_fc2 = nn.Linear(z_dim, fc2_weight_size)
        self.fc2_bias = nn.Parameter(torch.zeros(100))
        
        # 存储关键尺寸用于forward
        self.conv3_out_size = conv3_out_size
        self.fc1_input_size = fc1_input_size
        self.fc1_output_size = fc1_output_size
        
        self._init_hypernetworks()
        
    def _init_hypernetworks(self):
        """初始化HyperNetwork参数"""
        hypernetworks = [
            self.hyper_conv1, self.hyper_conv2, self.hyper_conv3,
            self.hyper_fc1, self.hyper_fc2
        ]
        
        for hyper_net in hypernetworks:
            nn.init.normal_(hyper_net.weight, std=0.01)
            nn.init.constant_(hyper_net.bias, 0.0)
    
    def generate_conv1_weights(self):
        """生成conv1权重: 3 -> in_size, kernel 5x5"""
        weights_flat = self.hyper_conv1(self.z_signal_conv1)
        return weights_flat.reshape(self.in_size, 3, self.f_size, self.f_size)
    
    def generate_conv2_weights(self):
        """生成conv2权重: in_size -> out_size, kernel 5x5"""
        weights_flat = self.hyper_conv2(self.z_signal_conv2)
        return weights_flat.reshape(self.out_size, self.in_size, self.f_size, self.f_size)
    
    def generate_conv3_weights(self):
        """生成conv3权重: out_size -> 256, kernel 3x3"""
        weights_flat = self.hyper_conv3(self.z_signal_conv3)
        return weights_flat.reshape(self.conv3_out_size, self.out_size, 3, 3)
    
    def generate_fc1_weights(self):
        """生成fc1权重: (256*4*4) -> 512"""
        weights_flat = self.hyper_fc1(self.z_signal_fc1)
        return weights_flat.reshape(self.fc1_output_size, self.fc1_input_size)
    
    def generate_fc2_weights(self):
        """生成fc2权重: 512 -> 100"""
        weights_flat = self.hyper_fc2(self.z_signal_fc2)
        return weights_flat.reshape(100, self.fc1_output_size)
        
    def forward(self, x):
        # Conv1块: 3 -> 64
        conv1_weights = self.generate_conv1_weights()
        x = F.conv2d(x, conv1_weights, bias=self.conv1_bias, padding='same')
        x = self.pool(F.relu(x))  # 32->16
        
        # Conv2块: 64 -> 128
        conv2_weights = self.generate_conv2_weights()
        x = F.conv2d(x, conv2_weights, bias=self.conv2_bias, padding='same')
        x = self.pool(F.relu(x))  # 16->8
        
        # Conv3块: 128 -> 256
        conv3_weights = self.generate_conv3_weights()
        x = F.conv2d(x, conv3_weights, bias=self.conv3_bias, padding='same')
        x = self.pool(F.relu(x))  # 8->4
        
        # 扁平化
        x = x.view(x.size(0), -1)  # (batch_size, 256*4*4)
        
        # FC1块: (256*4*4) -> 512
        fc1_weights = self.generate_fc1_weights()
        x = F.linear(x, fc1_weights, bias=self.fc1_bias)
        x = F.relu(x)
        x = self.dropout(x)
        
        # FC2块: 512 -> 100
        fc2_weights = self.generate_fc2_weights()
        x = F.linear(x, fc2_weights, bias=self.fc2_bias)
        
        return x

def train_epoch(model, device, train_loader, optimizer, epoch, rank, world_size, log_interval=50):
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
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

def train_worker(rank, world_size, args):
    """分布式训练工作进程"""
    
    # 设置分布式训练
    setup(rank, world_size)
    
    # 设置设备
    device = set_device(rank)
    
    # 随机种子设置（每个进程使用不同种子以获得数据多样性）
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    
    # 参数配置 - 扩展规模
    batch_size = args.batch_size    # 从命令行获取，默认256（原版128的2倍）
    test_batch_size = 512           # 扩展测试batch size
    epochs = args.epochs            # 从命令行获取
    lr = args.lr                    # 从命令行获取
    f_size = 5                      # 保持不变
    in_size = 64                    # 扩展: 32->64
    out_size = 128                  # 扩展: 64->128
    z_dim = 16                      # 扩展: 8->16
    
    if rank == 0:
        print("Starting CIFAR-100 ALL LAYERS HyperNetwork - Multi-GPU Scaled Version")
        print("="*80)
        print(f"Scaled Parameters Configuration:")
        print(f"  Z-dimension: {z_dim} (vs original 8)")
        print(f"  Conv channels: {in_size}->{out_size}->256 (vs original 32->64->128)")
        print(f"  FC layers: 256*4*4->512->100 (vs original 128*4*4->256->100)")
        print(f"  Batch size per GPU: {batch_size} (vs original 128)")
        print(f"  Effective batch size: {batch_size * world_size}")
        print(f"  Training epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print("="*80)
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # 添加更多数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 数据集加载
    if rank == 0:
        print("Loading CIFAR-100 dataset...")
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
    
    # 数据加载器 - 增加workers数量处理更大batch
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, sampler=test_sampler,
        num_workers=4, pin_memory=True
    )
    
    if rank == 0:
        print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        print(f"Batches per GPU per epoch: {len(train_loader)}")
        print("="*80)
    
    # 模型创建 - 使用扩展参数
    model = HyperCNNCIFAR_AllLayers_Scaled(
        f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim
    ).to(device)
    
    # DDP包装
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # 优化器和调度器 - 调整学习率衰减
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    
    # 参数统计（只在主进程）
    if rank == 0:
        print("\nScaled Multi-GPU ALL LAYERS HyperNetwork CNN parameters:")
        count_parameters(model, rank)
        print("="*80)
    
    # 训练循环
    best_val_err = 1.0
    for epoch in range(epochs):
        start_time = time.time()
        
        # 设置采样器的epoch（确保每个epoch数据顺序不同）
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{epochs} - Scaled Multi-GPU HyperNetwork CNN")
            print("-" * 60)
        
        # 训练
        train_loss, train_err = train_epoch(
            model, device, train_loader, optimizer, epoch, rank, world_size
        )
        
        # 验证
        val_loss, val_err = evaluate(model, device, val_loader, rank)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型（只在主进程）
        if rank == 0:
            if val_err < best_val_err:
                best_val_err = val_err
                torch.save(model.module.state_dict(), 'cifar100_all_layers_hyper_scaled_best.pt')
                print(f"    *** New best validation error: {100*val_err:.2f}% - Model saved! ***")
            
            # 打印结果
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            effective_samples_per_sec = len(train_dataset) * world_size / epoch_time
            
            print(f"\nResults: Train Err: {100*train_err:.2f}% | Val Err: {100*val_err:.2f}% | "
                  f"Best Val Err: {100*best_val_err:.2f}% | LR: {current_lr:.6f}")
            print(f"Time: {epoch_time:.1f}s | Effective Samples/sec: {effective_samples_per_sec:.0f}")
    
    # 最终评估
    if rank == 0:
        print("\n" + "="*80)
        print("FINAL RESULTS - Scaled Multi-GPU Version")
        print("="*80)
        
        # 加载最佳模型
        model.module.load_state_dict(torch.load('cifar100_all_layers_hyper_scaled_best.pt'))
        
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
        
        print(f"Standard CNN - Test Accuracy: {std_test_acc:.2f}%")
        print(f"Fixed HyperNetwork (1 layer) - Test Accuracy: {fixed_hyper_acc:.2f}%")
        print(f"Full HyperNetwork (complex) - Test Accuracy: {full_hyper_acc:.2f}%")
        print(f"ALL LAYERS HyperNetwork (original) - Test Accuracy: {original_all_layers_acc:.2f}%")
        print(f"ALL LAYERS HyperNetwork (Multi-GPU) - Test Accuracy: {multi_gpu_acc:.2f}%")
        print(f"ALL LAYERS HyperNetwork (Scaled Multi-GPU) - Test Accuracy: {test_acc:.2f}%")
        
        improvement_vs_multi_gpu = test_acc - multi_gpu_acc
        improvement_vs_original = test_acc - original_all_layers_acc
        print(f"Improvement vs Multi-GPU: {improvement_vs_multi_gpu:+.2f}%")
        print(f"Improvement vs Original: {improvement_vs_original:+.2f}%")
        
        # Z信号最终状态
        print(f"\nFinal hypernetwork signals:")
        print(f"z_signal_conv1: {model.module.z_signal_conv1.detach().cpu().numpy().flatten()}")
        print(f"z_signal_conv2: {model.module.z_signal_conv2.detach().cpu().numpy().flatten()}")
        print(f"z_signal_conv3: {model.module.z_signal_conv3.detach().cpu().numpy().flatten()}")
        print(f"z_signal_fc1: {model.module.z_signal_fc1.detach().cpu().numpy().flatten()}")
        print(f"z_signal_fc2: {model.module.z_signal_fc2.detach().cpu().numpy().flatten()}")
        
        print(f"\nUsed {world_size} GPUs for scaled training")
        print("Scaled Multi-GPU training completed!")
    
    # 清理
    cleanup()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIFAR-100 Scaled Multi-GPU HyperNetwork Training')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size per GPU for training (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    
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
    
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Effective batch size: {args.batch_size * world_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Available GPUs: {world_size}")
    print("="*80)
    
    # 启动多进程训练
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
