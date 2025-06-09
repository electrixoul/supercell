'''
CIFAR-100 HyperNetwork ALL LAYERS VERSION - Multi-GPU Stable Version
===================================================================
修复NCCL超时问题的稳定版本：
1. 增加NCCL超时设置
2. 优化内存使用
3. 添加错误恢复机制
4. 减小模型复杂度避免通信瓶颈
5. 改善分布式训练稳定性
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
import random
import warnings
import logging
from datetime import datetime
import json

def setup(rank, world_size):
    """初始化分布式训练进程组 - 增加稳定性配置"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 换一个端口避免冲突
    
    # NCCL配置优化
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟超时
    os.environ['NCCL_BLOCKING_WAIT'] = '1'  # 阻塞等待
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 异步错误处理
    os.environ['NCCL_DEBUG'] = 'WARN'  # 减少调试输出
    os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用InfiniBand
    os.environ['NCCL_P2P_DISABLE'] = '1'  # 禁用P2P以提高稳定性
    
    # 初始化进程组 - 增加超时时间
    from datetime import timedelta
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(minutes=30)  # 30分钟超时
    )

def setup_logging(rank):
    """设置日志系统 - 只在主进程设置"""
    if rank == 0:
        # 创建日志目录
        log_dir = "training_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置日志文件路径
        log_file = os.path.join(log_dir, f"cifar100_hypernetwork_stable_{timestamp}.log")
        json_file = os.path.join(log_dir, f"training_progress_{timestamp}.json")
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("="*80)
        logger.info("CIFAR-100 HyperNetwork Stable Training - Log Started")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Progress file: {json_file}")
        logger.info("="*80)
        
        return logger, json_file
    else:
        return None, None

def log_training_progress(json_file, epoch, train_err, val_err, best_val_err, lr, epoch_time, test_acc=None):
    """记录训练进度到JSON文件"""
    if json_file is None:
        return
        
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'epoch': epoch,
        'train_error': float(train_err),
        'validation_error': float(val_err),
        'best_validation_error': float(best_val_err),
        'learning_rate': float(lr),
        'epoch_time_seconds': float(epoch_time),
        'train_accuracy': float(1 - train_err),
        'validation_accuracy': float(1 - val_err),
        'best_validation_accuracy': float(1 - best_val_err)
    }
    
    if test_acc is not None:
        progress_data['final_test_accuracy'] = float(test_acc / 100)
    
    # 读取现有数据
    try:
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
        else:
            data = {'training_history': [], 'config': {}}
    except:
        data = {'training_history': [], 'config': {}}
    
    # 添加新的epoch数据
    data['training_history'].append(progress_data)
    
    # 保存到文件
    try:
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress to JSON: {e}")

def log_config(logger, json_file, config):
    """记录训练配置"""
    if logger:
        logger.info("Training Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    if json_file:
        try:
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {'training_history': [], 'config': {}}
            
            data['config'] = config
            data['start_time'] = datetime.now().isoformat()
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config to JSON: {e}")

def log_model_info(logger, total_params, hyper_params):
    """记录模型信息"""
    if logger:
        logger.info("Model Information:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  HyperNetwork parameters: {hyper_params:,}")
        logger.info(f"  Standard parameters: {total_params - hyper_params:,}")
        logger.info(f"  HyperNetwork ratio: {100.*hyper_params/total_params:.2f}%")

def log_final_results(logger, json_file, test_acc, z_signals):
    """记录最终结果"""
    if logger:
        logger.info("="*80)
        logger.info("FINAL TRAINING RESULTS")
        logger.info("="*80)
        logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
        logger.info("Final HyperNetwork Z-signals:")
        for name, signal in z_signals.items():
            logger.info(f"  {name}: {signal}")
        logger.info("Training completed successfully!")
        logger.info("="*80)
    
    # 更新JSON文件的最终结果
    if json_file:
        log_training_progress(json_file, -1, 0, 0, 0, 0, 0, test_acc)
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            data['final_results'] = {
                'test_accuracy': float(test_acc / 100),
                'z_signals': z_signals,
                'completion_time': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save final results to JSON: {e}")

def cleanup():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()

def set_device(rank):
    """设置设备，rank对应GPU ID"""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
        # 清理GPU内存
        torch.cuda.empty_cache()
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

class HyperCNNCIFAR_AllLayers_Stable(nn.Module):
    """稳定版全层HyperNetwork CIFAR-100分类器 - 减小复杂度"""
    
    def __init__(self, f_size=3, in_size=32, out_size=64, z_dim=8):  # 减小参数
        super(HyperCNNCIFAR_AllLayers_Stable, self).__init__()
        
        self.f_size = f_size
        self.in_size = in_size      
        self.out_size = out_size    
        self.z_dim = z_dim          
        
        # 固定组件 - 轻度正则化
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)   
        self.dropout2 = nn.Dropout(0.2)   
        self.dropout3 = nn.Dropout(0.25)   
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(in_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.bn3 = nn.BatchNorm2d(128)  # 减小
        self.bn_fc1 = nn.BatchNorm1d(256)  # 减小
        
        # Z信号参数 - 减小维度
        self.z_signal_conv1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_conv2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_conv3 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_fc1 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        self.z_signal_fc2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        
        # HyperNetwork生成器 - 减小规模
        conv1_weight_size = in_size * 3 * f_size * f_size
        self.hyper_conv1 = nn.Linear(z_dim, conv1_weight_size)
        self.conv1_bias = nn.Parameter(torch.zeros(in_size))
        
        conv2_weight_size = out_size * in_size * f_size * f_size
        self.hyper_conv2 = nn.Linear(z_dim, conv2_weight_size)
        self.conv2_bias = nn.Parameter(torch.zeros(out_size))
        
        conv3_weight_size = 128 * out_size * 3 * 3  # 减小
        self.hyper_conv3 = nn.Linear(z_dim, conv3_weight_size)
        self.conv3_bias = nn.Parameter(torch.zeros(128))
        
        fc1_input_size = 128 * 4 * 4  # 减小
        fc1_weight_size = 256 * fc1_input_size  # 减小
        self.hyper_fc1 = nn.Linear(z_dim, fc1_weight_size)
        self.fc1_bias = nn.Parameter(torch.zeros(256))
        
        fc2_weight_size = 100 * 256  # 减小
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
            nn.init.kaiming_normal_(hyper_net.weight, mode='fan_out', nonlinearity='relu')
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
        
        # FC2块 + 最终Dropout
        fc2_weights = self.generate_fc2_weights()
        x = self.dropout3(x)
        x = F.linear(x, fc2_weights, bias=self.fc2_bias)
        
        return x

def train_epoch_stable(model, device, train_loader, optimizer, epoch, rank, world_size, log_interval=50):
    """稳定版训练函数 - 减少通信频率"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # 累积梯度以减少通信频率
    accumulation_steps = 2
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # 前向传播
        output = model(data)
        loss = F.cross_entropy(output, target) / accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 累积梯度后再更新
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()
            
            # 清理中间变量
            torch.cuda.empty_cache()
        
        # 指标跟踪
        train_loss += loss.item() * accumulation_steps
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 只在主进程打印日志，减少频率
        if rank == 0 and batch_idx % log_interval == 0:
            current_acc = 100. * correct / total
            print(f'  Batch: {batch_idx:3d}/{len(train_loader)} | '
                  f'Loss: {loss.item() * accumulation_steps:.4f} | '
                  f'Acc: {current_acc:6.2f}% | GPU {rank}')
    
    train_loss /= len(train_loader)
    train_acc = correct / total
    train_err = 1.0 - train_acc
    
    return train_loss, train_err

def evaluate_stable(model, device, data_loader, rank):
    """稳定版评估函数 - 优化通信"""
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
            
            # 定期清理内存
            if total % 1000 == 0:
                torch.cuda.empty_cache()
    
    # 聚合所有GPU的结果 - 使用更稳定的方式
    try:
        loss_tensor = torch.tensor(loss, dtype=torch.float32).to(device)
        correct_tensor = torch.tensor(correct, dtype=torch.int64).to(device)
        total_tensor = torch.tensor(total, dtype=torch.int64).to(device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = loss_tensor.item() / total_tensor.item()
        total_acc = correct_tensor.item() / total_tensor.item()
        total_err = 1.0 - total_acc
        
    except Exception as e:
        if rank == 0:
            print(f"Warning: Communication error in evaluation: {e}")
        # 降级为本地结果
        total_loss = loss / total if total > 0 else 0
        total_acc = correct / total if total > 0 else 0
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
    """稳定版分布式训练工作进程 - 带日志功能"""
    
    # 设置日志系统（只在主进程）
    logger, json_file = setup_logging(rank)
    
    try:
        # 设置分布式训练
        setup(rank, world_size)
        
        # 设置设备
        device = set_device(rank)
        
        # 随机种子设置
        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)
        random.seed(42 + rank)
        
        # 超参数配置 - 稳定版本
        batch_size = 64  # 减小batch size
        test_batch_size = 128
        lr = 0.0005  # 降低学习率
        f_size = 3   # 减小卷积核
        in_size = 32 # 减小通道数
        out_size = 64
        z_dim = 8    # 减小Z维度
        weight_decay = 1e-4  
        
        # 训练配置字典
        config = {
            'world_size': world_size,
            'batch_size_per_gpu': batch_size,
            'effective_batch_size': batch_size * world_size,
            'epochs': epochs,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'z_dimension': z_dim,
            'filter_size': f_size,
            'in_channels': in_size,
            'out_channels': out_size,
            'strategy': 'Stable Training with NCCL fixes',
            'model_type': 'HyperNetwork ALL LAYERS',
            'dataset': 'CIFAR-100'
        }
        
        # 记录配置
        log_config(logger, json_file, config)
        
        if rank == 0:
            print(f"Stable Multi-GPU Training Configuration:")
            print(f"  World size (GPUs): {world_size}")
            print(f"  Batch size per GPU: {batch_size}")
            print(f"  Effective batch size: {batch_size * world_size}")
            print(f"  Epochs: {epochs}")
            print(f"  Learning rate: {lr}")
            print(f"  Weight decay: {weight_decay}")
            print(f"  Z-dimension: {z_dim}")
            print(f"  Strategy: Stable Training")
            print("="*70)
        
        # 简化的数据预处理
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
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
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, 5000])
        
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
        
        # 数据加载器 - 减少worker数量
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=2, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=test_batch_size, sampler=val_sampler,
            num_workers=1, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=test_batch_size, sampler=test_sampler,
            num_workers=1, pin_memory=True
        )
        
        if rank == 0:
            print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            print(f"Batches per GPU per epoch: {len(train_loader)}")
            print("="*70)
        
        # 模型创建
        model = HyperCNNCIFAR_AllLayers_Stable(
            f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim
        ).to(device)
        
        # DDP包装 - 优化设置
        model = DDP(
            model, 
            device_ids=[rank], 
            output_device=rank, 
            find_unused_parameters=False,
            broadcast_buffers=True,
            bucket_cap_mb=10  # 减小bucket大小
        )
        
        # 优化器和调度器
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # 参数统计（只在主进程）
        total_params, hyper_params = 0, 0
        if rank == 0:
            print("\nStable ALL LAYERS HyperNetwork CNN:")
            total_params, hyper_params = count_parameters(model, rank)
            log_model_info(logger, total_params, hyper_params)
            print("="*70)
        
        # 训练循环
        best_val_err = 1.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 设置采样器的epoch
            train_sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"\nEpoch {epoch+1}/{epochs} - Stable HyperNetwork CNN")
                print("-" * 60)
            
            # 训练
            try:
                train_loss, train_err = train_epoch_stable(
                    model, device, train_loader, optimizer, epoch, rank, world_size
                )
                
                # 验证
                val_loss, val_err = evaluate_stable(model, device, val_loader, rank)
                
                # 更新学习率
                scheduler.step()
                
                # 保存最佳模型（只在主进程）
                if rank == 0:
                    if val_err < best_val_err:
                        best_val_err = val_err
                        torch.save(model.module.state_dict(), 'cifar100_all_layers_hyper_stable_best.pt')
                        print(f"    *** New best validation error: {100*val_err:.2f}% - Model saved! ***")
                        if logger:
                            logger.info(f"New best validation error: {100*val_err:.2f}% - Model saved!")
                    
                    # 打印结果
                    epoch_time = time.time() - start_time
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    print(f"\nResults: Train Err: {100*train_err:.2f}% | Val Err: {100*val_err:.2f}% | "
                          f"Best Val Err: {100*best_val_err:.2f}% | LR: {current_lr:.6f}")
                    print(f"Time: {epoch_time:.1f}s")
                    
                    # 记录训练进度到日志和JSON
                    if logger:
                        logger.info(f"Epoch {epoch+1}/{epochs} completed:")
                        logger.info(f"  Train Error: {100*train_err:.2f}%")
                        logger.info(f"  Validation Error: {100*val_err:.2f}%")
                        logger.info(f"  Best Validation Error: {100*best_val_err:.2f}%")
                        logger.info(f"  Learning Rate: {current_lr:.6f}")
                        logger.info(f"  Epoch Time: {epoch_time:.1f}s")
                    
                    # 保存进度到JSON文件
                    log_training_progress(json_file, epoch+1, train_err, val_err, best_val_err, current_lr, epoch_time)
                    
                    # 内存清理
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if rank == 0:
                    print(f"Error in epoch {epoch+1}: {e}")
                # 继续下一个epoch
                continue
        
        # 最终评估
        if rank == 0:
            print("\n" + "="*70)
            print("FINAL RESULTS - Stable Version")
            print("="*70)
            
            # 加载最佳模型
            try:
                model.module.load_state_dict(torch.load('cifar100_all_layers_hyper_stable_best.pt'))
            except:
                print("Warning: Could not load best model, using current model")
                
        # 所有进程参与最终测试
        try:
            test_loss, test_err = evaluate_stable(model, device, test_loader, rank)
            test_acc = (1 - test_err) * 100
            
            if rank == 0:
                print(f"Stable ALL LAYERS HyperNetwork - Test Accuracy: {test_acc:.2f}%")
                
                # Z信号最终状态
                z_signals = {
                    'z_signal_conv1': model.module.z_signal_conv1.detach().cpu().numpy().flatten().tolist(),
                    'z_signal_conv2': model.module.z_signal_conv2.detach().cpu().numpy().flatten().tolist(),
                    'z_signal_conv3': model.module.z_signal_conv3.detach().cpu().numpy().flatten().tolist(),
                    'z_signal_fc1': model.module.z_signal_fc1.detach().cpu().numpy().flatten().tolist(),
                    'z_signal_fc2': model.module.z_signal_fc2.detach().cpu().numpy().flatten().tolist()
                }
                
                print(f"\nFinal hypernetwork signals:")
                for name, signal in z_signals.items():
                    print(f"{name}: {signal}")
                
                print(f"\nUsed {world_size} GPUs for stable training")
                print("Stable Multi-GPU training completed!")
                
                # 记录最终结果到日志
                log_final_results(logger, json_file, test_acc, z_signals)
                
        except Exception as e:
            if rank == 0:
                print(f"Error in final evaluation: {e}")
    
    except Exception as e:
        print(f"Rank {rank}: Fatal error: {e}")
    
    finally:
        # 清理
        cleanup()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIFAR-100 Stable Multi-GPU HyperNetwork Training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
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
    
    print("Starting CIFAR-100 ALL LAYERS HyperNetwork - Stable Multi-GPU Version")
    print(f"Using {world_size} GPUs: {[torch.cuda.get_device_name(i) for i in range(world_size)]}")
    print(f"Training for {args.epochs} epochs")
    print("Objective: Stable long-term training without NCCL timeouts")
    print("="*70)
    
    # 启动多进程训练
    try:
        mp.spawn(train_worker, args=(world_size, args.epochs), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
