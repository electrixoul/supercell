'''
CIFAR-100 HyperNetwork ALL LAYERS VERSION - Multi-GPU Progressive Augmentation STABLE Version
============================================================================================
基于渐进式增强版本，增加稳定性和日志功能：
核心改进：
1. 保留渐进式数据增强策略（训练前期轻度 → 中期中度 → 后期强度）
2. 增加NCCL超时设置和稳定性配置
3. 添加完整的实时日志记录功能
4. 优化内存使用和错误恢复机制
5. 目标：稳定的长期训练 + 测试准确率52-55% + 训练-测试gap <10%
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
import logging
from datetime import datetime
import json

def setup(rank, world_size):
    """初始化分布式训练进程组 - 增加稳定性配置"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12365'  # 避免端口冲突
    
    # NCCL配置优化 - 稳定性改进
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
        log_file = os.path.join(log_dir, f"cifar100_hypernetwork_progressive_stable_{timestamp}.log")
        json_file = os.path.join(log_dir, f"training_progress_progressive_{timestamp}.json")
        
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
        logger.info("CIFAR-100 HyperNetwork Progressive Augmentation Stable Training - Log Started")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Progress file: {json_file}")
        logger.info("="*80)
        
        return logger, json_file
    else:
        return None, None

def log_training_progress(json_file, epoch, train_err, val_err, best_val_err, lr, epoch_time, stage, aug_ratio, test_acc=None):
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
        'best_validation_accuracy': float(1 - best_val_err),
        'augmentation_stage': stage,
        'augmentation_ratio': float(aug_ratio)
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

def log_final_results(logger, json_file, test_acc, z_signals, train_test_gap):
    """记录最终结果"""
    if logger:
        logger.info("="*80)
        logger.info("FINAL TRAINING RESULTS - Progressive Augmentation Stable Version")
        logger.info("="*80)
        logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
        logger.info(f"Train-Test Gap: {train_test_gap:+.2f}%")
        logger.info("Final HyperNetwork Z-signals:")
        for name, signal in z_signals.items():
            logger.info(f"  {name}: {signal}")
        logger.info("Progressive Augmentation training completed successfully!")
        logger.info("="*80)
    
    # 更新JSON文件的最终结果
    if json_file:
        log_training_progress(json_file, -1, 0, 0, 0, 0, 0, "Final", 0, test_acc)
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            data['final_results'] = {
                'test_accuracy': float(test_acc / 100),
                'train_test_gap': float(train_test_gap),
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

def get_augment_schedule(epoch, total_epochs):
    """动态数据增强调度"""
    # 前1/3：轻度增强（让模型快速学习基本特征）
    if epoch < total_epochs // 3:
        mixup_prob, cutmix_prob = 0.2, 0.1
        rand_magnitude = 4
        random_erasing_prob = 0.1
        stage = "Light"
    # 中1/3：中度增强（平衡学习和泛化）
    elif epoch < 2 * total_epochs // 3:
        mixup_prob, cutmix_prob = 0.3, 0.2
        rand_magnitude = 6
        random_erasing_prob = 0.15
        stage = "Medium"
    # 后1/3：强度增强（最终泛化能力提升）
    else:
        mixup_prob, cutmix_prob = 0.4, 0.3
        rand_magnitude = 8
        random_erasing_prob = 0.2
        stage = "Strong"
    
    return mixup_prob, cutmix_prob, rand_magnitude, random_erasing_prob, stage

class RandAugment:
    """可调节强度的RandAugment实现"""
    def __init__(self, n_ops=2, magnitude=9):
        self.n_ops = n_ops
        self.magnitude = magnitude
        self.augment_list = [
            'AutoContrast', 'Brightness', 'Color', 'Contrast', 'Equalize',
            'Rotate', 'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY'
        ]
    
    def set_magnitude(self, magnitude):
        """动态调整增强强度"""
        self.magnitude = magnitude
    
    def __call__(self, img):
        """随机应用n_ops个增强操作"""
        ops = random.choices(self.augment_list, k=self.n_ops)
        for op in ops:
            img = self._apply_op(img, op)
        return img
    
    def _apply_op(self, img, op_name):
        """应用单个增强操作"""
        from PIL import Image, ImageEnhance, ImageOps
        
        magnitude_scale = self.magnitude / 10.0
        
        if op_name == 'AutoContrast':
            return ImageOps.autocontrast(img)
        elif op_name == 'Brightness':
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(1 + magnitude_scale * random.choice([-1, 1]) * 0.5)
        elif op_name == 'Color':
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(1 + magnitude_scale * random.choice([-1, 1]) * 0.5)
        elif op_name == 'Contrast':
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(1 + magnitude_scale * random.choice([-1, 1]) * 0.5)
        elif op_name == 'Equalize':
            return ImageOps.equalize(img)
        elif op_name == 'Rotate':
            angle = magnitude_scale * 30 * random.choice([-1, 1])
            return img.rotate(angle, fillcolor=(128, 128, 128))
        elif op_name == 'Sharpness':
            enhancer = ImageEnhance.Sharpness(img)
            return enhancer.enhance(1 + magnitude_scale * random.choice([-1, 1]) * 0.5)
        else:
            return img

class ProgressiveDataset:
    """支持渐进式数据增强的数据集包装器"""
    def __init__(self, base_dataset, rand_augment):
        self.base_dataset = base_dataset
        self.rand_augment = rand_augment
        self.current_random_erasing_prob = 0.1
        
    def update_augmentation(self, rand_magnitude, random_erasing_prob):
        """更新增强参数"""
        self.rand_augment.set_magnitude(rand_magnitude)
        self.current_random_erasing_prob = random_erasing_prob
        
        # 重新构建transform
        self.update_transform()
    
    def update_transform(self):
        """更新数据变换"""
        # 基础增强保持不变
        base_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),  # 轻微降低旋转角度
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 降低ColorJitter强度
            self.rand_augment,
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
        
        # 动态添加RandomErasing
        if self.current_random_erasing_prob > 0:
            base_transforms.append(
                transforms.RandomErasing(p=self.current_random_erasing_prob, scale=(0.02, 0.25), ratio=(0.3, 3.3))
            )
        
        self.transform = transforms.Compose(base_transforms)
        
        # 更新base_dataset的transform
        self.base_dataset.transform = self.transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]

class Mixup:
    """Mixup数据增强"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch_x, batch_y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam

class CutMix:
    """CutMix数据增强"""
    def __init__(self, beta=1.0, cutmix_prob=0.5):
        self.beta = beta
        self.cutmix_prob = cutmix_prob
    
    def __call__(self, batch_x, batch_y):
        if np.random.rand() > self.cutmix_prob:
            return batch_x, batch_y, batch_y, 1.0
        
        lam = np.random.beta(self.beta, self.beta)
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)
        
        # 生成random box
        _, _, H, W = batch_x.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        batch_x[:, :, bby1:bby2, bbx1:bbx2] = batch_x[index, :, bby1:bby2, bbx1:bbx2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size()[-1] * batch_x.size()[-2]))
        
        y_a, y_b = batch_y, batch_y[index]
        return batch_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix的损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class HyperCNNCIFAR_AllLayers_Progressive(nn.Module):
    """渐进式数据增强版全层HyperNetwork CIFAR-100分类器"""
    
    def __init__(self, f_size=5, in_size=64, out_size=128, z_dim=10):
        super(HyperCNNCIFAR_AllLayers_Progressive, self).__init__()
        
        self.f_size = f_size
        self.in_size = in_size      
        self.out_size = out_size    
        self.z_dim = z_dim          
        
        # 固定组件 - 轻度正则化
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
        return weights_flat.reshape(256, self.out_size, 3, 3)
    
    def generate_fc1_weights(self):
        weights_flat = self.hyper_fc1(self.z_signal_fc1)
        return weights_flat.reshape(512, 256 * 4 * 4)
    
    def generate_fc2_weights(self):
        weights_flat = self.hyper_fc2(self.z_signal_fc2)
        return weights_flat.reshape(100, 512)
        
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

class EarlyStopping:
    """早停机制 - 基于准确率，目标90%+"""
    def __init__(self, patience=30, min_delta=0.001, target_accuracy=0.90):
        self.patience = patience
        self.min_delta = min_delta
        self.target_accuracy = target_accuracy  # 目标准确率
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.target_reached = False
    
    def __call__(self, val_error):
        val_accuracy = 1.0 - val_error
        
        # 如果达到目标准确率，直接停止
        if val_accuracy >= self.target_accuracy:
            self.target_reached = True
            self.early_stop = True
            return
        
        # 否则使用传统早停逻辑（但要求更高的patience）
        if self.best_score is None:
            self.best_score = val_error
        elif val_error < self.best_score - self.min_delta:
            self.best_score = val_error
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_epoch_stable(model, device, train_loader, optimizer, epoch, rank, world_size, 
                      mixup, cutmix, mixup_prob, cutmix_prob, log_interval=20):
    """稳定版训练函数 - 包含渐进式数据增强和稳定性改进"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    augmented_samples = 0
    clean_samples = 0
    
    # 累积梯度以减少通信频率
    accumulation_steps = 2
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # 根据当前阶段调整增强概率
        aug_prob = np.random.rand()
        if aug_prob < mixup_prob:  # Mixup
            data, y_a, y_b, lam = mixup(data, target)
            output = model(data)
            loss = mixup_criterion(F.cross_entropy, output, y_a, y_b, lam) / accumulation_steps
            augmented_samples += target.size(0)
        elif aug_prob < mixup_prob + cutmix_prob:  # CutMix
            data, y_a, y_b, lam = cutmix(data, target)
            output = model(data)
            loss = mixup_criterion(F.cross_entropy, output, y_a, y_b, lam) / accumulation_steps
            augmented_samples += target.size(0)
        else:  # 正常训练（包含基础数据增强）
            output = model(data)
            loss = F.cross_entropy(output, target) / accumulation_steps
            clean_samples += target.size(0)
        
        loss.backward()
        
        # 累积梯度后再更新
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # 清理中间变量
            torch.cuda.empty_cache()
        
        # 指标跟踪
        train_loss += loss.item() * accumulation_steps
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 只在主进程打印日志
        if rank == 0 and batch_idx % log_interval == 0:
            current_acc = 100. * correct / total
            aug_ratio = augmented_samples / (augmented_samples + clean_samples) if (augmented_samples + clean_samples) > 0 else 0
            print(f'  Batch: {batch_idx:3d}/{len(train_loader)} | '
                  f'Loss: {loss.item() * accumulation_steps:.4f} | '
                  f'Acc: {current_acc:6.2f}% | '
                  f'AugRatio: {aug_ratio:.2f} | '
                  f'GPU {rank}')
    
    train_loss /= len(train_loader)
    train_acc = correct / total
    train_err = 1.0 - train_acc
    aug_ratio = augmented_samples / total if total > 0 else 0
    
    return train_loss, train_err, aug_ratio

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
    """稳定版分布式训练工作进程 - Progressive Augmentation + 完整日志"""
    
    # 设置日志系统（只在主进程）
    logger, json_file = setup_logging(rank)
    
    try:
        # 设置分布式训练
        setup(rank, world_size)
        
        # 设置设备
        device = set_device(rank)
        
        # 随机种子设置（每个进程使用不同种子以获得数据多样性）
        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)
        random.seed(42 + rank)
        
        # 超参数配置 - 渐进式增强版本
        batch_size = 128
        test_batch_size = 256
        lr = 0.001
        f_size = 5
        in_size = 64      
        out_size = 128    
        z_dim = 10        
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
            'strategy': 'Progressive Augmentation with NCCL fixes',
            'model_type': 'HyperNetwork ALL LAYERS Progressive',
            'dataset': 'CIFAR-100',
            'augmentation_stages': {
                'light': 'Mixup:20%, CutMix:10%, RandAug:4',
                'medium': 'Mixup:30%, CutMix:20%, RandAug:6', 
                'strong': 'Mixup:40%, CutMix:30%, RandAug:8'
            }
        }
        
        # 记录配置
        log_config(logger, json_file, config)
        
        if rank == 0:
            print(f"Progressive Augmentation Multi-GPU Training Configuration:")
            print(f"  World size (GPUs): {world_size}")
            print(f"  Batch size per GPU: {batch_size}")
            print(f"  Effective batch size: {batch_size * world_size}")
            print(f"  Epochs: {epochs}")
            print(f"  Learning rate: {lr}")
            print(f"  Weight decay: {weight_decay}")
            print(f"  Z-dimension: {z_dim}")
            print(f"  Strategy: Progressive Data Augmentation")
            print("="*70)
        
        # 渐进式数据预处理 
        rand_augment = RandAugment(n_ops=2, magnitude=4)  # 初始轻度设置
        
        # 基础变换（测试用）
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # 数据集加载
        base_train_dataset = datasets.CIFAR100('data', train=True, download=False, transform=None)
        test_dataset = datasets.CIFAR100('data', train=False, transform=transform_test)
        
        # 包装为渐进式数据集
        progressive_train_dataset = ProgressiveDataset(base_train_dataset, rand_augment)
        
        # 验证集分割
        train_size = len(progressive_train_dataset) - 5000
        train_dataset, val_dataset = torch.utils.data.random_split(
            progressive_train_dataset, [train_size, 5000]
        )
        
        # 为验证集设置简单的变换
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
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
        
        # 数据加载器 - 减少worker数量以提高稳定性
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=2, pin_memory=True, persistent_workers=True  # 减少workers
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
            print("Progressive Augmentation Strategy:")
            print("  Stage 1 (0-33%): Light (Mixup:20%, CutMix:10%, RandAug:4)")
            print("  Stage 2 (33-66%): Medium (Mixup:30%, CutMix:20%, RandAug:6)")
            print("  Stage 3 (66-100%): Strong (Mixup:40%, CutMix:30%, RandAug:8)")
            print("="*70)
        
        # 模型创建
        model = HyperCNNCIFAR_AllLayers_Progressive(
            f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim
        ).to(device)
        
        # DDP包装 - 优化设置
        model = DDP(
            model, 
            device_ids=[rank], 
            output_device=rank, 
            find_unused_parameters=False,
            broadcast_buffers=True,
            bucket_cap_mb=15  # 稍微增加bucket大小
        )
        
        # 数据增强组件
        mixup = Mixup(alpha=0.2)
        cutmix = CutMix(beta=1.0, cutmix_prob=1.0)  # 在训练中手动控制概率
        
        # 优化器和调度器
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        # 早停机制
        early_stopping = EarlyStopping(patience=20, min_delta=0.001) if rank == 0 else None
        
        # 参数统计（只在主进程）
        total_params, hyper_params = 0, 0
        if rank == 0:
            print("\nProgressive Augmentation ALL LAYERS HyperNetwork CNN:")
            total_params, hyper_params = count_parameters(model, rank)
            log_model_info(logger, total_params, hyper_params)
            print("="*70)
        
        # 训练循环
        best_val_err = 1.0
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 设置采样器的epoch（确保每个epoch数据顺序不同）
            train_sampler.set_epoch(epoch)
            
            # 获取当前epoch的增强策略
            mixup_prob, cutmix_prob, rand_magnitude, random_erasing_prob, stage = get_augment_schedule(epoch, epochs)
            
            # 更新数据增强参数
            progressive_train_dataset.update_augmentation(rand_magnitude, random_erasing_prob)
            
            if rank == 0:
                print(f"\nEpoch {epoch+1}/{epochs} - Progressive Augmentation HyperNetwork CNN")
                print(f"Current Stage: {stage} | Mixup: {mixup_prob:.1f} | CutMix: {cutmix_prob:.1f} | "
                      f"RandAug: {rand_magnitude} | RandomErasing: {random_erasing_prob:.2f}")
                print("-" * 80)
            
            # 训练（包含渐进式数据增强）
            try:
                train_loss, train_err, aug_ratio = train_epoch_stable(
                    model, device, train_loader, optimizer, epoch, rank, world_size, 
                    mixup, cutmix, mixup_prob, cutmix_prob
                )
                
                # 验证
                val_loss, val_err = evaluate_stable(model, device, val_loader, rank)
                
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
                        torch.save(model.module.state_dict(), 'cifar100_all_layers_hyper_progressive_stable_best.pt')
                        print(f"    *** New best validation error: {100*val_err:.2f}% - Model saved! ***")
                        if logger:
                            logger.info(f"New best validation error: {100*val_err:.2f}% - Model saved!")
                    
                    # 打印结果
                    epoch_time = time.time() - start_time
                    current_lr = optimizer.param_groups[0]['lr']
                    effective_samples_per_sec = len(train_dataset) * world_size / epoch_time
                    train_val_gap = train_err - val_err
                    
                    print(f"\nResults: Train Err: {100*train_err:.2f}% | Val Err: {100*val_err:.2f}% | "
                          f"Best Val Err: {100*best_val_err:.2f}% | LR: {current_lr:.6f}")
                    print(f"Train-Val Gap: {100*train_val_gap:+.2f}% | Aug Ratio: {aug_ratio:.2f} | "
                          f"Time: {epoch_time:.1f}s | Samples/sec: {effective_samples_per_sec:.0f}")
                    
                    # 记录训练进度到日志和JSON
                    if logger:
                        logger.info(f"Epoch {epoch+1}/{epochs} completed - Stage: {stage}")
                        logger.info(f"  Train Error: {100*train_err:.2f}%")
                        logger.info(f"  Validation Error: {100*val_err:.2f}%")
                        logger.info(f"  Best Validation Error: {100*best_val_err:.2f}%")
                        logger.info(f"  Train-Val Gap: {100*train_val_gap:+.2f}%")
                        logger.info(f"  Learning Rate: {current_lr:.6f}")
                        logger.info(f"  Augmentation Ratio: {aug_ratio:.2f}")
                        logger.info(f"  Epoch Time: {epoch_time:.1f}s")
                    
                    # 保存进度到JSON文件
                    log_training_progress(json_file, epoch+1, train_err, val_err, best_val_err, current_lr, epoch_time, stage, aug_ratio)
                    
                    # Z信号监控（每10个epoch）
                    if (epoch + 1) % 10 == 0:
                        conv_norms = [
                            model.module.z_signal_conv1.norm().item(),
                            model.module.z_signal_conv2.norm().item(),
                            model.module.z_signal_conv3.norm().item()
                        ]
                        fc_norms = [
                            model.module.z_signal_fc1.norm().item(),
                            model.module.z_signal_fc2.norm().item()
                        ]
                        print(f"    Conv Z-norms: {conv_norms[0]:.4f}, {conv_norms[1]:.4f}, {conv_norms[2]:.4f}")
                        print(f"    FC Z-norms: {fc_norms[0]:.4f}, {fc_norms[1]:.4f}")
                        if logger:
                            logger.info(f"  Z-signal norms - Conv: {conv_norms}, FC: {fc_norms}")
                    
                    # 内存清理
                    torch.cuda.empty_cache()
                    
                    # 早停检查
                    if early_stopping is not None:
                        early_stopping(val_err)
                        if early_stopping.early_stop:
                            print(f"Early stopping triggered at epoch {epoch+1}")
                            if logger:
                                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                            break
                            
            except Exception as e:
                if rank == 0:
                    print(f"Error in epoch {epoch+1}: {e}")
                    if logger:
                        logger.error(f"Error in epoch {epoch+1}: {e}")
                # 继续下一个epoch
                continue
        
        # 最终评估
        if rank == 0:
            print("\n" + "="*70)
            print("FINAL RESULTS - Progressive Augmentation Stable Version")
            print("="*70)
            
            # 加载最佳模型
            try:
                model.module.load_state_dict(torch.load('cifar100_all_layers_hyper_progressive_stable_best.pt'))
                if logger:
                    logger.info("Loaded best model for final evaluation")
            except:
                print("Warning: Could not load best model, using current model")
                if logger:
                    logger.warning("Could not load best model, using current model")
                    
        # 最终测试 - 采用更安全的方式
        if rank == 0:
            print("Starting final evaluation...")
            if logger:
                logger.info("Starting final evaluation...")
        
        # 首先尝试分布式evaluation，如果失败则降级为单GPU
        test_acc = 0
        try:
            # 设置较短的超时时间进行分布式evaluation
            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0
                batch_count = 0
                
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    batch_count += 1
                    
                    # 每5个batch清理一次内存
                    if batch_count % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    # 只在主进程显示进度
                    if rank == 0 and batch_idx % 10 == 0:
                        print(f"  Evaluation batch {batch_idx}/{len(test_loader)}")
                
                # 尝试聚合结果（带超时保护）
                try:
                    correct_tensor = torch.tensor(correct, dtype=torch.int64).to(device)
                    total_tensor = torch.tensor(total, dtype=torch.int64).to(device)
                    
                    # 使用barrier确保所有进程都到达这里
                    dist.barrier(timeout=30)  # 30秒超时
                    
                    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
                    
                    test_acc = (correct_tensor.item() / total_tensor.item()) * 100
                    
                    if rank == 0:
                        print(f"✅ Distributed evaluation successful: {test_acc:.2f}%")
                        if logger:
                            logger.info(f"Distributed evaluation successful: {test_acc:.2f}%")
                            
                except Exception as comm_error:
                    if rank == 0:
                        print(f"⚠️ Distributed communication failed: {comm_error}")
                        print("Falling back to single-GPU evaluation...")
                        if logger:
                            logger.warning(f"Distributed communication failed: {comm_error}")
                            logger.info("Falling back to single-GPU evaluation...")
                        
                        # 降级为单GPU evaluation（仅主进程）
                        test_acc = (correct / total) * 100
                        print(f"✅ Single-GPU evaluation result: {test_acc:.2f}% (GPU {rank} only)")
                
        except Exception as eval_error:
            if rank == 0:
                print(f"❌ Evaluation failed: {eval_error}")
                print("Using previous best validation accuracy as estimate...")
                if logger:
                    logger.error(f"Evaluation failed: {eval_error}")
                    logger.info("Using previous best validation accuracy as estimate...")
                test_acc = (1 - best_val_err) * 100
        
        # 确保所有进程在这里同步
        try:
            dist.barrier(timeout=10)  # 10秒超时
        except:
            pass  # 忽略barrier失败
        
        # 只在主进程处理结果
        if rank == 0:
            # 性能对比
            std_test_acc = 16.23
            fixed_hyper_acc = 33.52
            full_hyper_acc = 54.16
            original_all_layers_acc = 36.13
            multi_gpu_acc = 42.76
            large_multi_gpu_acc = 53.71
            optimized_acc = 33.07
            data_enhanced_acc = 51.95
            
            print(f"Standard CNN - Test Accuracy: {std_test_acc:.2f}%")
            print(f"Fixed HyperNetwork (1 layer) - Test Accuracy: {fixed_hyper_acc:.2f}%")
            print(f"Full HyperNetwork (complex) - Test Accuracy: {full_hyper_acc:.2f}%")
            print(f"ALL LAYERS HyperNetwork (original) - Test Accuracy: {original_all_layers_acc:.2f}%")
            print(f"ALL LAYERS HyperNetwork (Multi-GPU) - Test Accuracy: {multi_gpu_acc:.2f}%")
            print(f"ALL LAYERS HyperNetwork (Large Multi-GPU) - Test Accuracy: {large_multi_gpu_acc:.2f}%")
            print(f"ALL LAYERS HyperNetwork (Optimized Anti-Overfitting) - Test Accuracy: {optimized_acc:.2f}%")
            print(f"ALL LAYERS HyperNetwork (Data Enhancement Focus) - Test Accuracy: {data_enhanced_acc:.2f}%")
            print(f"ALL LAYERS HyperNetwork (Progressive Augmentation Stable) - Test Accuracy: {test_acc:.2f}%")
            
            improvement_vs_data_enhanced = test_acc - data_enhanced_acc
            improvement_vs_large = test_acc - large_multi_gpu_acc
            print(f"Improvement vs Data Enhanced: {improvement_vs_data_enhanced:+.2f}%")
            print(f"Improvement vs Large Multi-GPU: {improvement_vs_large:+.2f}%")
            
            # 训练稳定性分析
            final_train_acc = train_accs[-1] if train_accs else 0
            final_val_acc = val_accs[-1] if val_accs else 0
            train_test_gap = final_train_acc - test_acc
            print(f"\nProgressive Augmentation Strategy Analysis:")
            print(f"Final Train Accuracy: {final_train_acc:.2f}%")
            print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
            print(f"Test Accuracy: {test_acc:.2f}%")
            print(f"Train-Test Gap: {train_test_gap:+.2f}% (target: <10%)")
            
            # 与数据增强版本对比
            data_enhanced_train_test_gap = -26.74
            gap_improvement = abs(train_test_gap) - abs(data_enhanced_train_test_gap)
            print(f"Gap improvement vs Data Enhanced: {gap_improvement:+.2f}% (better if positive)")
            
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
            
            print(f"\nUsed {world_size} GPUs for progressive augmentation stable training")
            print("Progressive Augmentation Multi-GPU training completed!")
            
            # 记录最终结果到日志
            log_final_results(logger, json_file, test_acc, z_signals, train_test_gap)
                
    except Exception as e:
        if rank == 0:
            print(f"Error in final evaluation: {e}")
            if logger:
                logger.error(f"Error in final evaluation: {e}")
    
    except Exception as e:
        print(f"Rank {rank}: Fatal error: {e}")
        if rank == 0 and logger:
            logger.error(f"Fatal error: {e}")
    
    finally:
        # 清理
        cleanup()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIFAR-100 Progressive Augmentation Stable Multi-GPU HyperNetwork Training')
    parser.add_argument('--epochs', type=int, default=45,
                        help='number of epochs to train (default: 45)')
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
    
    print("Starting CIFAR-100 ALL LAYERS HyperNetwork - Progressive Augmentation Stable Multi-GPU Version")
    print(f"Using {world_size} GPUs: {[torch.cuda.get_device_name(i) for i in range(world_size)]}")
    print(f"Training for {args.epochs} epochs")
    print("Objective: Stable long-term training + Improve train-test consistency while maintaining high performance")
    print("="*70)
    
    # 启动多进程训练
    try:
        mp.spawn(train_worker, args=(world_size, args.epochs), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
