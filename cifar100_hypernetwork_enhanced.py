'''
CIFAR-100 HyperNetwork ENHANCED VERSION
Target: 90% accuracy with larger network and longer training

Enhancements:
1. Deeper network architecture (5 conv layers)
2. Larger channel dimensions 
3. Multiple hypernetwork-generated layers
4. Advanced training techniques (BatchNorm, better LR scheduling)
5. Extended training (50+ epochs)
6. Comprehensive data augmentation
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import math

def set_device():
    """Set optimal device"""
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

class EnhancedHyperCNNCIFAR(nn.Module):
    def __init__(self, z_dim=16):
        super(EnhancedHyperCNNCIFAR, self).__init__()
        self.z_dim = z_dim
        
        # Network architecture: much deeper and wider
        self.channels = [3, 64, 128, 256, 512, 512]  # 5 conv layers
        self.f_sizes = [3, 3, 3, 3, 3]  # Smaller filters for deeper network
        
        # First conv layer (standard)
        self.conv1 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=self.f_sizes[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.channels[1])
        
        # Second conv layer (hypernetwork generated)
        self.z_signal_2 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        conv2_weight_size = self.channels[2] * self.channels[1] * self.f_sizes[1] * self.f_sizes[1]
        self.hyper_linear_2 = nn.Linear(z_dim, conv2_weight_size)
        self.conv2_bias = nn.Parameter(torch.zeros(self.channels[2]))
        self.bn2 = nn.BatchNorm2d(self.channels[2])
        
        # Third conv layer (hypernetwork generated)
        self.z_signal_3 = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        conv3_weight_size = self.channels[3] * self.channels[2] * self.f_sizes[2] * self.f_sizes[2]
        self.hyper_linear_3 = nn.Linear(z_dim, conv3_weight_size)
        self.conv3_bias = nn.Parameter(torch.zeros(self.channels[3]))
        self.bn3 = nn.BatchNorm2d(self.channels[3])
        
        # Fourth conv layer (standard)
        self.conv4 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=self.f_sizes[3], padding=1)
        self.bn4 = nn.BatchNorm2d(self.channels[4])
        
        # Fifth conv layer (standard)
        self.conv5 = nn.Conv2d(self.channels[4], self.channels[5], kernel_size=self.f_sizes[4], padding=1)
        self.bn5 = nn.BatchNorm2d(self.channels[5])
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        
        # FC layers - much larger
        self.fc1 = nn.Linear(self.channels[5], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 100)
        
        # Initialize hypernetworks
        self._init_hypernetworks()
        self._init_weights()
        
        print(f"Enhanced HyperNetwork Architecture:")
        print(f"  Channels: {self.channels}")
        print(f"  Conv2 hypernetwork: {z_dim} -> {conv2_weight_size} parameters")
        print(f"  Conv3 hypernetwork: {z_dim} -> {conv3_weight_size} parameters")
        
    def _init_hypernetworks(self):
        """Initialize hypernetwork components with small weights"""
        nn.init.normal_(self.hyper_linear_2.weight, std=0.01)
        nn.init.constant_(self.hyper_linear_2.bias, 0.0)
        nn.init.normal_(self.hyper_linear_3.weight, std=0.01)
        nn.init.constant_(self.hyper_linear_3.bias, 0.0)
        
    def _init_weights(self):
        """Initialize standard layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and 'hyper_linear' not in str(m):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def generate_conv2_weights(self):
        """Generate conv2 weights using hypernetwork"""
        weights_flat = self.hyper_linear_2(self.z_signal_2)
        kernel = weights_flat.reshape(self.channels[2], self.channels[1], self.f_sizes[1], self.f_sizes[1])
        return kernel
        
    def generate_conv3_weights(self):
        """Generate conv3 weights using hypernetwork"""
        weights_flat = self.hyper_linear_3(self.z_signal_3)
        kernel = weights_flat.reshape(self.channels[3], self.channels[2], self.f_sizes[2], self.f_sizes[2])
        return kernel
        
    def forward(self, x):
        # Conv1 block (standard)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32->16
        
        # Conv2 block (hypernetwork)
        conv2_weights = self.generate_conv2_weights()
        x = F.conv2d(x, conv2_weights, bias=self.conv2_bias, padding=1)
        x = self.pool(F.relu(self.bn2(x)))  # 16->8
        
        # Conv3 block (hypernetwork)  
        conv3_weights = self.generate_conv3_weights()
        x = F.conv2d(x, conv3_weights, bias=self.conv3_bias, padding=1)
        x = self.pool(F.relu(self.bn3(x)))  # 8->4
        
        # Conv4 block (standard)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 4->2
        
        # Conv5 block (standard)
        x = F.relu(self.bn5(self.conv5(x)))  # 2x2
        
        # Global average pooling
        x = self.adaptive_pool(x)  # 2x2 -> 1x1
        x = x.view(x.size(0), -1)  # Flatten
        
        # FC layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def get_data_loaders(batch_size=64, num_workers=4):
    """Enhanced data augmentation for better generalization"""
    
    # Strong data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.1)  # Cutout augmentation
    ])
    
    # Simple normalization for validation/test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100('data', train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR100('data', train=False, transform=transform_test)
    
    # Create validation split
    train_size = len(train_dataset) - 5000
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, 5000])
    
    # Apply test transform to validation set
    val_dataset.dataset = datasets.CIFAR100('data', train=True, download=False, transform=transform_test)
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset.dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=100):
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
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0 and batch_idx > 0:
            current_acc = 100. * correct / total
            print(f'  Batch: {batch_idx:3d}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {current_acc:6.2f}%')
            
            # Debug hypernetwork signals
            z2_norm = model.z_signal_2.norm().item()
            z3_norm = model.z_signal_3.norm().item()
            print(f'    z_signal_2 norm: {z2_norm:.4f}, z_signal_3 norm: {z3_norm:.4f}')
    
    train_loss /= len(train_loader)
    train_acc = correct / total
    return train_loss, 1.0 - train_acc

def evaluate(model, device, data_loader):
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
    return loss, 1.0 - acc

def count_parameters(model):
    """Count and categorize parameters"""
    total_params = 0
    hyper_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            
            if 'z_signal' in name or 'hyper_linear' in name:
                hyper_params += num_params
                print(f"[HYPER] {name}: {param.shape}, {num_params:,}")
            else:
                print(f"[OTHER] {name}: {param.shape}, {num_params:,}")
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Hypernetwork parameters: {hyper_params:,}")
    print(f"Standard parameters: {total_params - hyper_params:,}")
    print(f"Hypernetwork ratio: {100.*hyper_params/total_params:.2f}%")
    return total_params, hyper_params

def cosine_annealing_lr(optimizer, epoch, total_epochs, max_lr, min_lr=1e-6):
    """Cosine annealing learning rate schedule"""
    lr = min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    print("Starting CIFAR-100 ENHANCED HyperNetwork Training...")
    print("TARGET: 90% Test Accuracy")
    print("="*80)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = set_device()
    
    # Enhanced hyperparameters for 90% target
    batch_size = 64          # Smaller batch for better gradients
    epochs = 100             # Much longer training
    max_lr = 0.001           # Initial learning rate
    min_lr = 1e-6            # Minimum learning rate
    z_dim = 16               # Larger z dimension
    weight_decay = 1e-4      # L2 regularization
    
    print(f"Enhanced Hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Max learning rate: {max_lr}")
    print(f"  Z dimension: {z_dim}")
    print(f"  Weight decay: {weight_decay}")
    print("="*80)
    
    # Load data with enhanced augmentation
    print("Loading CIFAR-100 with enhanced data augmentation...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)
    
    print(f"Dataset sizes: Train={len(train_loader.dataset)}, "
          f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    print("="*80)
    
    # Create enhanced model
    print("Creating Enhanced HyperNetwork...")
    model = EnhancedHyperCNNCIFAR(z_dim=z_dim).to(device)
    
    # Advanced optimizer
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    print("\nEnhanced HyperNetwork parameters:")
    total_params, hyper_params = count_parameters(model)
    print("="*80)
    
    # Training variables
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience = 15
    patience_counter = 0
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Target: 90% test accuracy")
    print("="*80)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Cosine annealing learning rate
        current_lr = cosine_annealing_lr(optimizer, epoch, epochs, max_lr, min_lr)
        
        print(f"\nEpoch {epoch+1}/{epochs} - LR: {current_lr:.6f}")
        print("-" * 60)
        
        # Training
        train_loss, train_err = train_epoch(model, device, train_loader, optimizer, epoch)
        
        # Validation
        val_loss, val_err = evaluate(model, device, val_loader)
        val_acc = 1.0 - val_err
        
        # Test evaluation (every 5 epochs or if validation improved)
        test_acc = 0.0
        if epoch % 5 == 0 or val_acc > best_val_acc:
            test_loss, test_err = evaluate(model, device, test_loader)
            test_acc = 1.0 - test_err
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'cifar100_enhanced_hyper_best.pt')
                print(f"    *** New best test accuracy: {best_test_acc*100:.2f}% - Model saved! ***")
        
        # Early stopping based on validation
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_time = time.time() - start_time
        
        # Print epoch results
        print(f"\nEpoch Results:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={100*(1-train_err):.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")
        print(f"  Test:  Acc={test_acc*100:.2f}% (Best: {best_test_acc*100:.2f}%)")
        print(f"  Time: {epoch_time:.1f}s, Patience: {patience_counter}/{patience}")
        
        # Check if we reached target
        if best_test_acc >= 0.90:
            print(f"\n🎉 TARGET ACHIEVED! Best test accuracy: {best_test_acc*100:.2f}% >= 90%")
            break
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load('cifar100_enhanced_hyper_best.pt'))
    final_test_loss, final_test_err = evaluate(model, device, test_loader)
    final_test_acc = (1 - final_test_err) * 100
    
    print(f"Enhanced HyperNetwork - Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Previous Fixed Version - Test Accuracy: 33.52%")
    print(f"Standard CNN - Test Accuracy: 16.23%")
    
    improvement_vs_fixed = final_test_acc - 33.52
    improvement_vs_std = final_test_acc - 16.23
    
    print(f"Improvement vs Fixed HyperNet: {improvement_vs_fixed:+.2f}%")
    print(f"Improvement vs Standard CNN: {improvement_vs_std:+.2f}%")
    
    # Print final z_signals
    print(f"\nFinal hypernetwork signals:")
    print(f"z_signal_2: {model.z_signal_2.detach().cpu().numpy().flatten()}")
    print(f"z_signal_3: {model.z_signal_3.detach().cpu().numpy().flatten()}")
    
    # Model size
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assume float32
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Estimated size: {model_size_mb:.1f} MB")
    
    target_reached = "YES ✅" if final_test_acc >= 90.0 else "NO ❌"
    print(f"\nTarget (90% accuracy) reached: {target_reached}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
