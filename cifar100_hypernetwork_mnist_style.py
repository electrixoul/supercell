'''
CIFAR-100 HyperNetwork Based on MNIST Success Pattern
Adapting the successful MNIST hypernetwork design to CIFAR-100

Key adaptations from MNIST version:
1. Same hypernetwork architecture (2-layer with reshape)
2. Adam optimizer with exponential LR decay
3. Small weight initialization
4. Gradual complexity increase
5. Proper gradient clipping
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
    """Set optimal device for M4 Pro chip"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) acceleration on M4 Pro")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

class StandardCNNCIFAR(nn.Module):
    def __init__(self, f_size=5, in_size=32, out_size=64):
        super(StandardCNNCIFAR, self).__init__()
        # First conv layer
        self.conv1 = nn.Conv2d(3, in_size, kernel_size=f_size, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        
        # Second conv layer (to be compared with hypernetwork)
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size=f_size, padding='same')
        
        # Third conv layer
        self.conv3 = nn.Conv2d(out_size, 128, kernel_size=3, padding='same')
        
        # FC layers - after 3 pooling: 32->16->8->4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 100)
        self.dropout = nn.Dropout(0.5)
        
        # Initialize with small weights like MNIST version
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                m.bias.data.fill_(0.0)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))      # 32->16
        x = self.pool(F.relu(self.conv2(x)))      # 16->8  
        x = self.pool(F.relu(self.conv3(x)))      # 8->4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_conv2_weights(self):
        return self.conv2.weight.detach().cpu().numpy()

class HyperCNNCIFAR(nn.Module):
    def __init__(self, f_size=5, in_size=32, out_size=64, z_dim=8):
        super(HyperCNNCIFAR, self).__init__()
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.z_dim = z_dim
        
        # First conv layer (standard, not generated)
        self.conv1 = nn.Conv2d(3, in_size, kernel_size=f_size, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        
        # Hypernetwork for generating conv2 weights (MNIST style)
        # Z signal - small initialization like MNIST
        self.z_signal = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        
        # W2: maps z_signal to intermediate (MNIST style)
        self.w2 = nn.Parameter(torch.randn(z_dim, in_size * z_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(in_size * z_dim))
        
        # W1: maps intermediate to conv2 weights (MNIST style)
        conv2_weight_size = out_size * f_size * f_size
        self.w1 = nn.Parameter(torch.randn(z_dim, conv2_weight_size) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(conv2_weight_size))
        
        # Conv2 bias (not generated)
        self.conv2_bias = nn.Parameter(torch.zeros(out_size))
        
        # Third conv layer (standard)
        self.conv3 = nn.Conv2d(out_size, 128, kernel_size=3, padding='same')
        
        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 100)
        self.dropout = nn.Dropout(0.5)
        
        # Initialize like MNIST version
        self._init_weights()
        
    def _init_weights(self):
        # First conv layer
        nn.init.normal_(self.conv1.weight, std=0.01)
        self.conv1.bias.data.fill_(0.0)
        
        # Hypernetwork weights already initialized in __init__
        
        # Other layers
        nn.init.normal_(self.conv3.weight, std=0.01)
        self.conv3.bias.data.fill_(0.0)
        nn.init.normal_(self.fc1.weight, std=0.01)
        self.fc1.bias.data.fill_(0.0)
        nn.init.normal_(self.fc2.weight, std=0.01)
        self.fc2.bias.data.fill_(0.0)
        
    def generate_conv2_weights(self):
        """Generate conv2 weights using hypernetwork (MNIST style)"""
        # Forward through hypernetwork
        h_in = torch.matmul(self.z_signal, self.w2) + self.b2
        h_in = h_in.reshape(self.in_size, self.z_dim)  # Key reshape step from MNIST
        h_final = torch.matmul(h_in, self.w1) + self.b1
        
        # Reshape to conv filter format
        kernel = h_final.reshape(self.out_size, self.in_size, self.f_size, self.f_size)
        return kernel
        
    def forward(self, x):
        # First conv+pool
        x = self.pool(F.relu(self.conv1(x)))      # 32->16
        
        # Generate conv2 weights and apply
        conv2_weights = self.generate_conv2_weights()
        x = F.conv2d(x, conv2_weights, bias=self.conv2_bias, padding='same')
        x = self.pool(F.relu(x))                  # 16->8
        
        # Third conv+pool
        x = self.pool(F.relu(self.conv3(x)))      # 8->4
        
        # FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_conv2_weights(self):
        return self.generate_conv2_weights().detach().cpu().numpy()

def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=50):
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
        
        # Gradient clipping like MNIST version
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0:
            current_acc = 100. * correct / total
            print(f'  Batch: {batch_idx:3d}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {current_acc:6.2f}%')
            
            # Debug info for hypernetwork
            if hasattr(model, 'z_signal'):
                z_norm = model.z_signal.norm().item()
                print(f'    z_signal norm: {z_norm:.4f}')
    
    train_loss /= len(train_loader)
    train_acc = correct / total
    train_err = 1.0 - train_acc
    return train_loss, train_err

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
    err = 1.0 - acc
    return loss, err

def count_parameters(model):
    """Count and print the number of trainable parameters"""
    total_params = 0
    hyper_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            
            if 'z_signal' in name or 'w1' in name or 'w2' in name or 'b1' in name or 'b2' in name:
                hyper_params += num_params
                print(f"[HYPER] {name}: {param.shape}, {num_params}")
            else:
                print(f"[OTHER] {name}: {param.shape}, {num_params}")
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Hypernetwork parameters: {hyper_params:,}")
    print(f"Other parameters: {total_params - hyper_params:,}")
    if hyper_params > 0:
        print(f"Hypernetwork ratio: {100.*hyper_params/total_params:.2f}%")
    return total_params, hyper_params

def main():
    print("Starting CIFAR-100 HyperNetwork (MNIST Style)...")
    print("="*60)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = set_device()
    
    # MNIST-style hyperparameters
    batch_size = 128  # Reasonable for CIFAR-100
    test_batch_size = 256
    epochs = 20  # More epochs for complex dataset
    lr = 0.005      # Same as MNIST
    f_size = 5      # Smaller filter size
    in_size = 32    # Reasonable first layer size
    out_size = 64   # Reasonable second layer size  
    z_dim = 8       # Larger than MNIST but not too large
    
    print(f"Hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Filter size: {f_size}")
    print(f"  Z dimension: {z_dim}")
    print("="*60)
    
    # Simple data transforms (avoid complex augmentation initially)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    print("Loading CIFAR-100 dataset...")
    train_dataset = datasets.CIFAR100('data', train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR100('data', train=False, transform=transform_test)
    
    # Validation split
    train_size = len(train_dataset) - 5000
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, 5000]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=2, pin_memory=True)
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print("="*60)
    
    # Train both models for comparison
    results = {}
    
    # 1. Standard CNN baseline
    print("Training Standard CNN baseline...")
    model_std = StandardCNNCIFAR(f_size=f_size, in_size=in_size, out_size=out_size).to(device)
    optimizer_std = optim.Adam(model_std.parameters(), lr=lr)
    scheduler_std = optim.lr_scheduler.ExponentialLR(optimizer_std, gamma=0.99)  # Like MNIST
    
    print("\nStandard CNN parameters:")
    count_parameters(model_std)
    print("="*60)
    
    best_val_err_std = 1.0
    # for epoch in range(epochs):
    #     start_time = time.time()
    #     print(f"\nEpoch {epoch+1}/{epochs} - Standard CNN")
    #     print("-" * 40)
        
    #     train_loss, train_err = train_epoch(model_std, device, train_loader, optimizer_std, epoch)
    #     val_loss, val_err = evaluate(model_std, device, val_loader)
    #     scheduler_std.step()
        
    #     if val_err < best_val_err_std:
    #         best_val_err_std = val_err
    #         torch.save(model_std.state_dict(), 'cifar100_mnist_style_std_best.pt')
        
    #     epoch_time = time.time() - start_time
    #     current_lr = optimizer_std.param_groups[0]['lr']
    #     print(f"\nResults: Train Err: {100*train_err:.2f}% | Val Err: {100*val_err:.2f}% | "
    #           f"Best Val Err: {100*best_val_err_std:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
    
    # # 2. HyperNetwork CNN
    # print("\n" + "="*60)
    # print("Training HyperNetwork CNN...")
    # model_hyper = HyperCNNCIFAR(f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim).to(device)
    # optimizer_hyper = optim.Adam(model_hyper.parameters(), lr=lr)
    # scheduler_hyper = optim.lr_scheduler.ExponentialLR(optimizer_hyper, gamma=0.99)  # Like MNIST
    
    print("\nHyperNetwork CNN parameters:")
    count_parameters(model_hyper)
    print("="*60)
    
    best_val_err_hyper = 1.0
    for epoch in range(epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs} - HyperNetwork CNN")
        print("-" * 40)
        
        train_loss, train_err = train_epoch(model_hyper, device, train_loader, optimizer_hyper, epoch)
        val_loss, val_err = evaluate(model_hyper, device, val_loader)
        scheduler_hyper.step()
        
        if val_err < best_val_err_hyper:
            best_val_err_hyper = val_err
            torch.save(model_hyper.state_dict(), 'cifar100_mnist_style_hyper_best.pt')
        
        epoch_time = time.time() - start_time
        current_lr = optimizer_hyper.param_groups[0]['lr']
        print(f"\nResults: Train Err: {100*train_err:.2f}% | Val Err: {100*val_err:.2f}% | "
              f"Best Val Err: {100*best_val_err_hyper:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # Load best models and test
    model_std.load_state_dict(torch.load('cifar100_mnist_style_std_best.pt'))
    model_hyper.load_state_dict(torch.load('cifar100_mnist_style_hyper_best.pt'))
    
    std_test_loss, std_test_err = evaluate(model_std, device, test_loader)
    hyper_test_loss, hyper_test_err = evaluate(model_hyper, device, test_loader)
    
    std_test_acc = (1 - std_test_err) * 100
    hyper_test_acc = (1 - hyper_test_err) * 100
    
    print(f"Standard CNN - Test Accuracy: {std_test_acc:.2f}%")
    print(f"HyperNetwork CNN - Test Accuracy: {hyper_test_acc:.2f}%")
    
    improvement = hyper_test_acc - std_test_acc
    print(f"Improvement: {improvement:+.2f}%")
    
    # Print final z_signal
    print(f"\nFinal z_signal: {model_hyper.z_signal.detach().cpu().numpy().flatten()}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
