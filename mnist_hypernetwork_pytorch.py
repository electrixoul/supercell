'''
MNIST Static HyperNetwork Example (PyTorch Version)
Based on the paper "Hypernetworks" by David Ha, Andrew Dai, and Quoc V. Le.

This script demonstrates static hypernetworks on MNIST dataset,
comparing a standard CNN to a hypernetwork-based CNN.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Configure numpy output format
np.set_printoptions(precision=5, edgeitems=8, linewidth=200)

# Utility function for orthogonal initialization
def orthogonal_init(tensor, gain=1):
    """Initialize weights with orthogonal initialization"""
    if isinstance(tensor, torch.nn.Parameter):
        orthogonal_init(tensor.data, gain)
        return
    if tensor.ndimension() < 2:
        return
    
    # For convolutional layers, reshape to 2D then back
    original_shape = tensor.shape
    num_rows = original_shape[0]
    num_cols = tensor.numel() // num_rows
    
    flat_tensor = tensor.new(num_rows, num_cols).normal_(0, 1)
    
    # SVD
    u, _, v = torch.linalg.svd(flat_tensor, full_matrices=False)
    q = u if u.shape[0] == num_rows else v
    q = q[:num_rows, :num_cols]
    
    # Reshape back
    with torch.no_grad():
        tensor.view_as(flat_tensor).copy_(q)
        tensor.mul_(gain)
    return tensor

# Model for standard CNN approach
class StandardCNN(nn.Module):
    def __init__(self, f_size=7, in_size=16, out_size=16):
        super(StandardCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, in_size, kernel_size=f_size, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer (standard)
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size=f_size, padding='same')
        # Calculate the output size after convolutions and pooling
        # After two 2x2 max pools, 28x28 -> 7x7
        self.fc = nn.Linear(out_size * 7 * 7, 10)
        
        # Initialize weights with truncated normal (like TensorFlow)
        nn.init.normal_(self.conv1.weight, std=0.01)
        self.conv1.bias.data.fill_(0.0)
        nn.init.normal_(self.conv2.weight, std=0.01)
        self.conv2.bias.data.fill_(0.0)
        
    def forward(self, x):
        # First conv+pool block
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv+pool block
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten and fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_conv2_weights(self):
        """Return the second convolutional layer's weights for visualization"""
        return self.conv2.weight.detach().cpu().numpy()

# Hypernetwork CNN model
class HyperCNN(nn.Module):
    def __init__(self, f_size=7, in_size=16, out_size=16, z_dim=4):
        super(HyperCNN, self).__init__()
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.z_dim = z_dim
        
        # First convolutional layer (standard, not generated)
        self.conv1 = nn.Conv2d(1, in_size, kernel_size=f_size, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        
        # Hypernetwork for generating conv2 weights
        # Z signal for the hypernetwork
        self.z_signal = nn.Parameter(torch.randn(1, z_dim) * 0.01)
        
        # W2: maps z_signal to intermediate representation
        self.w2 = nn.Parameter(torch.randn(z_dim, in_size * z_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(in_size * z_dim))
        
        # W1: maps intermediate representation to final weights
        self.w1 = nn.Parameter(torch.randn(z_dim, out_size * f_size * f_size) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(out_size * f_size * f_size))
        
        # Bias for second conv layer (not generated)
        self.conv2_bias = nn.Parameter(torch.zeros(out_size))
        
        # Fully connected layer
        self.fc = nn.Linear(out_size * 7 * 7, 10)
        
        # Initialize weights with truncated normal (like TensorFlow)
        nn.init.normal_(self.conv1.weight, std=0.01)
        self.conv1.bias.data.fill_(0.0)
        nn.init.normal_(self.w1, std=0.01)
        nn.init.normal_(self.w2, std=0.01)
        
    def generate_conv2_weights(self):
        # Generate weights using hypernetwork
        h_in = torch.matmul(self.z_signal, self.w2) + self.b2
        h_in = h_in.reshape(self.in_size, self.z_dim)
        h_final = torch.matmul(h_in, self.w1) + self.b1
        # Reshape to convolutional filter format
        kernel = h_final.reshape(self.out_size, self.in_size, self.f_size, self.f_size)
        # PyTorch uses (out_channels, in_channels, height, width) format so no transpose needed
        return kernel
        
    def forward(self, x):
        # First conv+pool block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Generate conv2 weights using hypernetwork
        conv2_weights = self.generate_conv2_weights()
        
        # Manual convolution with generated weights
        batch_size = x.size(0)
        # Reshape input for unfold (batch_size, channels, height, width)
        x_unf = F.unfold(x, kernel_size=(self.f_size, self.f_size), padding=self.f_size//2)
        # Get dimensions
        in_height = x.size(2)
        in_width = x.size(3)
        # Apply convolution with generated weights
        x = F.conv2d(x, conv2_weights, bias=self.conv2_bias, padding='same')
        
        # Continue with pool and fully connected
        x = self.pool(F.relu(x))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
    
    def get_conv2_weights(self):
        """Return the generated second convolutional layer's weights for visualization"""
        return self.generate_conv2_weights().detach().cpu().numpy()

# Training and evaluation functions
def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Learning rate scheduler
    lr_scheduler = None
    if hasattr(optimizer, "param_groups"):
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0:
            print(f'Batch: {batch_idx}/{len(train_loader)} '
                  f'Loss: {loss.item():.4f} '
                  f'Err: {1-correct/total:.4f}')
    
    # Update learning rate
    if lr_scheduler:
        lr_scheduler.step()
    
    train_loss /= len(train_loader)
    train_acc = correct / total
    train_err = 1.0 - train_acc
    
    return train_loss, train_err, optimizer.param_groups[0]['lr']

def evaluate(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    loss /= total
    acc = correct / total
    err = 1.0 - acc
    
    return loss, err

def show_filter_stats(conv_filter, title="Filter Stats"):
    """Display statistics for a convolutional filter"""
    print(f"\n{title}:")
    print(f"Filter shape: {conv_filter.shape}")
    print(f"mean = {np.mean(conv_filter):.5f}")
    print(f"stddev = {np.std(conv_filter):.5f}")
    print(f"max = {np.max(conv_filter):.5f}")
    print(f"min = {np.min(conv_filter):.5f}")
    print(f"median = {np.median(conv_filter):.5f}")

def count_parameters(model):
    """Count and print the number of trainable parameters in the model"""
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {param.shape}, {num_params}")
    
    print(f"Total trainable parameters: {total_params}")
    return total_params

def visualize_filters(conv_filter, save_path):
    """Visualize convolutional filters and save to a file"""
    f_size = conv_filter.shape[2]
    in_dim = conv_filter.shape[1]
    out_dim = conv_filter.shape[0]
    
    # Create a canvas for visualization
    canvas = np.zeros(((f_size+1)*out_dim, (f_size+1)*in_dim))
    
    # Plot each filter
    for i in range(out_dim):
        for j in range(in_dim):
            canvas[i*(f_size+1):i*(f_size+1)+f_size, j*(f_size+1):j*(f_size+1)+f_size] = conv_filter[i, j]
    
    # Add padding
    canvas_fixed = np.zeros((canvas.shape[0]+1, canvas.shape[1]+1))
    canvas_fixed[1:, 1:] = canvas
    
    # Plot and save
    plt.figure(figsize=(16, 16))
    plt.imshow(canvas_fixed.T, cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def main():
    print("Starting MNIST Hypernetwork Example (PyTorch Version)...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 1000  # Match original implementation
    test_batch_size = 1000
    epochs = 10  # Reduced for demonstration (original was 50)
    lr = 0.005
    min_lr = 0.0001
    f_size = 7
    in_size = 16
    out_size = 16
    z_dim = 4
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # MNIST test set has 10,000 samples
    # We'll use all of it as test set since we're already using 60,000 training samples
    test_dataset_full = test_dataset
    test_dataset = test_dataset_full
    
    # Initialize validation set from a subset of training data
    train_size = len(train_dataset) - 10000  # Use 10,000 samples for validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, 10000]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Train standard CNN model
    print("\nTraining standard CNN model...")
    model_std = StandardCNN(f_size=f_size, in_size=in_size, out_size=out_size).to(device)
    optimizer_std = optim.Adam(model_std.parameters(), lr=lr)
    
    best_val_err = 1.0
    best_val_loss = float('inf')
    
    # Count model parameters
    count_parameters(model_std)
    
    # Training loop
    for epoch in range(epochs):
        # Train and get metrics
        train_loss, train_err, current_lr = train(model_std, device, train_loader, optimizer_std, epoch)
        
        # Evaluate on validation set
        val_loss, val_err = evaluate(model_std, device, val_loader)
        
        # Track best model
        test_err = 0.0
        if val_err <= best_val_err:
            best_val_err = val_err
            best_val_loss = val_loss
            
            # Evaluate on test set - only every few epochs to save time
            if epoch == 0 or epoch % 3 == 0 or epoch == epochs-1:
                test_loss, test_err = evaluate(model_std, device, test_loader)
        
        # Print epoch results
        print(f"Epoch: {epoch}, "
              f"train_loss={train_loss:.4f}, "
              f"train_err={train_err:.4f}, "
              f"val_err={val_err:.4f}, "
              f"best_val_err={best_val_err:.4f}, "
              f"test_err={test_err:.4f}, "
              f"lr={current_lr:.6f}")
    
    # Visualize standard CNN filters
    conv_filter_std = model_std.get_conv2_weights()
    show_filter_stats(conv_filter_std, "Standard CNN Filter Stats")
    visualize_filters(conv_filter_std, "pytorch_standard_cnn_filter.png")
    
    # Train hypernetwork CNN model
    print("\nTraining hypernetwork CNN model...")
    model_hyper = HyperCNN(f_size=f_size, in_size=in_size, out_size=out_size, z_dim=z_dim).to(device)
    optimizer_hyper = optim.Adam(model_hyper.parameters(), lr=lr)
    
    best_val_err = 1.0
    best_val_loss = float('inf')
    
    # Count model parameters
    count_parameters(model_hyper)
    
    # Training loop
    for epoch in range(epochs):
        # Train and get metrics
        train_loss, train_err, current_lr = train(model_hyper, device, train_loader, optimizer_hyper, epoch)
        
        # Evaluate on validation set
        val_loss, val_err = evaluate(model_hyper, device, val_loader)
        
        # Track best model
        test_err = 0.0
        if val_err <= best_val_err:
            best_val_err = val_err
            best_val_loss = val_loss
            
            # Evaluate on test set - only every few epochs to save time
            if epoch == 0 or epoch % 3 == 0 or epoch == epochs-1:
                test_loss, test_err = evaluate(model_hyper, device, test_loader)
        
        # Print epoch results
        print(f"Epoch: {epoch}, "
              f"train_loss={train_loss:.4f}, "
              f"train_err={train_err:.4f}, "
              f"val_err={val_err:.4f}, "
              f"best_val_err={best_val_err:.4f}, "
              f"test_err={test_err:.4f}, "
              f"lr={current_lr:.6f}")
    
    # Visualize hypernetwork CNN filters
    conv_filter_hyper = model_hyper.get_conv2_weights()
    show_filter_stats(conv_filter_hyper, "Hypernetwork CNN Filter Stats")
    visualize_filters(conv_filter_hyper, "pytorch_hypernetwork_cnn_filter.png")
    
    print("\nExecution complete. Filter visualizations saved as PNG files.")

if __name__ == "__main__":
    main()
