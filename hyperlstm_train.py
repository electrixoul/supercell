'''
HyperLSTM Training Script
Based on the paper "HyperNetworks" by David Ha, Andrew Dai, and Quoc V. Le.

理论背景：
HyperLSTM是一种在序列建模任务中表现优异的循环神经网络变体。它通过使用一个辅助LSTM网络
（称为hyper-LSTM）来动态生成主LSTM网络的权重，从而实现对序列上下文的自适应。

原论文提出了两种生成权重的方式：
1. 权重调制（weight scaling）：生成一个乘性因子来调整原始权重
2. 全权重生成（full weight generation）：完全替代原始权重

本实现中采用了权重调制方法，因为它在计算效率和参数数量方面具有优势，同时保留了
动态调整能力。此外，还使用了以下技术来增强模型性能：
- 层归一化（Layer Normalization）：稳定训练过程
- 正交初始化（Orthogonal Initialization）：改善梯度流
- 循环Dropout（Recurrent Dropout）：防止过拟合

这个训练脚本提供了训练、评估和文本生成功能，完整实现了论文中描述的HyperLSTM的
训练和推理过程。
'''

import os
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from hyperlstm_pytorch import HyperLSTM, CharDataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HyperLSTM Language Model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/input.txt',
                        help='Path to the training data file')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='Sequence length for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    
    # Model parameters
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Character embedding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden size of the main LSTM')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--hyper_num_units', type=int, default=128,
                        help='Number of units in the HyperLSTM cell')
    parser.add_argument('--hyper_embedding_size', type=int, default=16,
                        help='Size of the hypernetwork embeddings')
    parser.add_argument('--use_layer_norm', action='store_true',
                        help='Use layer normalization')
    parser.add_argument('--use_recurrent_dropout', action='store_true',
                        help='Use recurrent dropout')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.9,
                        help='Keep probability for recurrent dropout')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='Learning rate decay factor')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for learning rate scheduling')
    parser.add_argument('--clip', type=float, default=5.0,
                        help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling during text generation')
    parser.add_argument('--gen_length', type=int, default=1000,
                        help='Length of generated text')
    parser.add_argument('--prime_text', type=str, default='The ',
                        help='Prime text to start generation')
    
    # Runtime parameters
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--model_name', type=str, default='hyperlstm',
                        help='Model name for saving/loading')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'generate'],
                        help='Mode: train, evaluate, or generate text')
    parser.add_argument('--load_model', type=str, default='',
                        help='Path to load a saved model from')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_and_preprocess_data(args):
    """Load and preprocess the text data"""
    print(f"Loading data from {args.data_path}")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
    
    # If file doesn't exist, create a small sample file
    if not os.path.exists(args.data_path):
        print(f"Data file {args.data_path} not found. Creating a sample file.")
        sample_text = (
            "The quick brown fox jumps over the lazy dog. "
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Machine learning and artificial intelligence are fascinating fields "
            "that combine mathematics, statistics, and computer science. "
            "Neural networks, especially recurrent neural networks and their variants "
            "like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit), "
            "are powerful tools for sequence modeling tasks such as natural language processing. "
            "Hypernetworks, proposed by David Ha, Andrew Dai, and Quoc V. Le, "
            "are a novel approach where one network generates the weights for another network."
        )
        # Replicate the text to make it longer
        sample_text = sample_text * 50
        with open(args.data_path, 'w') as f:
            f.write(sample_text)
    
    # Load the text data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded {len(text)} characters")
    
    # Create the dataset
    dataset = CharDataset(text, args.seq_length)
    
    # Split into training and validation sets
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    return train_loader, val_loader, dataset

def create_model(args, vocab_size):
    """Create the HyperLSTM model"""
    model = HyperLSTM(
        input_size=vocab_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        output_size=vocab_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        hyper_num_units=args.hyper_num_units,
        hyper_embedding_size=args.hyper_embedding_size,
        use_layer_norm=args.use_layer_norm,
        use_recurrent_dropout=args.use_recurrent_dropout,
        dropout_keep_prob=args.dropout_keep_prob
    )
    
    return model

def train(args, model, train_loader, val_loader, device):
    """Train the model"""
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay, patience=args.patience
    )
    
    # Training variables
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, args.clip)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model_path = os.path.join(args.save_dir, f"{args.model_name}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, model_path)
            print(f"Best model saved to {model_path}")
        
        # Print epoch summary
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {math.exp(val_loss):.2f} | "
              f"Time: {elapsed:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f"{args.model_name}_final.pt")
    torch.save({
        'epoch': args.epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'args': args
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, f"{args.model_name}_loss.png"))
    print(f"Loss curve saved to {os.path.join(args.save_dir, f'{args.model_name}_loss.png')}")
    
    # Print best model summary
    print(f"Best model was saved at epoch {best_epoch+1} with validation loss {best_val_loss:.4f} (PPL: {math.exp(best_val_loss):.2f})")
    
    return model

def train_epoch(model, data_loader, optimizer, device, clip_value):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    start_time = time.time()
    
    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        # Move to device
        input_batch = input_batch.to(device).transpose(0, 1)  # (seq_len, batch_size)
        target_batch = target_batch.to(device).transpose(0, 1)  # (seq_len, batch_size)
        batch_size = input_batch.size(1)
        
        # Forward pass
        hidden = model.init_hidden(batch_size, device)
        optimizer.zero_grad()
        output, _ = model(input_batch, hidden)
        
        # Reshape for loss calculation
        output = output.view(-1, model.output_size)
        target = target_batch.reshape(-1)
        
        # Calculate loss
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Print progress
        if batch_idx % 20 == 0:
            elapsed = time.time() - start_time
            ms_per_batch = elapsed * 1000 / (batch_idx + 1)
            progress = batch_idx / len(data_loader) * 100
            print(f"  {progress:.2f}% | Batch {batch_idx}/{len(data_loader)} | "
                  f"Loss: {loss.item():.4f} | {ms_per_batch:.2f} ms/batch")
    
    return total_loss / total_samples

def evaluate_model(model, data_loader, device):
    """Evaluate the model on the validation set"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            # Move to device
            input_batch = input_batch.to(device).transpose(0, 1)  # (seq_len, batch_size)
            target_batch = target_batch.to(device).transpose(0, 1)  # (seq_len, batch_size)
            batch_size = input_batch.size(1)
            
            # Forward pass
            hidden = model.init_hidden(batch_size, device)
            output, _ = model(input_batch, hidden)
            
            # Reshape for loss calculation
            output = output.view(-1, model.output_size)
            target = target_batch.reshape(-1)
            
            # Calculate loss
            loss = nn.functional.cross_entropy(output, target)
            
            # Track loss
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    return total_loss / total_samples

def generate_text(model, dataset, device, prime_text, length, temperature=1.0):
    """
    Generate text using the trained model
    
    理论背景：
    文本生成是使用语言模型的一个常见应用。在HyperLSTM的背景下，
    模型通过上下文动态调整的权重可以生成更连贯、更有语义连续性的文本序列。
    
    生成过程采用了自回归方法（autoregressive generation）：
    1. 从一个"种子"文本（prime_text）开始
    2. 模型预测下一个字符的概率分布
    3. 使用温度参数（temperature）控制采样的随机性：
       - 低温（<1.0）：使分布更"尖锐"，偏向高概率选择，生成更确定性的文本
       - 高温（>1.0）：使分布更"平坦"，增加随机性，生成更多样化的文本
    4. 从调整后的分布中采样下一个字符
    5. 将采样的字符添加到生成文本中，并用作下一步的输入
    6. 重复此过程直到达到指定长度
    
    这种方法结合了HyperLSTM的动态权重生成能力，可以根据生成文本的上下文
    不断调整模型的内部表示，从而产生更具连贯性的文本。
    """
    model.eval()
    
    # Convert prime text to tensor
    prime_chars = [ch for ch in prime_text]
    prime_input = torch.tensor([dataset.char_to_idx.get(ch, 0) for ch in prime_chars], 
                             dtype=torch.long, device=device).unsqueeze(1)
    
    # Initialize hidden state
    hidden = model.init_hidden(1, device)
    
    # Generate the prime text (already in the output)
    with torch.no_grad():
        output, hidden = model(prime_input, hidden)
    
    # Start with the prime text
    generated_text = prime_text
    current_char = prime_chars[-1]
    
    # Generate new text
    with torch.no_grad():
        for _ in range(length):
            # Convert current character to input tensor
            input_tensor = torch.tensor([dataset.char_to_idx.get(current_char, 0)], 
                                     dtype=torch.long, device=device).unsqueeze(0)
            
            # Forward pass through the model
            output, hidden = model(input_tensor, hidden)
            
            # Apply temperature to logits
            logits = output[-1].squeeze() / temperature
            probs = torch.softmax(logits, dim=0)
            
            # Sample from the distribution
            char_idx = torch.multinomial(probs, 1).item()
            
            # Convert index to character
            current_char = dataset.idx_to_char[char_idx]
            
            # Add to generated text
            generated_text += current_char
    
    return generated_text

def load_model_from_checkpoint(path, vocab_size=None, device=torch.device('cpu')):
    """Load a trained model from a checkpoint"""
    # 设置weights_only=False以兼容PyTorch 2.6+版本
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Get the arguments used to create the model
    saved_args = checkpoint['args']
    
    # Create a model with the same architecture
    model = create_model(saved_args, vocab_size or saved_args.vocab_size)
    model.to(device)
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, saved_args

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    train_loader, val_loader, dataset = load_and_preprocess_data(args)
    vocab_size = dataset.vocab_size
    
    # Create or load model
    if args.load_model:
        model, loaded_args = load_model_from_checkpoint(args.load_model, vocab_size, device)
        print(f"Loaded model from {args.load_model}")
    else:
        model = create_model(args, vocab_size)
        model.to(device)
        print("Created new model")
    
    # Print model architecture
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Perform operations based on mode
    if args.mode == 'train':
        print("Starting training...")
        model = train(args, model, train_loader, val_loader, device)
        
    elif args.mode == 'eval':
        print("Evaluating model...")
        val_loss = evaluate_model(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
        
    elif args.mode == 'generate':
        print("Generating text...")
        generated_text = generate_text(
            model, dataset, device, args.prime_text, args.gen_length, args.temperature
        )
        print(f"\nGenerated text (temperature={args.temperature}):\n")
        print(generated_text)
        
        # Save generated text to file
        out_path = os.path.join(args.save_dir, f"generated_text_{args.temperature}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"\nGenerated text saved to {out_path}")
    
    print("Done!")

if __name__ == '__main__':
    main()
