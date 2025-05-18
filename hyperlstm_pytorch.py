'''
HyperLSTM PyTorch Implementation
Based on the paper "HyperNetworks" by David Ha, Andrew Dai, and Quoc V. Le.

理论背景:
HyperNetworks是一种通过另一个网络动态生成权重的方法。在HyperLSTM中，一个较小的
辅助LSTM（称为hyper-LSTM）被用来生成主LSTM的权重。这种方法有几个好处：
1. 参数共享：通过较小的网络生成较大网络的权重，可以实现更高效的参数共享。
2. 上下文自适应：权重可以根据当前处理的序列动态调整。
3. 更强的表示能力：允许模型处理更复杂的模式和长期依赖性。

与传统LSTM相比，HyperLSTM在多个序列预测任务上表现出更好的性能。HyperLSTM中的
关键创新是使用一个较小的网络来生成主网络的权重，而不是使用固定权重。这使得模型
可以根据输入序列的上下文动态适应其参数。

此脚本实现了完整的HyperLSTM模型，用于字符级语言建模任务。
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import time
import argparse
import math
from typing import Tuple, Optional, Dict, List, Union

# Configure numpy output format
np.set_printoptions(precision=5, edgeitems=8, linewidth=200)

def orthogonal_init(tensor, gain=1.0):
    """
    Orthogonal initialization (similar to tf.orthogonal_initializer)
    """
    if isinstance(tensor, torch.nn.Parameter):
        orthogonal_init(tensor.data, gain)
        return
    if tensor.ndimension() < 2:
        return
    
    # Use PyTorch's built-in orthogonal initialization function instead
    # This is more reliable and tested extensively
    nn.init.orthogonal_(tensor, gain=gain)
    return tensor

class LayerNorm(nn.Module):
    """
    Layer Normalization module for HyperLSTM
    """
    def __init__(self, feature_size, eps=1e-6, use_bias=True, gamma_start=1.0):
        super(LayerNorm, self).__init__()
        self.feature_size = feature_size
        self.eps = eps
        self.use_bias = use_bias
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(feature_size) * gamma_start)
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(feature_size))
        
    def forward(self, x):
        # x shape: [batch_size, feature_size]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps
        normalized = (x - mean) / std
        
        output = self.gamma * normalized
        if self.use_bias:
            output = output + self.beta
            
        return output

def layer_norm_all(h, batch_size, base, num_units, gamma, beta=None, epsilon=1e-3):
    """
    Performs layer norm on multiple features at once (4 gates in LSTM)
    """
    # Reshape h to perform layer norm in parallel
    h_reshape = h.reshape(batch_size, base, num_units)
    mean = h_reshape.mean(dim=2, keepdim=True)
    var = ((h_reshape - mean) ** 2).mean(dim=2, keepdim=True)
    rstd = torch.rsqrt(var + epsilon)
    h_reshape = (h_reshape - mean) * rstd
    
    # Reshape back to original
    h = h_reshape.reshape(batch_size, base * num_units)
    
    # Apply gamma and beta (gamma is a vector of size 4*num_units)
    output = gamma * h
    if beta is not None:
        output = output + beta
    
    return output

class SuperLinear(nn.Module):
    """
    Linear layer with custom initializations (similar to super_linear in TF)
    """
    def __init__(self, input_size, output_size, init_w="ortho", weight_start=0.0, 
                 use_bias=True, bias_start=0.0):
        super(SuperLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        
        if init_w == "zeros":
            nn.init.constant_(self.weight, 0.0)
        elif init_w == "constant":
            nn.init.constant_(self.weight, weight_start)
        elif init_w == "gaussian":
            nn.init.normal_(self.weight, std=weight_start)
        elif init_w == "ortho":
            orthogonal_init(self.weight, gain=1.0)
        
        if use_bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            nn.init.constant_(self.bias, bias_start)
    
    def forward(self, x):
        output = torch.matmul(x, self.weight)
        if self.use_bias:
            output = output + self.bias
        return output

class LSTMCell(nn.Module):
    """
    Layer-Normalized LSTM with orthogonal initialization and recurrent dropout
    """
    def __init__(self, input_size, hidden_size, forget_bias=1.0, 
                use_layer_norm=False, use_recurrent_dropout=False, 
                dropout_keep_prob=0.90):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.use_layer_norm = use_layer_norm
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob
        
        # Weight matrices: input-to-hidden and hidden-to-hidden
        self.W_xh = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        
        # Initialize with orthogonal matrices
        orthogonal_init(self.W_xh)
        orthogonal_init(self.W_hh)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        
        # Layer normalization parameters
        if use_layer_norm:
            self.ln_gamma = nn.Parameter(torch.ones(4 * hidden_size))
            self.ln_beta = nn.Parameter(torch.zeros(4 * hidden_size))
            self.ln_c_gamma = nn.Parameter(torch.ones(hidden_size))
            self.ln_c_beta = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x, state):
        """
        x: [batch_size, input_size]
        state: (c, h) tuple of [batch_size, hidden_size]
        """
        c, h = state
        batch_size = x.size(0)
        
        # Concatenate x and h for efficiency
        concat = torch.cat([x, h], dim=1)
        
        # Linear transform
        W_full = torch.cat([self.W_xh, self.W_hh], dim=0)
        concat = torch.matmul(concat, W_full) + self.bias
        
        # Apply layer norm if enabled
        if self.use_layer_norm:
            concat = layer_norm_all(concat, batch_size, 4, self.hidden_size, 
                                   self.ln_gamma, self.ln_beta)
        
        # Split into gates
        i, j, f, o = torch.split(concat, self.hidden_size, dim=1)
        
        # Apply recurrent dropout to the candidate activation if enabled
        if self.use_recurrent_dropout:
            j = F.dropout(torch.tanh(j), p=1-self.dropout_keep_prob, training=self.training)
        else:
            j = torch.tanh(j)
        
        # Compute new cell state
        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * j
        
        # Apply layer norm to cell state if enabled
        if self.use_layer_norm:
            new_h = torch.tanh(layer_norm_all(new_c.unsqueeze(1), batch_size, 1, 
                                           self.hidden_size, self.ln_c_gamma, 
                                           self.ln_c_beta).squeeze(1)) * torch.sigmoid(o)
        else:
            new_h = torch.tanh(new_c) * torch.sigmoid(o)
        
        return new_h, (new_c, new_h)

class HyperLinear(nn.Module):
    """
    Hypernetwork for dynamic weight generation
    
    理论背景：
    HyperLinear实现了超网络的核心概念——使用一个网络动态生成另一个网络的权重。
    这里使用了两级映射结构：
    1. 首先将输入z映射到一个低维嵌入空间
    2. 然后从嵌入空间映射到目标权重空间
    
    这种结构在原论文中被称为"weight factorization"（权重分解），它允许以相对
    少量的参数生成大量的动态权重。原论文表明，这种设计比直接映射更高效、更有效。
    
    在HyperLSTM中，这个组件负责动态调整LSTM的门控参数，使其根据当前的上下文
    进行自适应调整，从而提高模型处理序列数据的能力。
    """
    def __init__(self, input_size, embedding_size, output_size, use_bias=True):
        super(HyperLinear, self).__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # 第一级映射：z -> embedding（将hyper LSTM的输出映射到低维嵌入空间）
        self.z_linear = SuperLinear(input_size, embedding_size, 
                                    init_w="gaussian", weight_start=0.01,
                                    use_bias=True, bias_start=1.0)
        
        # 第二级映射：embedding -> output weights（从嵌入空间生成目标权重）
        # 使用常数初始化，权重值与嵌入尺寸成比例，有助于稳定训练
        self.weight_linear = SuperLinear(embedding_size, output_size, 
                                        init_w="constant", weight_start=0.1/embedding_size,
                                        use_bias=False)
        
        if use_bias:
            # 偏置项的超网络参数（类似于权重的两级映射）
            self.z_bias_linear = SuperLinear(input_size, embedding_size,
                                            init_w="gaussian", weight_start=0.01,
                                            use_bias=False)
            
            self.bias_linear = SuperLinear(embedding_size, output_size,
                                          init_w="constant", weight_start=0.0,
                                          use_bias=False)
    
    def forward(self, z, layer):
        """
        Generate weights for a layer and apply them
        
        参数:
            z: hypernetwork input [batch_size, input_size] - 超网络LSTM的输出状态
            layer: tensor to be modified [batch_size, layer_size] - 需要被调制的层激活值
            
        返回:
            调制后的层激活值
            
        注意：这里不是直接生成权重矩阵，而是生成一个缩放因子，用元素乘法调制激活值。
        这种设计在原论文中被称为"weight scaling"，比完全替换权重计算效率更高。
        """
        # 生成权重缩放因子
        z_w = self.z_linear(z)  # 第一级映射到嵌入空间
        weights = self.weight_linear(z_w)  # 第二级映射到目标大小
        
        # 应用权重调制（元素乘法）
        output = weights * layer
        
        # 生成并应用偏置项调制（如果启用）
        if self.use_bias:
            z_b = self.z_bias_linear(z)
            bias = self.bias_linear(z_b)
            output = output + bias
            
        return output

class HyperLSTMCell(nn.Module):
    """
    HyperLSTM Cell Implementation
    Based on the paper "HyperNetworks" by David Ha, Andrew Dai, and Quoc V. Le.
    
    理论背景：
    HyperLSTMCell是HyperLSTM的核心组件，它将常规LSTM与超网络结合起来。具体而言：
    1. 辅助LSTM（hyper-LSTM）根据当前输入和主LSTM的前一状态生成动态参数
    2. 这些动态参数用于调制主LSTM的权重和偏置
    3. 调制后的权重用于计算主LSTM的下一状态
    
    与传统LSTM相比，HyperLSTM能够根据序列的上下文动态调整其计算，而不是使用固定权重。
    论文表明，这种动态权重生成机制显著提高了模型在多种序列建模任务中的性能。
    
    具体实现中，每个LSTM门（输入门、遗忘门、输出门和单元激活）都有自己的动态权重生成器，
    使模型可以根据当前上下文为不同的门分配不同的权重调制。
    """
    def __init__(self, input_size, hidden_size, forget_bias=1.0,
                use_recurrent_dropout=False, dropout_keep_prob=0.90, 
                use_layer_norm=True, hyper_num_units=128, 
                hyper_embedding_size=16, hyper_use_recurrent_dropout=False):
        super(HyperLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob
        self.use_layer_norm = use_layer_norm
        self.hyper_num_units = hyper_num_units
        self.hyper_embedding_size = hyper_embedding_size
        self.hyper_use_recurrent_dropout = hyper_use_recurrent_dropout
        
        # Total hidden size (main LSTM + HyperLSTM)
        self.total_hidden_size = self.hidden_size + self.hyper_num_units
        
        # Create the HyperLSTM cell
        self.hyper_cell = LSTMCell(input_size + hidden_size, hyper_num_units,
                                  use_recurrent_dropout=hyper_use_recurrent_dropout,
                                  use_layer_norm=use_layer_norm,
                                  dropout_keep_prob=dropout_keep_prob)
        
        # Weight matrices for main LSTM
        self.W_xh = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        
        # Initialize with orthogonal matrices
        orthogonal_init(self.W_xh)
        orthogonal_init(self.W_hh)
        
        # Bias for main LSTM
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        
        # Layer normalization parameters for main LSTM
        if use_layer_norm:
            self.ln_gamma = nn.Parameter(torch.ones(4 * hidden_size))
            self.ln_beta = nn.Parameter(torch.zeros(4 * hidden_size))
            self.ln_c_gamma = nn.Parameter(torch.ones(hidden_size))
            self.ln_c_beta = nn.Parameter(torch.zeros(hidden_size))
        
        # Hypernetworks for main LSTM's input weights
        self.hyper_ix = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        self.hyper_jx = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        self.hyper_fx = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        self.hyper_ox = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        
        # Hypernetworks for main LSTM's hidden weights
        self.hyper_ih = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        self.hyper_jh = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        self.hyper_fh = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        self.hyper_oh = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        
        # Hypernetworks for main LSTM's biases
        self.hyper_ib = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        self.hyper_jb = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        self.hyper_fb = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
        self.hyper_ob = HyperLinear(hyper_num_units, hyper_embedding_size, hidden_size)
    
    def forward(self, x, state):
        """
        x: [batch_size, input_size]
        state: (total_c, total_h) tuple of [batch_size, total_hidden_size]
        """
        # Split the state into main and hyper states
        total_c, total_h = state
        c = total_c[:, :self.hidden_size]
        h = total_h[:, :self.hidden_size]
        hyper_c = total_c[:, self.hidden_size:]
        hyper_h = total_h[:, self.hidden_size:]
        hyper_state = (hyper_c, hyper_h)
        
        batch_size = x.size(0)
        
        # Create hypernet input by concatenating x and h
        hyper_input = torch.cat([x, h], dim=1)
        
        # Run the hypernet LSTM cell
        hyper_output, hyper_new_state = self.hyper_cell(hyper_input, hyper_state)
        
        # Compute input and hidden matrices for main LSTM
        xh = torch.matmul(x, self.W_xh)
        hh = torch.matmul(h, self.W_hh)
        
        # Split input contributions
        ix, jx, fx, ox = torch.split(xh, self.hidden_size, dim=1)
        ix = self.hyper_ix(hyper_output, ix)
        jx = self.hyper_jx(hyper_output, jx)
        fx = self.hyper_fx(hyper_output, fx)
        ox = self.hyper_ox(hyper_output, ox)
        
        # Split hidden contributions
        ih, jh, fh, oh = torch.split(hh, self.hidden_size, dim=1)
        ih = self.hyper_ih(hyper_output, ih)
        jh = self.hyper_jh(hyper_output, jh)
        fh = self.hyper_fh(hyper_output, fh)
        oh = self.hyper_oh(hyper_output, oh)
        
        # Split bias
        ib, jb, fb, ob = torch.split(self.bias, self.hidden_size)
        ib = self.hyper_ib(hyper_output, ib.unsqueeze(0).expand(batch_size, -1))
        jb = self.hyper_jb(hyper_output, jb.unsqueeze(0).expand(batch_size, -1))
        fb = self.hyper_fb(hyper_output, fb.unsqueeze(0).expand(batch_size, -1))
        ob = self.hyper_ob(hyper_output, ob.unsqueeze(0).expand(batch_size, -1))
        
        # Combine the gates
        i = ix + ih + ib
        j = jx + jh + jb
        f = fx + fh + fb
        o = ox + oh + ob
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            concat = torch.cat([i, j, f, o], dim=1)
            concat = layer_norm_all(concat, batch_size, 4, self.hidden_size, 
                                   self.ln_gamma, self.ln_beta)
            i, j, f, o = torch.split(concat, self.hidden_size, dim=1)
        
        # Apply recurrent dropout if enabled
        if self.use_recurrent_dropout:
            j_activation = F.dropout(torch.tanh(j), p=1-self.dropout_keep_prob, training=self.training)
        else:
            j_activation = torch.tanh(j)
        
        # Compute new cell state
        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * j_activation
        
        # Apply layer norm to cell state if enabled
        if self.use_layer_norm:
            new_c_normalized = layer_norm_all(new_c.unsqueeze(1), batch_size, 1, 
                                           self.hidden_size, self.ln_c_gamma, 
                                           self.ln_c_beta).squeeze(1)
            new_h = torch.tanh(new_c_normalized) * torch.sigmoid(o)
        else:
            new_h = torch.tanh(new_c) * torch.sigmoid(o)
        
        # Combine main and hyper states
        hyper_new_c, hyper_new_h = hyper_new_state
        new_total_c = torch.cat([new_c, hyper_new_c], dim=1)
        new_total_h = torch.cat([new_h, hyper_new_h], dim=1)
        
        return new_h, (new_total_c, new_total_h)

class HyperLSTM(nn.Module):
    """Full HyperLSTM model for character-level language modeling"""
    def __init__(self, input_size, embedding_size, hidden_size, output_size,
                num_layers=1, dropout=0.0, hyper_num_units=128,
                hyper_embedding_size=16, use_layer_norm=True,
                use_recurrent_dropout=False, dropout_keep_prob=0.9):
        super(HyperLSTM, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Character embedding
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Define layers
        self.layers = nn.ModuleList()
        
        # First layer takes embeddings as input
        self.layers.append(
            HyperLSTMCell(embedding_size, hidden_size, 
                         hyper_num_units=hyper_num_units,
                         hyper_embedding_size=hyper_embedding_size,
                         use_layer_norm=use_layer_norm,
                         use_recurrent_dropout=use_recurrent_dropout,
                         dropout_keep_prob=dropout_keep_prob)
        )
        
        # Additional layers
        for _ in range(1, num_layers):
            self.layers.append(
                HyperLSTMCell(hidden_size, hidden_size,
                             hyper_num_units=hyper_num_units,
                             hyper_embedding_size=hyper_embedding_size,
                             use_layer_norm=use_layer_norm,
                             use_recurrent_dropout=use_recurrent_dropout,
                             dropout_keep_prob=dropout_keep_prob)
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize parameters"""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize output layer
        nn.init.uniform_(self.output_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, input_seq, initial_state=None):
        """
        Forward pass through the network
        
        Args:
            input_seq: [seq_len, batch_size] tensor of character indices
            initial_state: Initial hidden state
            
        Returns:
            output: [seq_len, batch_size, output_size] tensor of output logits
            final_state: Final hidden state
        """
        # 处理不同维度的输入 (可能是 [seq_len, batch_size] 或 [1, 1, 1])
        if input_seq.dim() == 3:  # For generate_text function
            seq_len, batch_size, _ = input_seq.size()
        else:  # For normal training
            seq_len, batch_size = input_seq.size()
        
        # Convert character indices to embeddings
        embedded = self.embedding(input_seq)  # [seq_len, batch_size, embedding_size]
        
        # Initialize states if not provided
        if initial_state is None:
            total_hidden_size = self.hidden_size + self.layers[0].hyper_num_units
            initial_state = []
            for _ in range(self.num_layers):
                layer_state = (
                    torch.zeros(batch_size, total_hidden_size, device=embedded.device),
                    torch.zeros(batch_size, total_hidden_size, device=embedded.device)
                )
                initial_state.append(layer_state)
        
        # Process each time step
        states = initial_state
        outputs = []
        
        for t in range(seq_len):
            input_t = embedded[t]  # [batch_size, embedding_size]
            
            # Apply dropout between layers
            for i, layer in enumerate(self.layers):
                if i > 0 and self.dropout > 0:
                    input_t = F.dropout(input_t, p=self.dropout, training=self.training)
                
                # Run layer
                output_t, new_state = layer(input_t, states[i])
                states[i] = new_state
                input_t = output_t  # Next layer's input
            
            # Final output for this time step
            logits_t = self.output_layer(output_t)  # [batch_size, output_size]
            outputs.append(logits_t)
        
        # Stack all time step outputs
        outputs = torch.stack(outputs)  # [seq_len, batch_size, output_size]
        
        return outputs, states
    
    def init_hidden(self, batch_size, device=torch.device('cpu')):
        """Initialize hidden state"""
        total_hidden_size = self.hidden_size + self.layers[0].hyper_num_units
        states = []
        for _ in range(self.num_layers):
            layer_state = (
                torch.zeros(batch_size, total_hidden_size, device=device),
                torch.zeros(batch_size, total_hidden_size, device=device)
            )
            states.append(layer_state)
        return states

class CharDataset(Dataset):
    """Character-level dataset for language modeling"""
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = {ch: i for i, ch in enumerate(sorted(set(text)))}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Create examples
        self.data = []
        for i in range(0, len(text) - seq_length, 1):
            input_seq = text[i:i+seq_length]
            target_seq = text[i+1:i+seq_length+1]
            self.data.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        
        # Convert to indices
        input_tensor = torch.tensor([self.char_to_idx[ch] for ch in input_seq], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_idx[ch] for ch in target_seq], dtype=torch.long)
        
        return input_tensor, target_tensor

def train_char_lm(model, data_loader, optimizer, device, clip_value=100.0):
    """Train the language model for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        # Move to device
        input_batch = input_batch.to(device).transpose(0, 1)  # (seq_len, batch_size)
        target_batch = target_batch.to(device).transpose(0, 1)  # (seq_len, batch_size)
        
        # Forward pass
        hidden = model.init_hidden(input_batch.size(1), device)
        optimizer.zero_grad()
        output, _ = model(input_batch, hidden)
        
        # Reshape for loss calculation
        output = output.view(-1, model.output_size)
        target = target_batch.reshape(-1)
        
        # Calculate loss
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += loss.item() * input_batch.size(1)
        
        # Print progress
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            print(f'Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s')
            start_time = time.time()
    
    return total_loss / len(data_loader.dataset)

def evaluate(model, data_loader, device):
    """Evaluate the language model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            # Move to device
            input_batch = input_batch.to(device).transpose(0, 1)  # (seq_len, batch_size)
            target_batch = target_batch.to(device).transpose(0, 1)  # (seq_len, batch_size)
            
            # Forward pass
            hidden = model.init_hidden(input_batch.size(1), device)
            output, _ = model(input_batch, hidden)
            
            # Reshape for loss calculation
            output = output.view(-1, model.output_size)
            target = target_batch.reshape(-1)
            
            # Calculate loss
            loss = F.cross_entropy(output, target)
            
            # Track loss
            total_loss += loss.item() * input_batch.size(1)
    
    return total_loss / len(data_loader.dataset)
