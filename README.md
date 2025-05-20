# 超网络 (HyperNetworks) 项目

本项目基于论文 ["HyperNetworks"](https://arxiv.org/pdf/1609.09106.pdf) (Ha, Dai & Le, 2016)，实现了静态超网络和动态超网络（HyperLSTM）模型。静态超网络应用于MNIST图像分类任务，而动态超网络（HyperLSTM）应用于字符级语言建模任务。项目提供了PyTorch和TensorFlow实现。

## 📝 目录

- [项目概述](#项目概述)
- [超网络理论](#超网络理论)
  - [基本概念](#基本概念)
  - [静态超网络](#静态超网络)
  - [动态超网络](#动态超网络)
- [实现一：静态超网络用于图像分类](#实现一静态超网络用于图像分类)
  - [静态超网络实现细节](#静态超网络实现细节)
  - [参数比较](#参数比较)
  - [实验结果](#静态超网络实验结果)
- [实现二：HyperLSTM用于序列建模](#实现二hyperlstm用于序列建模)
  - [HyperLSTM架构](#hyperlstm架构)
  - [HyperLSTM实现细节](#hyperlstm实现细节)
  - [实验结果](#hyperlstm实验结果)
- [使用指南](#使用指南)
  - [环境配置](#环境配置)
  - [运行静态超网络](#运行静态超网络)
  - [运行HyperLSTM](#运行hyperlstm)
- [参考资料](#参考资料)
- [许可证](#许可证)

## 项目概述

超网络是一种元学习架构，其核心思想是"使用一个网络生成另一个网络的权重"。在传统神经网络中，权重是通过反向传播直接学习的参数；而在超网络中，权重是由另一个网络（称为超网络）动态生成的。

本项目实现了论文中的两种超网络变体：

1. **静态超网络**：生成CNN的卷积层权重，应用于MNIST图像分类任务
2. **动态超网络（HyperLSTM）**：使用辅助LSTM动态生成主LSTM权重，应用于字符级语言建模任务

这两种实现展示了超网络在不同任务类型上的应用和优势。

## 超网络理论

### 基本概念

在论文中，超网络被定义为一个生成另一个网络权重的网络。如果我们将主网络表示为函数 f(x; θ)，其中θ是权重参数，那么超网络可以表示为 g(z; φ)，其中:

- f 是主网络（被生成的网络）
- g 是超网络（生成权重的网络）
- x 是主网络的输入
- z 是超网络的输入
- θ = g(z; φ) 是由超网络生成的主网络权重
- φ 是超网络自己的权重参数

这种方法允许通过一个小型网络间接参数化一个可能更大的网络，从而提供更紧凑的模型表示和更灵活的权重生成机制。

### 静态超网络

静态超网络是超网络的一种特殊情况，其中输入 z 是可学习的参数而非外部输入。静态超网络的工作流程如下：

1. 初始化一个潜在向量 z（低维）
2. z 通过超网络 g 转换为权重参数 θ
3. 主网络 f 使用生成的权重 θ 进行正常的前向传播
4. 在反向传播中，梯度流经主网络到超网络，同时更新 z、超网络参数 φ 和主网络中的其他参数

![静态超网络架构](assets/static_hyper_network.svg)

### 动态超网络

动态超网络将输入 z 视为外部输入（如当前网络状态或输入特征的函数），这使得网络能够为每个输入动态生成不同的权重。这在处理序列数据等任务中特别有用，因为它允许模型根据上下文动态调整其参数。

动态超网络的一个重要实现是HyperLSTM，它使用一个小型LSTM（称为hyper-LSTM）生成主LSTM的权重，特别适合需要根据上下文动态调整处理策略的任务，如复杂的文本建模。

## 实现一：静态超网络用于图像分类

### 静态超网络实现细节

在PyTorch实现中，静态超网络模型定义在`HyperCNN`类中。其核心组件包括：

#### 1. 潜在向量 z (z_signal)

```python
# 论文3.1节提到，静态超网络的输入是一个固定的潜在向量z
self.z_signal = nn.Parameter(torch.randn(1, z_dim) * 0.01)
```

这是超网络的输入，是一个可学习的参数，在训练过程中会随着梯度更新。z_dim控制了超网络的表达能力，在本实现中默认设为4。

#### 2. 超网络层

```python
# 第一层变换 - z → 中间表示
self.w2 = nn.Parameter(torch.randn(z_dim, in_size * z_dim) * 0.01)
self.b2 = nn.Parameter(torch.zeros(in_size * z_dim))

# 第二层变换 - 中间表示 → 最终权重
self.w1 = nn.Parameter(torch.randn(z_dim, out_size * f_size * f_size) * 0.01)
self.b1 = nn.Parameter(torch.zeros(out_size * f_size * f_size))
```

超网络由两层线性变换组成，将低维潜在向量 z 映射到高维权重空间。

#### 3. 权重生成函数

```python
def generate_conv2_weights(self):
    """使用超网络生成第二个卷积层的权重"""
    # 步骤1: z_signal通过第一层变换
    h_in = torch.matmul(self.z_signal, self.w2) + self.b2
    h_in = h_in.reshape(self.in_size, self.z_dim)
    
    # 步骤2: 中间表示通过第二层变换
    h_final = torch.matmul(h_in, self.w1) + self.b1
    
    # 步骤3: 重塑为卷积核格式
    kernel = h_final.reshape(self.out_size, self.in_size, self.f_size, self.f_size)
    return kernel
```

这个函数实现了超网络生成主网络权重的核心逻辑：从潜在向量 z 开始，经过两次线性变换，最终生成卷积权重矩阵。

### 参数比较

论文第4.2节讨论了超网络如何减少参数数量。以我们的实现为例，标准CNN和超网络CNN的参数数量如下：

**标准CNN参数**:
- 第二个卷积层：7×7×16×16 = 12,544个参数

**超网络参数**:
- 潜在向量 z：4个参数
- 第一层变换：4×(16×4) + 16×4 = 320个参数
- 第二层变换：4×(16×7×7) + 16×7×7 = 3,200个参数
- 总计：3,524个参数

虽然在这个简单示例中，超网络的参数减少不是很显著，但在更大的模型中，参数减少可能会更加明显。

### 静态超网络实验结果

在MNIST数据集上，标准CNN和超网络CNN都能达到类似的分类性能。我们的实验表明：

1. 标准CNN在训练早期通常收敛更快
2. 超网络CNN最终可以达到与标准CNN相当的测试精度
3. 超网络CNN在参数数量上有一定优势

通过可视化生成的卷积滤波器，我们可以观察到：

1. 标准CNN学习的滤波器通常更加分散和多样化
2. 超网络生成的滤波器展现出更多的结构和规律性
3. 这种差异体现了超网络通过潜在空间的低维表示施加的隐式正则化

## 实现二：HyperLSTM用于序列建模

### HyperLSTM架构

HyperLSTM 是超网络的一种动态实现，其中权重由一个辅助网络根据当前输入和状态动态生成。HyperLSTM 的核心思想是：

1. **动态权重生成**：使用一个小型 LSTM（称为 hyper-LSTM）根据当前的输入和状态动态生成主 LSTM 的权重。

2. **权重调制策略**：原论文中提出了两种权重生成策略：
   - **权重缩放**（Weight Scaling）：保留原始权重，但通过生成的因子进行缩放调制
   - **全权重生成**（Full Weight Generation）：直接生成替换原始权重的新权重

3. **权重分解**（Weight Factorization）：为了减少参数数量和计算量，超网络通过一个两级映射系统生成权重：首先将输入映射到低维嵌入空间，然后从嵌入空间映射到目标权重空间。

HyperLSTM的基本结构如下图所示：

```
┌────────────────────────────────────────────────────────────────┐
│                         HyperLSTM Cell                         │
│                                                                │
│  ┌────────────┐                                                │
│  │            │                                                │
│  │ Input (x)  │                                                │
│  │            │                                                │
│  └─────┬──────┘                                                │
│        │                                                       │
│        ▼                                                       │
│  ┌────────────┐        ┌────────────┐        ┌────────────┐    │
│  │            │        │            │        │            │    │
│  │ Hyper-LSTM ├───────►│HyperLinear ├───────►│ Main LSTM  │    │
│  │            │        │            │        │            │    │
│  └────────────┘        └────────────┘        └─────┬──────┘    │
│                                                    │           │
│                                                    ▼           │
│                                             ┌────────────┐     │
│                                             │            │     │
│                                             │  Output    │     │
│                                             │            │     │
│                                             └────────────┘     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### HyperLSTM实现细节

我们的HyperLSTM实现包含以下核心组件：

#### 1. HyperLSTMCell

HyperLSTM单元是整个模型的核心，它包含主LSTM和辅助的hyper-LSTM。它的工作流程是：

1. 辅助LSTM（hyper-LSTM）接收输入和上一时刻主LSTM的隐藏状态
2. 辅助LSTM生成动态参数
3. 这些动态参数通过HyperLinear模块调制主LSTM的权重
4. 调制后的主LSTM处理当前输入并生成输出

#### 2. HyperLinear模块

HyperLinear模块实现了权重分解的两级映射系统：

```python
class HyperLinear(nn.Module):
    """
    Hypernetwork for dynamic weight generation
    """
    def __init__(self, input_size, embedding_size, output_size, use_bias=True):
        super(HyperLinear, self).__init__()
        
        # 第一级映射：z -> embedding（将hyper LSTM的输出映射到低维嵌入空间）
        self.z_linear = SuperLinear(input_size, embedding_size, 
                                    init_w="gaussian", weight_start=0.01,
                                    use_bias=True, bias_start=1.0)
        
        # 第二级映射：embedding -> output weights（从嵌入空间生成目标权重）
        self.weight_linear = SuperLinear(embedding_size, output_size, 
                                        init_w="constant", weight_start=0.1/embedding_size,
                                        use_bias=False)
```

#### 3. 增强技术

为了提高模型性能，我们实现了几种增强技术：

- **层归一化**（Layer Normalization）：稳定训练过程，减少内部协变量偏移
- **正交初始化**（Orthogonal Initialization）：改善梯度流
- **循环 Dropout**（Recurrent Dropout）：增强正则化，防止过拟合

### HyperLSTM实验结果

在字符级语言建模任务上，HyperLSTM相比标准LSTM表现出明显的优势：

1. **长期依赖处理**：通过动态调整权重，能够更好地保持和利用长序列中的远距离信息
2. **上下文敏感性**：对输入序列的变化更敏感，能够自适应调整处理策略
3. **泛化能力**：在有限训练数据上表现出更好的泛化能力
4. **表现力**：尽管参数总量可能相似，但动态权重生成机制大大增强了模型的表达能力

## 使用指南

### 环境配置

本项目需要以下环境：

```bash
conda create -n sae python=3.8
conda activate sae

# 基本依赖
pip install torch torchvision matplotlib numpy

# 对于TensorFlow版本
pip install tensorflow==2.8.0 keras
```

### 运行静态超网络

**PyTorch版本**:

```bash
python mnist_hypernetwork_pytorch.py
```

**TensorFlow版本**:

```bash
python mnist_hypernetwork_simplified.py
```

这些脚本会训练标准CNN和超网络CNN，并生成权重可视化结果。

### 运行HyperLSTM

**训练模型**:

```bash
python hyperlstm_train.py --mode train \
    --data_path data/your_data.txt \
    --epochs 20 \
    --seq_length 100 \
    --batch_size 32 \
    --hidden_size 512 \
    --hyper_num_units 128 \
    --hyper_embedding_size 16 \
    --use_layer_norm
```

**生成文本**:

```bash
python hyperlstm_train.py --mode generate \
    --data_path data/your_data.txt \
    --prime_text "The" \
    --gen_length 1000 \
    --temperature 0.8 \
    --load_model models/hyperlstm_best.pt
```

**评估模型**:

```bash
python hyperlstm_train.py --mode eval \
    --data_path data/your_data.txt \
    --load_model models/hyperlstm_best.pt
```

#### HyperLSTM参数说明

##### 数据参数
- `--data_path`: 训练数据文件路径
- `--seq_length`: 训练序列长度
- `--batch_size`: 批量大小
- `--val_split`: 验证集比例

##### 模型参数
- `--embedding_size`: 字符嵌入大小
- `--hidden_size`: 主 LSTM 的隐藏层大小
- `--num_layers`: LSTM 层数
- `--dropout`: 层间 Dropout 概率
- `--hyper_num_units`: HyperLSTM 辅助网络中的单元数量
- `--hyper_embedding_size`: 超网络嵌入空间的大小
- `--use_layer_norm`: 启用层归一化
- `--use_recurrent_dropout`: 启用循环 Dropout
- `--dropout_keep_prob`: 循环 Dropout 的保留概率

##### 训练参数
- `--epochs`: 训练轮数
- `--lr`: 初始学习率
- `--lr_decay`: 学习率衰减因子
- `--patience`: 学习率调度的耐心值
- `--clip`: 梯度裁剪值

##### 生成参数
- `--temperature`: 生成文本时的温度参数（控制随机性，越高越随机）
- `--gen_length`: 生成文本的长度
- `--prime_text`: 用于启动生成的种子文本

## 参考资料

1. [HyperNetworks论文](https://arxiv.org/pdf/1609.09106.pdf) - David Ha, Andrew Dai, Quoc V. Le (2016)
2. [原始TensorFlow实现](https://github.com/hardmaru/supercell/blob/master/supercell.py) - David Ha

## 许可证

MIT

---

*注：此项目为学术研究和教育目的而创建。*

## 更新（2025年5月20日）

### 扩展实现：三z_signal共享超网络模型

作为对原始超网络实现的扩展，我们增加了一个使用**单一共享超网络**解码**三个独立z_signal**的改进模型。这一实现更加贴近原论文中描述的超网络概念。

#### 改进架构

新的实现(`mnist_hypernetwork_triple_shared.py`)具有以下特点：

1. **三个独立z_signal**：
   - `z_signal_0`: 用于生成第一卷积层(conv1)权重
   - `z_signal_1`: 用于生成第二卷积层(conv2)权重
   - `z_signal_2`: 用于生成第三卷积层(conv3)权重

2. **共享超网络核心**：
   ```python
   # 所有z_signal共享一个超网络的核心层
   self.hyper_w1 = nn.Parameter(torch.randn(z_dim, h_dim) * 0.01)
   self.hyper_b1 = nn.Parameter(torch.zeros(h_dim))
   ```
   这一核心超网络层将所有z_signal从4维映射到共同的64维隐藏表示空间。

3. **层特定输出投影**：
   ```python
   # 为不同卷积层使用专门的输出投影
   self.hyper_w2_conv1 = nn.Parameter(torch.randn(h_dim, in_size * f_size * f_size) * 0.01)
   self.hyper_w2_conv2 = nn.Parameter(torch.randn(h_dim, mid_size * in_size * f_size * f_size) * 0.01)
   self.hyper_w2_conv3 = nn.Parameter(torch.randn(h_dim, out_size * mid_size * f_size * f_size) * 0.01)
   ```
   从共享隐藏表示到各层特定维度权重的最终映射。

4. **统一权重生成流程**：
   ```python
   def generate_weights(self, z_signal, hyper_w2, hyper_b2, out_shape):
       # 第一层（共享）：z -> 隐藏表示
       h = torch.matmul(z_signal, self.hyper_w1) + self.hyper_b1
       
       # 第二层（层特定）：隐藏表示 -> 权重
       weights = torch.matmul(h, hyper_w2) + hyper_b2
       
       # 重塑为卷积滤波器格式
       kernel = weights.reshape(out_shape)
       return kernel
   ```

#### 实验结果

共享超网络模型展现出了优秀的性能特性：

1. **测试错误率**：
   - 标准CNN: 1.02%
   - 共享超网络CNN: 0.95%

2. **z_signal与权重的关系**：
   - 每个4维z_signal能够产生大量的卷积滤波器权重：
     - z_signal_0 (4维) → 生成400个参数 (16×1×5×5)
     - z_signal_1 (4维) → 生成12,800个参数 (32×16×5×5)
     - z_signal_2 (4维) → 生成51,200个参数 (64×32×5×5)

3. **可视化**：
   - **z_signal柱状图**：展示了三个z_signal的相对值，每个z_signal学习到独特的潜在表示
   - **损失曲线比较**：共享超网络模型的收敛速度和稳定性都有所提升
   - **滤波器可视化**：共享超网络生成的滤波器展现出更多结构化的特征

#### 模型文件

此实现包含了模型保存和加载功能：

```python
def save_model(model, save_path='hypernetwork_model.pt'):
    """保存模型到文件"""
    torch.save(model.state_dict(), save_path)
    
    # 单独保存z_signal以便后续分析
    z_signals = {
        'z_signal_0': model.z_signal_0.detach().cpu().numpy(),
        'z_signal_1': model.z_signal_1.detach().cpu().numpy(),
        'z_signal_2': model.z_signal_2.detach().cpu().numpy()
    }
    np.save('z_signals.npy', z_signals)
```

#### 结论

这一扩展实现展示了超网络架构的灵活性和强大潜力。通过单一共享超网络从多个独立z_signal生成多层卷积权重，我们不仅实现了参数的高效利用，还略微提升了模型性能。特别是，z_signal作为低维潜在表示，提供了对卷积层特性的高度抽象编码，这种编码在共享超网络的映射下转化为完整的权重矩阵。

这一实验验证了论文中提出的超网络作为"一种使用另一个网络生成权重的方法"的核心理念，并展示了它在深度卷积网络中的实用性。
