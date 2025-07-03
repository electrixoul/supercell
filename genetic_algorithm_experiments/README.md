# 遗传算法实验

这个文件夹包含了所有遗传算法相关的实验代码和分析工具。

## 📁 文件说明

### 主要实验程序
- `genetic_algorithm_experiment.py` - 基础遗传算法实验
- `genetic_algorithm_comparison_experiment.py` - 有/无交叉操作的对比实验（**推荐使用**）
- `genetic_algorithm_experiment_no_crossover.py` - 无交叉操作的遗传算法实验
- `genetic_algorithm_experiment_optimized.py` - 优化版遗传算法实验
- `genetic_algorithm_experiment_selective_retention.py` - 选择性保留机制实验

### 数据分析工具
- `genetic_algorithm_data_plotter.py` - 数据可视化和曲线拟合分析工具

## 🚀 使用方法

### 运行对比实验
```bash
python genetic_algorithm_comparison_experiment.py
```

### 分析实验数据
```bash
python genetic_algorithm_data_plotter.py
```

## 📊 实验结果

最新的实验结果显示：
- **P值效率比**: 11.7x（有交叉 vs 无交叉）
- **适应度改进比**: 32.8x（有交叉 vs 无交叉）
- **关键改进**: 随机交叉点替代固定交叉点，显著提升算法性能

## 🔬 研究发现

1. **交叉操作的重要性**: 交叉操作显著提升遗传算法的收敛速度和优化效果
2. **交叉点多样化**: 使用随机交叉点比固定交叉点效果更好
3. **遗传多样性**: 交叉点多样化增强了种群的遗传多样性

## 📈 数据文件

实验数据文件（.npz格式）已被添加到 .gitignore 中，不会被版本控制跟踪。
