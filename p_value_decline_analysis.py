# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def analyze_p_value_decline():
    """
    分析P值下降现象的根本原因
    """
    
    print("=" * 80)
    print("P值下降现象分析 - 早熟收敛问题")
    print("=" * 80)
    
    print("\n🔍 关键差异对比：")
    print("-" * 50)
    
    print("【原版本 - genetic_algorithm_experiment_english.py】:")
    print("  ✓ 种群无限增长：10 → 55 → 100 → ... → 4,510")
    print("  ✓ 保留所有个体（包括低适应度个体）")
    print("  ✓ 种群多样性丰富")
    print("  ✓ P值保持相对稳定（约0.22）")
    
    print("\n【优化版本 - genetic_algorithm_experiment_optimized.py】:")
    print("  ⚠️  种群规模受限：最多200个个体")
    print("  ⚠️  只保留适应度最高的个体（精英选择）")
    print("  ⚠️  种群多样性急剧减少")
    print("  ⚠️  P值逐渐下降到0（早熟收敛）")
    
    print("\n🧬 生物学原理解释：")
    print("-" * 30)
    print("这是遗传算法中的【早熟收敛】(Premature Convergence)现象：")
    print("  1. 精英选择策略只保留高适应度个体")
    print("  2. 种群基因多样性快速丧失")
    print("  3. 所有个体趋向于相似的高适应度")
    print("  4. 交叉操作难以产生更优子代")
    print("  5. P值趋向于0")
    
    # 模拟适应度演化过程
    print("\n📊 适应度演化模拟：")
    print("-" * 25)
    
    # 原版本：多样性保持
    print("原版本适应度分布（第100代）:")
    print("  [150, 180, 200, 220, 250, 280, 300, 320, 350, ...]")
    print("  → 多样性丰富，仍有改进空间")
    
    # 优化版本：收敛到高值
    print("\n优化版本适应度分布（第100代）:")
    print("  [410, 412, 415, 416, 417, 417, 418, 418, 419, ...]")
    print("  → 高度同质化，难以进一步改进")
    
    print("\n⚡ P值下降的数学原因：")
    print("-" * 30)
    print("P = (子代适应度 > 两个父代适应度的个体数) / 总子代数")
    print()
    print("原版本场景：")
    print("  父代1: 200, 父代2: 250 → 子代: 225")
    print("  225 > 200 且 225 > 250? → 否")
    print("  父代1: 180, 父代2: 280 → 子代: 230")
    print("  230 > 180 且 230 > 280? → 否")
    print("  但由于父代适应度差异大，仍有概率产生更优子代")
    print()
    print("优化版本场景（后期）：")
    print("  父代1: 417, 父代2: 418 → 子代: 417.5")
    print("  417.5 > 417 且 417.5 > 418? → 否")
    print("  父代1: 416, 父代2: 419 → 子代: 417.5")
    print("  417.5 > 416 且 417.5 > 419? → 否")
    print("  由于父代都是高适应度，子代几乎不可能同时超过两个父代")
    
    # 可视化分析
    plt.figure(figsize=(15, 10))
    
    # 模拟P值变化趋势
    generations = np.arange(1, 101)
    
    # 原版本：相对稳定
    original_p = 0.22 + 0.05 * np.sin(generations * 0.1) + np.random.normal(0, 0.02, 100)
    original_p = np.clip(original_p, 0, 1)
    
    # 优化版本：逐渐下降
    optimized_p = 0.25 * np.exp(-generations/40) + 0.05 * np.random.normal(0, 0.01, 100)
    optimized_p = np.clip(optimized_p, 0, 1)
    
    plt.subplot(2, 2, 1)
    plt.plot(generations, original_p, 'b-', linewidth=2, label='Original Version', alpha=0.8)
    plt.plot(generations, optimized_p, 'r-', linewidth=2, label='Optimized Version', alpha=0.8)
    plt.title('P Value Comparison', fontsize=14)
    plt.xlabel('Generation')
    plt.ylabel('P Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Population diversity simulation
    plt.subplot(2, 2, 2)
    original_diversity = 100 + 20 * np.sin(generations * 0.05)  # Maintain diversity
    optimized_diversity = 100 * np.exp(-generations/30)  # Diversity loss
    
    plt.plot(generations, original_diversity, 'b-', linewidth=2, label='Original Version', alpha=0.8)
    plt.plot(generations, optimized_diversity, 'r-', linewidth=2, label='Optimized Version', alpha=0.8)
    plt.title('Population Diversity Changes', fontsize=14)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Average fitness changes
    plt.subplot(2, 2, 3)
    original_fitness = 245 + 3 * generations  # Slow improvement
    optimized_fitness = 245 + 170 * (1 - np.exp(-generations/20))  # Fast saturation
    
    plt.plot(generations, original_fitness, 'b-', linewidth=2, label='Original Version', alpha=0.8)
    plt.plot(generations, optimized_fitness, 'r-', linewidth=2, label='Optimized Version', alpha=0.8)
    plt.title('Average Fitness Changes', fontsize=14)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Improvement potential analysis
    plt.subplot(2, 2, 4)
    improvement_potential_original = 50 - 0.2 * generations  # Maintain improvement potential
    improvement_potential_optimized = 50 * np.exp(-generations/25)  # Rapid decline in improvement potential
    
    plt.plot(generations, np.clip(improvement_potential_original, 0, 50), 'b-', 
             linewidth=2, label='Original Version', alpha=0.8)
    plt.plot(generations, improvement_potential_optimized, 'r-', 
             linewidth=2, label='Optimized Version', alpha=0.8)
    plt.title('Improvement Potential Changes', fontsize=14)
    plt.xlabel('Generation')
    plt.ylabel('Improvement Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 解决方案建议：")
    print("-" * 20)
    print("1. 【多样性保持策略】:")
    print("   - 不要只选择最优个体")
    print("   - 采用轮盘赌选择或锦标赛选择")
    print("   - 保留一定比例的中等适应度个体")
    
    print("\n2. 【精英主义 + 多样性平衡】:")
    print("   - 保留20%精英个体")
    print("   - 随机保留30%中等个体")
    print("   - 重新生成50%新个体")
    
    print("\n3. 【变异操作增强】:")
    print("   - 增加变异概率")
    print("   - 适应性变异率（早期高，后期低）")
    
    print("\n4. 【岛屿模型】:")
    print("   - 分多个子种群独立演化")
    print("   - 定期进行种群间个体交换")
    
    print("\n📈 实验结论：")
    print("-" * 15)
    print("✅ 原版本虽然计算慢，但保持了种群多样性")
    print("❌ 优化版本虽然计算快，但存在早熟收敛问题")
    print("🎯 最佳策略：性能优化 + 多样性保持的平衡设计")

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    analyze_p_value_decline()
