import numpy as np
import matplotlib.pyplot as plt
from histogram_3d_visualization import Histogram3DVisualizer

def main():
    """
    Automatic Glass Sheet 3D visualization - no user interaction required
    """
    print("=== 3D Glass Sheet Histogram Visualization ===")
    print("Loading data...")
    
    # Initialize the visualizer
    visualizer, skip_generations= Histogram3DVisualizer('fitness_histograms_Navix_Empty_Random_6x6_v0_40gen_3000pop.npz'), 1
    # visualizer, skip_generations = Histogram3DVisualizer('fitness_histograms_cifar100_256h_1000gen_2000pop.npz'), 50
    # visualizer, skip_generations = Histogram3DVisualizer('fitness_histograms_cifar10_256h_4000gen_2000pop.npz'), 50

    # Show summary
    visualizer.create_summary_statistics()
    
    print("\n=== Creating Glass Sheet View ===")
    
    print(f"Generating 3D glass sheet visualization (every {skip_generations} generation(s))...")
    
    # Automatically create and save the glass sheet view with skip parameter
    visualizer.create_glass_sheet_view(save_path='glass_sheet_visualization.png', skip_generations=skip_generations)
    
    print("✅ Glass sheet visualization completed!")
    print("📁 Saved as: glass_sheet_visualization.png")
    print("🎯 The 3D view has been displayed and saved automatically.")
    print(f"📊 Used skip_generations={skip_generations} to reduce visual complexity.")
    
    # 新功能：绘制所有 generation 的最大 fitness 折线图
    print("\n=== Creating Maximum Fitness Evolution Plot ===")
    create_max_fitness_plot(visualizer)
    print("✅ Maximum fitness evolution plot completed!")
    print("📁 Saved as: max_fitness_evolution.png")
    
    # 新功能：绘制所有 generation 的平均 fitness 折线图
    print("\n=== Creating Average Fitness Evolution Plot ===")
    create_average_fitness_plot(visualizer)
    print("✅ Average fitness evolution plot completed!")
    print("📁 Saved as: average_fitness_evolution.png")

def create_max_fitness_plot(visualizer):
    """
    创建所有 generation 的最大 fitness 值折线图
    
    Args:
        visualizer: Histogram3DVisualizer 实例
    """
    # 从数据中获取最大 fitness 值
    max_vals = visualizer.data['max_vals']
    generations = visualizer.generations
    
    # 创建 2D 折线图
    plt.figure(figsize=(12, 8))
    
    # 绘制折线图
    plt.plot(generations, max_vals, linewidth=2.5, color='red', marker='o', 
             markersize=6, markerfacecolor='darkred', markeredgewidth=0, 
             alpha=0.9, label='Maximum Fitness')
    
    # 自定义图表
    plt.title('Maximum Fitness Evolution Over Generations', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Generation', fontsize=14, fontweight='bold')
    plt.ylabel('Maximum Fitness Value', fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 添加统计信息
    initial_max = max_vals[0]
    final_max = max_vals[-1]
    improvement = final_max - initial_max
    

    
    # 添加改进信息文本框
    textstr = f'Fitness Improvement: {improvement:.2f}\nTotal Generations: {len(generations)}\nRange: [{min(max_vals):.2f}, {max(max_vals):.2f}]'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=10)
    
    # 设置坐标轴范围
    plt.xlim(min(generations) - 0.5, max(generations) + 0.5)
    y_range = max(max_vals) - min(max_vals)
    plt.ylim(min(max_vals) - y_range * 0.1, max(max_vals) + y_range * 0.1)
    
    # 美化坐标轴
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('max_fitness_evolution.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print(f"\n=== Maximum Fitness Statistics ===")
    print(f"Initial maximum fitness: {initial_max:.3f}")
    print(f"Final maximum fitness: {final_max:.3f}")
    print(f"Total improvement: {improvement:.3f}")
    print(f"Average maximum fitness: {np.mean(max_vals):.3f}")
    print(f"Standard deviation: {np.std(max_vals):.3f}")

def create_average_fitness_plot(visualizer):
    """
    创建所有 generation 的平均 fitness 值折线图
    
    Args:
        visualizer: Histogram3DVisualizer 实例
    """
    # 计算每个generation的平均fitness值
    average_vals = []
    for i in range(len(visualizer.generations)):
        # 使用bin_centers和histogram_tensor计算加权平均
        bin_centers = visualizer.bin_centers[i]
        frequencies = visualizer.histogram_tensor[i]
        
        # 计算加权平均fitness
        total_weight = np.sum(frequencies)
        if total_weight > 0:
            weighted_avg = np.sum(bin_centers * frequencies) / total_weight
        else:
            weighted_avg = 0
        average_vals.append(weighted_avg)
    
    average_vals = np.array(average_vals)
    generations = visualizer.generations
    
    # 创建 2D 折线图
    plt.figure(figsize=(12, 8))
    
    # 绘制折线图
    plt.plot(generations, average_vals, linewidth=2.5, color='red', marker='o', 
             markersize=6, markerfacecolor='darkred', markeredgewidth=0, 
             alpha=0.9, label='Average Fitness')
    
    
    # 自定义图表
    plt.title('Average Fitness Evolution Over Generations', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Generation', fontsize=14, fontweight='bold')
    plt.ylabel('Average Fitness Value', fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 添加统计信息
    initial_avg = average_vals[0]
    final_avg = average_vals[-1]
    improvement = final_avg - initial_avg
    
    # 添加改进信息文本框
    textstr = f'Fitness Improvement: {improvement:.2f}\nTotal Generations: {len(generations)}\nRange: [{min(average_vals):.2f}, {max(average_vals):.2f}]'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=10)
    
    # 设置坐标轴范围
    plt.xlim(min(generations) - 0.5, max(generations) + 0.5)
    y_range = max(average_vals) - min(average_vals)
    plt.ylim(min(average_vals) - y_range * 0.1, max(average_vals) + y_range * 0.1)
    
    # 美化坐标轴
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('average_fitness_evolution.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print(f"\n=== Average Fitness Statistics ===")
    print(f"Initial average fitness: {initial_avg:.3f}")
    print(f"Final average fitness: {final_avg:.3f}")
    print(f"Total improvement: {improvement:.3f}")
    print(f"Average of average fitness: {np.mean(average_vals):.3f}")
    print(f"Standard deviation: {np.std(average_vals):.3f}")


if __name__ == "__main__":
    main()
