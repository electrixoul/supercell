import numpy as np
from histogram_3d_visualization import Histogram3DVisualizer

def main():
    """
    Automatic Glass Sheet 3D visualization - no user interaction required
    """
    print("=== 3D Glass Sheet Histogram Visualization ===")
    print("Loading data...")
    
    # Initialize the visualizer
    # visualizer, skip_generations= Histogram3DVisualizer('fitness_histograms_Navix_Empty_Random_6x6_v0_40gen_3000pop.npz'), 1
    # visualizer, skip_generations = Histogram3DVisualizer('fitness_histograms_cifar100_256h_1000gen_2000pop.npz'), 50
    visualizer, skip_generations = Histogram3DVisualizer('fitness_histograms_cifar10_256h_4000gen_2000pop.npz'), 50

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

if __name__ == "__main__":
    main()
