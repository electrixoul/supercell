import numpy as np
from histogram_3d_visualization import Histogram3DVisualizer

def main():
    """
    Automatic Glass Sheet 3D visualization - no user interaction required
    """
    print("=== 3D Glass Sheet Histogram Visualization ===")
    print("Loading data...")
    
    # Initialize the visualizer
    visualizer = Histogram3DVisualizer('fitness_histograms_cifar10_256h_2000gen_2000pop.npz')
    
    # Show summary
    visualizer.create_summary_statistics()
    
    print("\n=== Creating Glass Sheet View ===")
    
    # Skip generations parameter - change this value to control how many generations to skip
    # 1 = draw all generations, 2 = draw every other generation, 3 = draw every 3rd generation, etc.
    skip_generations = 50  # 📝 Modify this value to change skip pattern
    
    print(f"Generating 3D glass sheet visualization (every {skip_generations} generation(s))...")
    
    # Automatically create and save the glass sheet view with skip parameter
    visualizer.create_glass_sheet_view(save_path='glass_sheet_visualization.png', skip_generations=skip_generations)
    
    print("✅ Glass sheet visualization completed!")
    print("📁 Saved as: glass_sheet_visualization.png")
    print("🎯 The 3D view has been displayed and saved automatically.")
    print(f"📊 Used skip_generations={skip_generations} to reduce visual complexity.")

if __name__ == "__main__":
    main()
