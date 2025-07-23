import numpy as np
from histogram_3d_visualization import Histogram3DVisualizer

def main():
    """
    Interactive demo for the 3D histogram visualizer
    """
    print("=== 3D Histogram Visualization Demo ===")
    print("Loading data...")
    
    # Initialize the visualizer
    visualizer = Histogram3DVisualizer('fitness_histograms_Navix_Empty_Random_6x6_v0_20gen_3000pop.npz')
    
    # Show summary
    visualizer.create_summary_statistics()
    
    print("\n=== Available Visualization Options ===")
    print("1. Wireframe view (3D mesh)")
    print("2. Surface view (3D surface with colors)")
    print("3. Line plots view (Individual generation lines)")
    print("4. Detailed planes view (Histograms as vertical planes)")
    print("5. Fitness-based view (Using actual fitness values)")
    print("6. Glass sheet view (Translucent glass-like sheets)")
    print("7. Animated view (Evolution animation)")
    print("8. Generate all static views")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-8): ").strip()
            
            if choice == '0':
                print("Exiting...")
                break
            elif choice == '1':
                print("Creating wireframe view...")
                visualizer.create_static_3d_view(style='wireframe', save_path='demo_wireframe.png')
            elif choice == '2':
                print("Creating surface view...")
                visualizer.create_static_3d_view(style='surface', save_path='demo_surface.png')
            elif choice == '3':
                print("Creating line plots view...")
                visualizer.create_static_3d_view(style='line', save_path='demo_lines.png')
            elif choice == '4':
                print("Creating detailed planes view...")
                visualizer.create_detailed_planes_view(save_path='demo_planes.png')
            elif choice == '5':
                print("Creating fitness-based view...")
                visualizer.create_fitness_value_based_view(save_path='demo_fitness.png')
            elif choice == '6':
                print("Creating glass sheet view...")
                visualizer.create_glass_sheet_view(save_path='demo_glass_sheets.png')
            elif choice == '7':
                print("Creating animated view...")
                print("Note: Animation will show in a window and can be saved as GIF")
                save_anim = input("Save animation as GIF? (y/n): ").strip().lower()
                save_path = 'histogram_evolution_animation.gif' if save_anim == 'y' else None
                anim = visualizer.create_animated_view(save_path=save_path)
            elif choice == '8':
                print("Generating all static views...")
                visualizer.create_static_3d_view(style='wireframe', save_path='all_wireframe.png')
                visualizer.create_static_3d_view(style='surface', save_path='all_surface.png')
                visualizer.create_static_3d_view(style='line', save_path='all_lines.png')
                visualizer.create_detailed_planes_view(save_path='all_planes.png')
                visualizer.create_fitness_value_based_view(save_path='all_fitness.png')
                visualizer.create_glass_sheet_view(save_path='all_glass_sheets.png')
                print("All static views generated!")
            else:
                print("Invalid choice. Please enter a number between 0-8.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
