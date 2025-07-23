import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import seaborn as sns

class Histogram3DVisualizer:
    def __init__(self, npz_file_path):
        """
        Initialize the 3D histogram visualizer
        
        Args:
            npz_file_path (str): Path to the .npz file containing histogram data
        """
        self.data = np.load(npz_file_path)
        self.histogram_tensor = self.data['histogram_tensor']  # (20, 10)
        self.generations = self.data['generations']  # (20,)
        self.bin_edges = self.data['bin_edges']  # (20, 11)
        
        # Calculate bin centers for each generation
        self.bin_centers = []
        for i in range(len(self.generations)):
            centers = (self.bin_edges[i][:-1] + self.bin_edges[i][1:]) / 2
            self.bin_centers.append(centers)
        self.bin_centers = np.array(self.bin_centers)  # (20, 10)
        
        print(f"Loaded data: {len(self.generations)} generations, {self.histogram_tensor.shape[1]} bins per histogram")
        
    def create_static_3d_view(self, style='wireframe', colormap='viridis', save_path=None):
        """
        Create a static 3D visualization of histogram evolution
        
        Args:
            style (str): 'wireframe', 'surface', or 'line'
            colormap (str): matplotlib colormap name
            save_path (str): path to save the figure (optional)
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up color map
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(self.generations)))
        
        if style == 'wireframe':
            # Create wireframe surface
            X, Y = np.meshgrid(range(self.histogram_tensor.shape[1]), self.generations)
            Z = self.histogram_tensor
            
            ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.7, linewidth=1.5)
            
        elif style == 'surface':
            # Create surface plot
            X, Y = np.meshgrid(range(self.histogram_tensor.shape[1]), self.generations)
            Z = self.histogram_tensor
            
            surf = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=0.8, 
                                 linewidth=0.5, edgecolors='black')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Frequency')
            
        elif style == 'line':
            # Create line plots for each generation
            for i, gen in enumerate(self.generations):
                x_vals = range(len(self.histogram_tensor[i]))
                y_vals = [gen] * len(self.histogram_tensor[i])
                z_vals = self.histogram_tensor[i]
                
                ax.plot(x_vals, y_vals, z_vals, color=colors[i], 
                       linewidth=2, alpha=0.8, marker='o', markersize=4)
        
        # Customize the plot
        ax.set_xlabel('Bin Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Generation', fontsize=12, fontweight='bold')
        ax.set_zlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('3D Evolution of Fitness Histograms\n(Navix Empty Random 6x6)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set better viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Customize tick parameters
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved static 3D visualization to {save_path}")
            
        plt.show()
    
    def create_detailed_planes_view(self, save_path=None):
        """
        Create a detailed view with individual histogram planes
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color scheme for generations
        colors = plt.cm.plasma(np.linspace(0, 1, len(self.generations)))
        
        for i, gen in enumerate(self.generations):
            # Create a plane for each generation
            x_vals = np.arange(len(self.histogram_tensor[i]))
            y_val = gen
            z_vals = self.histogram_tensor[i]
            
            # Plot the histogram as a line on a vertical plane
            ax.plot(x_vals, [y_val] * len(x_vals), z_vals, 
                   color=colors[i], linewidth=3, alpha=0.8, 
                   marker='o', markersize=5, label=f'Gen {gen}')
            
            # Add vertical lines from base to histogram values
            for j, (x, z) in enumerate(zip(x_vals, z_vals)):
                ax.plot([x, x], [y_val, y_val], [0, z], 
                       color=colors[i], alpha=0.4, linewidth=1)
        
        # Customize the plot
        ax.set_xlabel('Bin Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Generation', fontsize=12, fontweight='bold')
        ax.set_zlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('3D Histogram Evolution: Individual Generation Planes\n(Navix Empty Random 6x6)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set viewing angle
        ax.view_init(elev=15, azim=60)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved detailed planes view to {save_path}")
            
        plt.show()
    
    def create_fitness_value_based_view(self, save_path=None):
        """
        Create 3D view using actual fitness values instead of bin indices
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(self.generations)))
        
        for i, gen in enumerate(self.generations):
            # Use actual bin centers (fitness values) for x-axis
            x_vals = self.bin_centers[i]
            y_val = gen
            z_vals = self.histogram_tensor[i]
            
            # Plot histogram line
            ax.plot(x_vals, [y_val] * len(x_vals), z_vals, 
                   color=colors[i], linewidth=2.5, alpha=0.9, 
                   marker='o', markersize=4)
            
            # Fill area under curve for better visualization
            ax.plot(x_vals, [y_val] * len(x_vals), [0] * len(x_vals), 
                   color=colors[i], alpha=0.2, linewidth=1)
            
            # Connect histogram to base plane
            for x, z in zip(x_vals, z_vals):
                ax.plot([x, x], [y_val, y_val], [0, z], 
                       color=colors[i], alpha=0.3, linewidth=0.8)
        
        # Customize the plot
        ax.set_xlabel('Fitness Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Generation', fontsize=12, fontweight='bold')
        ax.set_zlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('3D Fitness Distribution Evolution\n(Actual Fitness Values)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved fitness-based view to {save_path}")
            
        plt.show()
    
    def create_glass_sheet_view(self, save_path=None):
        """
        Create 3D view with translucent glass-like sheets for each generation
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(self.generations)))
        
        for i, gen in enumerate(self.generations):
            # Use actual bin centers (fitness values) for x-axis
            x_vals = self.bin_centers[i]
            y_val = gen
            z_vals = self.histogram_tensor[i]
            
            # Create the filled polygon as a translucent sheet
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # Create vertices for the polygon that fills the area under the histogram
            verts = []
            
            # Start from the first point at the bottom
            verts.append([x_vals[0], y_val, 0])
            
            # Add all the histogram points (top edge)
            for j in range(len(x_vals)):
                verts.append([x_vals[j], y_val, z_vals[j]])
            
            # Close back to the last point at the bottom
            verts.append([x_vals[-1], y_val, 0])
            
            # Create the polygon collection with correct format
            collection = Poly3DCollection([verts], alpha=0.4, facecolor=colors[i], 
                                        edgecolor=colors[i], linewidth=1.5)
            ax.add_collection3d(collection)
            
            # Add the histogram outline for clarity
            ax.plot(x_vals, [y_val] * len(x_vals), z_vals, 
                   color=colors[i], linewidth=3.0, alpha=0.9, 
                   marker='o', markersize=3, zorder=10)
        
        # Customize the plot
        ax.set_xlabel('Fitness Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Generation', fontsize=12, fontweight='bold')
        ax.set_zlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('3D Glass Sheet Evolution View\n(Translucent Histogram Sheets)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set viewing angle for best glass sheet effect
        ax.view_init(elev=20, azim=35)
        ax.grid(True, alpha=0.2)
        
        # Set axis limits to better show the glass sheets
        ax.set_xlim(np.min(self.bin_centers), np.max(self.bin_centers))
        ax.set_ylim(np.min(self.generations)-1, np.max(self.generations)+1)
        ax.set_zlim(0, np.max(self.histogram_tensor)*1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved glass sheet view to {save_path}")
            
        plt.show()
    
    def create_animated_view(self, save_path=None):
        """
        Create an animated 3D view showing histogram evolution
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame):
            ax.clear()
            
            # Show evolution up to current frame
            colors = plt.cm.viridis(np.linspace(0, 1, frame + 1))
            
            for i in range(frame + 1):
                gen = self.generations[i]
                x_vals = np.arange(len(self.histogram_tensor[i]))
                y_val = gen
                z_vals = self.histogram_tensor[i]
                
                alpha = 0.3 if i < frame else 1.0  # Highlight current generation
                linewidth = 1.5 if i < frame else 3.0
                
                ax.plot(x_vals, [y_val] * len(x_vals), z_vals, 
                       color=colors[i], linewidth=linewidth, alpha=alpha, 
                       marker='o', markersize=3)
            
            ax.set_xlabel('Bin Index')
            ax.set_ylabel('Generation')
            ax.set_zlabel('Frequency')
            ax.set_title(f'Histogram Evolution - Generation {self.generations[frame]}')
            ax.view_init(elev=20, azim=45)
            ax.grid(True, alpha=0.3)
        
        anim = FuncAnimation(fig, animate, frames=len(self.generations), 
                           interval=800, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1.5)
            print(f"Saved animation to {save_path}")
        
        plt.show()
        return anim
    
    def create_summary_statistics(self):
        """
        Print summary statistics about the histogram evolution
        """
        print("=== Histogram Evolution Summary ===")
        print(f"Total generations: {len(self.generations)}")
        print(f"Bins per histogram: {self.histogram_tensor.shape[1]}")
        print(f"Total population per generation: {np.sum(self.histogram_tensor[0])}")
        
        print("\n=== Evolution Statistics ===")
        # Calculate some evolution metrics
        max_freq_per_gen = np.max(self.histogram_tensor, axis=1)
        dominant_bin_per_gen = np.argmax(self.histogram_tensor, axis=1)
        
        print(f"Maximum frequency range: {np.min(max_freq_per_gen)} - {np.max(max_freq_per_gen)}")
        print(f"Most common dominant bin: {np.bincount(dominant_bin_per_gen).argmax()}")
        
        # Fitness range evolution
        print(f"\n=== Fitness Range Evolution ===")
        print(f"Initial fitness range: {self.data['min_vals'][0]:.1f} to {self.data['max_vals'][0]:.1f}")
        print(f"Final fitness range: {self.data['min_vals'][-1]:.1f} to {self.data['max_vals'][-1]:.1f}")


def main():
    """
    Main function to run the 3D visualization
    """
    # Initialize the visualizer
    visualizer = Histogram3DVisualizer('fitness_histograms_Navix_Empty_Random_6x6_v0_20gen_3000pop.npz')
    
    # Print summary statistics
    visualizer.create_summary_statistics()
    
    print("\n=== Creating 3D Visualizations ===")
    
    # Create different types of 3D visualizations
    print("1. Creating wireframe view...")
    visualizer.create_static_3d_view(style='wireframe', save_path='histogram_3d_wireframe.png')
    
    print("2. Creating surface view...")
    visualizer.create_static_3d_view(style='surface', save_path='histogram_3d_surface.png')
    
    print("3. Creating line plot view...")
    visualizer.create_static_3d_view(style='line', save_path='histogram_3d_lines.png')
    
    print("4. Creating detailed planes view...")
    visualizer.create_detailed_planes_view(save_path='histogram_3d_planes.png')
    
    print("5. Creating fitness-based view...")
    visualizer.create_fitness_value_based_view(save_path='histogram_3d_fitness.png')
    
    print("6. Creating glass sheet view...")
    visualizer.create_glass_sheet_view(save_path='histogram_3d_glass_sheets.png')
    
    print("\n=== All visualizations created successfully! ===")
    print("Generated files:")
    print("- histogram_3d_wireframe.png")
    print("- histogram_3d_surface.png")
    print("- histogram_3d_lines.png")
    print("- histogram_3d_planes.png")
    print("- histogram_3d_fitness.png")
    print("- histogram_3d_glass_sheets.png")


if __name__ == "__main__":
    main()
