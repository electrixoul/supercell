import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

class PlotlyHistogram3DVisualizer:
    def __init__(self, npz_file_path):
        """
        Initialize the Plotly 3D histogram visualizer
        
        Args:
            npz_file_path (str): Path to the .npz file containing histogram data
        """
        self.data = np.load(npz_file_path)
        self.histogram_tensor = self.data['histogram_tensor']
        self.generations = self.data['generations']
        self.bin_edges = self.data['bin_edges']
        
        # Calculate bin centers for each generation
        self.bin_centers = []
        for i in range(len(self.generations)):
            centers = (self.bin_edges[i][:-1] + self.bin_edges[i][1:]) / 2
            self.bin_centers.append(centers)
        self.bin_centers = np.array(self.bin_centers)
        
        print(f"Loaded data: {len(self.generations)} generations, {self.histogram_tensor.shape[1]} bins per histogram")
    
    def create_interactive_glass_sheets(self, skip_generations=1, save_html=True, auto_open=True):
        """
        Create interactive 3D glass sheet visualization using Plotly
        
        Args:
            skip_generations (int): draw every nth generation
            save_html (bool): save as HTML file for sharing
            auto_open (bool): automatically open in browser
        """
        # Select generations to draw
        selected_indices = list(range(0, len(self.generations), skip_generations))
        selected_generations = [self.generations[i] for i in selected_indices]
        
        print(f"Creating interactive visualization with {len(selected_indices)} generations")
        print(f"Selected generations: {selected_generations}")
        
        # Create figure
        fig = go.Figure()
        
        # Color scale for generations (matching matplotlib coolwarm)
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Use matplotlib's coolwarm colormap for consistency
        coolwarm_colors = cm.coolwarm(np.linspace(0, 1, len(selected_indices)))
        colors = ['rgba({},{},{},{})'.format(int(r*255), int(g*255), int(b*255), 1.0) 
                 for r, g, b, a in coolwarm_colors]
        
        for idx, i in enumerate(selected_indices):
            gen = self.generations[i]
            x_vals = self.bin_centers[i]
            z_vals = self.histogram_tensor[i]
            
            # MAXIMUM resolution for ultra-smooth rendering
            n_interp = 500  # Extreme high resolution - 5x more than before
            x_smooth = np.linspace(x_vals[0], x_vals[-1], n_interp)
            z_smooth = np.interp(x_smooth, x_vals, z_vals)
            
            # Create surface coordinates with perfect alignment
            X_surface = np.array([x_smooth, x_smooth])
            Y_surface = np.array([[gen, gen] for _ in range(n_interp)]).T
            Z_surface = np.array([np.zeros_like(z_smooth), z_smooth])
            
            # Add the glass sheet as an ultra-smooth surface
            fig.add_trace(go.Surface(
                x=X_surface,
                y=Y_surface, 
                z=Z_surface,
                colorscale=[[0, colors[idx]], [1, colors[idx]]],
                opacity=0.65,  # Slightly more transparent for better glass effect
                showscale=False,
                name=f'Generation {gen}',
                showlegend=True,
                hovertemplate='Generation: %{y}<br>Fitness: %{x:.2f}<br>Frequency: %{z}<extra></extra>',
                lighting=dict(
                    ambient=0.5,   # Higher ambient light
                    diffuse=0.6,   # Balanced diffuse
                    specular=0.4,  # More specular for glass
                    roughness=0.02, # Ultra-smooth surface
                    fresnel=0.4    # Strong glass reflection
                ),
                lightposition=dict(x=100, y=200, z=0),
                # Better edge definition with contours disabled
                contours=dict(
                    x=dict(show=False),
                    y=dict(show=False), 
                    z=dict(show=False)
                )
            ))
            
            # Use the SAME interpolated points for the outline - ZERO gap
            fig.add_trace(go.Scatter3d(
                x=x_smooth,  # Use same interpolated x points
                y=[gen] * len(x_smooth),  # Same y coordinate
                z=z_smooth,  # Same interpolated z points
                mode='lines+markers',
                line=dict(
                    color=colors[idx], 
                    width=6  # Slightly thicker for better visibility
                ),
                marker=dict(
                    size=1.0,  # Even smaller markers
                    color=colors[idx],
                    # Force marker anti-aliasing
                    line=dict(width=0)
                ),
                name=f'Outline Gen {gen}',
                showlegend=False,
                hovertemplate='Generation: %{y}<br>Fitness: %{x:.2f}<br>Frequency: %{z}<extra></extra>'
            ))
        
        # Update layout for better 3D visualization
        fig.update_layout(
            title={
                'text': f'Interactive 3D Glass Sheet Histogram Evolution<br><sub>Skip every {skip_generations} generation(s) - {len(selected_indices)} sheets total</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            scene=dict(
                xaxis_title='Fitness Value',
                yaxis_title='Generation',
                zaxis_title='Frequency',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),  # Optimal viewing angle
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='cube',  # Better proportions
                bgcolor='rgba(255, 255, 255, 0)',  # Transparent background - no blue walls
                xaxis=dict(
                    backgroundcolor='rgba(255, 255, 255, 0)',  # Remove axis background
                    gridcolor='rgba(150, 150, 150, 0.3)',      # Light grid lines
                    showbackground=False                        # Remove background plane
                ),
                yaxis=dict(
                    backgroundcolor='rgba(255, 255, 255, 0)',
                    gridcolor='rgba(150, 150, 150, 0.3)',
                    showbackground=False
                ),
                zaxis=dict(
                    backgroundcolor='rgba(255, 255, 255, 0)',
                    gridcolor='rgba(150, 150, 150, 0.3)',
                    showbackground=False
                )
            ),
            width=1400,
            height=900,
            font=dict(size=12),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            )
        )
        
        # Configure high-quality rendering and anti-aliasing
        fig.update_layout(
            # Enhanced rendering settings for maximum quality
            autosize=True,
            # High DPI display support
            template='plotly_white'
        )
        
        # Update scene with high-quality settings
        fig.update_scenes(
            # Enhanced camera projection for anti-aliasing
            camera_projection_type='perspective',
            # Better axis rendering
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False
        )
        
        # Save as HTML with high-quality settings
        if save_html:
            html_filename = 'interactive_glass_sheets.html'
            # High-quality HTML export with anti-aliasing
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['pan3d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'glass_sheet_visualization',
                    'height': 1200,
                    'width': 1600,
                    'scale': 2  # High DPI for crisp export
                },
                # WebGL anti-aliasing
                'plotGlPixelRatio': 2,  # Higher pixel ratio for better quality
                'editable': False,
                'scrollZoom': True
            }
            
            # Create HTML with custom full-screen CSS
            html_content = fig.to_html(
                include_plotlyjs='cdn',
                config=config,
                div_id="glass-sheet-plot"
            )
            
            # Inject minimal CSS for better appearance
            minimal_css = '''
<style>
/* Clean layout with breathing room */
body {
    margin: 20px;
    background-color: #fafafa;
    font-family: "Open Sans", verdana, arial, sans-serif;
}

/* Make the plotly div look professional */
#glass-sheet-plot {
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    background: white;
    margin: 0 auto;
}

/* Enhance modebar visibility */
.modebar {
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

/* Better legend styling */
.legend {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}
</style>
'''
            
            # Insert minimal CSS before closing head tag
            html_content = html_content.replace('</head>', minimal_css + '</head>')
            
            # Write enhanced HTML
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            print(f"✅ Saved FULL-SCREEN interactive visualization as: {html_filename}")
        
        # Show the plot with high-quality configuration
        if auto_open:
            # Configure for maximum quality display
            config = {
                'displayModeBar': True,
                'plotGlPixelRatio': 2,  # High DPI rendering
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'glass_sheet_visualization',
                    'height': 1200,
                    'width': 1600,
                    'scale': 2
                }
            }
            fig.show(config=config)
            print("🚀 High-quality interactive visualization opened in browser!")
        
        return fig
    
    def create_simple_surface_view(self, skip_generations=1):
        """
        Create a simpler surface plot for better performance
        """
        selected_indices = list(range(0, len(self.generations), skip_generations))
        
        fig = go.Figure()
        
        # Create surface plot
        X = []
        Y = []
        Z = []
        
        for i in selected_indices:
            gen = self.generations[i]
            x_vals = self.bin_centers[i]
            z_vals = self.histogram_tensor[i]
            
            X.append(x_vals)
            Y.append([gen] * len(x_vals))
            Z.append(z_vals)
        
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='RdYlBu_r',
            opacity=0.8,
            showscale=True,
            hovertemplate='Generation: %{y}<br>Fitness: %{x:.2f}<br>Frequency: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='3D Surface Evolution View',
            scene=dict(
                xaxis_title='Fitness Value',
                yaxis_title='Generation',
                zaxis_title='Frequency',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=1000,
            height=700
        )
        
        fig.show()
        return fig
    
    def create_animated_evolution(self, skip_generations=1):
        """
        Create an animated version showing evolution over time
        """
        selected_indices = list(range(0, len(self.generations), skip_generations))
        
        frames = []
        for frame_idx, i in enumerate(selected_indices):
            gen = self.generations[i]
            
            # Show all generations up to current frame
            frame_data = []
            colors = px.colors.sample_colorscale("RdYlBu_r", frame_idx + 1)
            
            for j in range(frame_idx + 1):
                idx = selected_indices[j]
                gen_j = self.generations[idx]
                x_vals = self.bin_centers[idx]
                y_vals = [gen_j] * len(x_vals)
                z_vals = self.histogram_tensor[idx]
                
                alpha = 0.3 if j < frame_idx else 0.7
                
                frame_data.append(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='lines+markers',
                    line=dict(color=colors[j], width=4),
                    marker=dict(size=3, color=colors[j]),
                    opacity=alpha,
                    name=f'Gen {gen_j}'
                ))
            
            frames.append(go.Frame(data=frame_data, name=str(gen)))
        
        # Initial frame
        fig = go.Figure(data=frames[0].data)
        
        fig.frames = frames
        
        fig.update_layout(
            title='Animated Histogram Evolution',
            scene=dict(
                xaxis_title='Fitness Value',
                yaxis_title='Generation',
                zaxis_title='Frequency'
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}}])
                ]
            )]
        )
        
        fig.show()
        return fig


def main():
    """
    Main function for interactive glass sheet visualization
    """
    try:
        print("=== Interactive 3D Glass Sheet Visualization (Plotly) ===")
        print("Loading data...")
        
        # Initialize visualizer
        visualizer = PlotlyHistogram3DVisualizer('fitness_histograms_Navix_Empty_Random_6x6_v0_20gen_3000pop.npz')
        
        # Configuration
        skip_generations = 2  # 📝 Modify this to change skip pattern
        
        print(f"\n=== Creating Interactive Glass Sheet View ===")
        print("⚡ Using Plotly for smooth 3D interaction on macOS")
        
        # Create interactive visualization
        fig = visualizer.create_interactive_glass_sheets(
            skip_generations=skip_generations,
            save_html=True,
            auto_open=True
        )
        
        print("\n✨ Interactive features available:")
        print("🖱️  Mouse: Rotate, zoom, pan")
        print("🔍 Hover: View detailed data points")
        print("👁️  Legend: Click to show/hide generations")
        print("📱 Responsive: Works on all devices")
        print("🌐 Shareable: HTML file can be shared with others")
        
        # Clean shutdown to avoid command line errors
        print("\n🎯 Program completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Ensure clean exit
        import sys
        sys.exit(0)


if __name__ == "__main__":
    main()
