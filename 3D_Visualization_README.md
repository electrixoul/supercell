# 3D Histogram Evolution Visualization

This project provides comprehensive 3D visualization tools for analyzing the evolution of fitness histograms over multiple generations in genetic algorithms.

## Overview

The fitness histogram data contains:
- **20 generations** of evolution
- **10 bins** per histogram
- **3000 individuals** per generation
- Fitness range: -500.0 to -1.0

## Generated Visualizations

### 1. Wireframe View (`histogram_3d_wireframe.png`)
- 3D wireframe mesh showing the histogram surface
- Clear structural view of the evolution landscape
- Best for understanding overall shape and topology

### 2. Surface View (`histogram_3d_surface.png`)
- Colored 3D surface with gradient coloring
- Includes colorbar for frequency interpretation
- Best for visualizing frequency magnitudes with colors

### 3. Line Plots View (`histogram_3d_lines.png`)
- Individual line plots for each generation
- Each generation shown as a colored line in 3D space
- Best for seeing individual generation patterns

### 4. Detailed Planes View (`histogram_3d_planes.png`)
- Histograms displayed as vertical planes
- Each generation on a separate vertical plane
- Includes connecting lines from base to histogram values
- Best for detailed analysis of individual generations

### 5. Fitness-Based View (`histogram_3d_fitness.png`)
- Uses actual fitness values instead of bin indices
- X-axis shows real fitness values (-500 to -1)
- Most scientifically accurate representation
- Best for understanding fitness distribution evolution

## Usage

### Main Visualization Script
```bash
python histogram_3d_visualization.py
```
This generates all five visualization types automatically.

### Interactive Demo
```bash
python demo_3d_visualization.py
```
This provides an interactive menu to:
- Generate individual visualizations
- Create animations
- Save specific formats
- Explore different viewing options

## Key Features

### Data Analysis
- **Population Size**: 3000 individuals per generation
- **Evolution Tracking**: 20 generations of genetic algorithm evolution
- **Fitness Distribution**: 10-bin histograms showing fitness value distribution
- **Dynamic Binning**: Adaptive bin edges for each generation

### Visualization Types
1. **Static 3D Views**: Multiple perspectives of the evolution surface
2. **Animated Views**: Time-based evolution animation
3. **Scientific Accuracy**: Real fitness values vs. simplified bin indices
4. **Multiple Colormaps**: Various color schemes for different insights

### Technical Details
- **3D Rendering**: Uses matplotlib's 3D plotting capabilities
- **High Resolution**: 300 DPI output for publication quality
- **Multiple Formats**: PNG images and GIF animations
- **Customizable**: Easy to modify colors, viewing angles, and styles

## Files Structure

```
├── histogram_3d_visualization.py     # Main visualization class
├── demo_3d_visualization.py         # Interactive demo script
├── 3D_Visualization_README.md       # This documentation
├── fitness_histograms_Navix_Empty_Random_6x6_v0_20gen_3000pop.npz  # Data file
└── Generated visualizations:
    ├── histogram_3d_wireframe.png
    ├── histogram_3d_surface.png
    ├── histogram_3d_lines.png
    ├── histogram_3d_planes.png
    └── histogram_3d_fitness.png
```

## Understanding the Evolution

### Key Observations
- **Dominant Bin**: Bin 0 (lowest fitness values) consistently has highest frequency
- **Population Distribution**: Initial frequency of 1401 in the lowest bin
- **Evolution Trend**: Shows how genetic algorithm population shifts over time
- **Fitness Landscape**: Reveals the fitness distribution characteristics

### Biological Interpretation
- **Selection Pressure**: High frequency in low-fitness bins suggests strong selection
- **Population Dynamics**: Changes in histogram shape reveal evolutionary pressures
- **Genetic Drift**: Random changes in small frequency bins
- **Convergence**: Later generations show different distribution patterns

## Customization Options

### Color Schemes
- `viridis` (default): Blue-green-yellow gradient
- `plasma`: Purple-pink-yellow gradient
- `coolwarm`: Blue-white-red gradient
- `inferno`: Black-red-yellow gradient

### Viewing Angles
- **Elevation**: Vertical viewing angle (0-90 degrees)
- **Azimuth**: Horizontal rotation angle (0-360 degrees)
- **Distance**: Zoom level for detail inspection

### Export Options
- **High Resolution**: 300 DPI for publications
- **Multiple Formats**: PNG, PDF, SVG support
- **Animation**: GIF format for presentations
- **Interactive**: Matplotlib viewer for exploration

## Requirements

```python
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Installation

```bash
# Activate the AI environment
source ~/.bash_profile
conda activate ai_env

# Run the visualization
python histogram_3d_visualization.py
```

## Examples

### Quick Start
```python
from histogram_3d_visualization import Histogram3DVisualizer

# Load and visualize
viz = Histogram3DVisualizer('fitness_histograms_Navix_Empty_Random_6x6_v0_20gen_3000pop.npz')
viz.create_static_3d_view(style='surface', save_path='my_surface.png')
```

### Custom Animation
```python
# Create custom animation
anim = viz.create_animated_view(save_path='evolution.gif')
```

## Scientific Applications

This visualization tool is particularly useful for:
- **Genetic Algorithm Analysis**: Understanding population evolution
- **Fitness Landscape Studies**: Visualizing selection pressures
- **Algorithm Comparison**: Comparing different GA strategies
- **Parameter Optimization**: Analyzing the effect of GA parameters
- **Educational Purposes**: Teaching evolutionary computation concepts

## Troubleshooting

### Common Issues
1. **Memory Error**: Reduce figure size or use fewer generations
2. **Display Issues**: Ensure X11 forwarding for remote sessions
3. **Animation Problems**: Install Pillow for GIF support: `pip install Pillow`

### Performance Tips
- Use `style='wireframe'` for faster rendering
- Reduce figure size for quicker generation
- Close figures after saving to free memory

## Future Enhancements

Potential improvements include:
- Interactive 3D web visualization using Plotly
- Real-time evolution visualization
- Comparative analysis tools for multiple experiments
- Statistical analysis integration
- Machine learning pattern recognition

---

Created for genetic algorithm fitness evolution analysis. This tool provides deep insights into how population fitness distributions evolve over generations in evolutionary computation experiments.
