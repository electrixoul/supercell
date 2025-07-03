# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Set matplotlib font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class ExperimentDataLoader:
    """
    Load and reconstruct experiment data from NPZ file
    """
    def __init__(self, npz_file_path):
        """
        Initialize with NPZ file path
        """
        self.npz_file_path = npz_file_path
        self.data = None
        self.exp_with = None
        self.exp_without = None
        
    def load_data(self):
        """
        Load data from NPZ file
        """
        if not os.path.exists(self.npz_file_path):
            raise FileNotFoundError(f"NPZ file not found: {self.npz_file_path}")
        
        print(f"Loading data from: {self.npz_file_path}")
        self.data = np.load(self.npz_file_path, allow_pickle=True)
        print(f"Successfully loaded {len(self.data.files)} data arrays")
        
        # Print available data keys for verification
        print("\nAvailable data keys:")
        for key in sorted(self.data.files):
            if key != 'experiment_parameters':
                array_data = self.data[key]
                print(f"  {key}: shape {array_data.shape}, dtype {array_data.dtype}")
            else:
                print(f"  {key}: {dict(self.data[key].item())}")
        
    def create_mock_experiment_objects(self):
        """
        Create mock experiment objects to mimic the original structure
        """
        class MockExperiment:
            def __init__(self):
                pass
        
        # Create mock objects
        self.exp_with = MockExperiment()
        self.exp_without = MockExperiment()
        
        # Assign data from NPZ file
        self.exp_with.P_values = self.data['P_values_with_crossover']
        self.exp_with.max_fitness_values = self.data['max_fitness_with_crossover']
        self.exp_with.offspring_retained_count = self.data['offspring_retained_with_crossover']
        self.exp_with.offspring_generated_count = self.data['offspring_generated_with_crossover']
        self.exp_with.mutation_count = self.data['mutation_count_with_crossover']
        self.exp_with.eliminated_count = self.data['eliminated_count_with_crossover']
        
        self.exp_without.P_values = self.data['P_values_without_crossover']
        self.exp_without.max_fitness_values = self.data['max_fitness_without_crossover']
        self.exp_without.offspring_retained_count = self.data['offspring_retained_without_crossover']
        self.exp_without.offspring_generated_count = self.data['offspring_generated_without_crossover']
        self.exp_without.mutation_count = self.data['mutation_count_without_crossover']
        self.exp_without.eliminated_count = self.data['eliminated_count_without_crossover']
        
        # Get experiment parameters
        params = self.data['experiment_parameters'].item()
        for key, value in params.items():
            setattr(self.exp_with, key, value)
            setattr(self.exp_without, key, value)
        
        print(f"\nReconstructed experiment objects:")
        print(f"  WITH crossover: {len(self.exp_with.P_values)} generations")
        print(f"  WITHOUT crossover: {len(self.exp_without.P_values)} generations")
        print(f"  Parameters: {params}")


def plot_comparison_results_from_data(exp_with, exp_without, data_loader):
    """
    Plot comparison results from loaded data (identical to original function)
    """
    plt.figure(figsize=(15, 20))
    
    generations_list = list(range(1, len(exp_with.P_values) + 1))
    max_fitness_generations = list(range(0, len(exp_with.max_fitness_values)))
    
    # Plot P value comparison
    plt.subplot(4, 1, 1)
    plt.plot(generations_list, exp_with.P_values, linewidth=2, color='red', 
             marker='o', markersize=3, label='WITH Crossover', alpha=0.8)
    plt.plot(generations_list, exp_without.P_values, linewidth=2, color='blue', 
             marker='s', markersize=3, label='WITHOUT Crossover (Mutation-Only)', alpha=0.8)
    
    plt.title('P Value Comparison: WITH vs WITHOUT Crossover', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('P Value (Proportion of Better Offspring)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, max(max(exp_with.P_values), max(exp_without.P_values)) * 1.1)
    
    # Add average lines
    mean_P_with = np.mean(exp_with.P_values)
    mean_P_without = np.mean(exp_without.P_values)
    plt.axhline(y=mean_P_with, color='red', linestyle='--', alpha=0.5, 
               label=f'WITH Avg = {mean_P_with:.4f}')
    plt.axhline(y=mean_P_without, color='blue', linestyle='--', alpha=0.5, 
               label=f'WITHOUT Avg = {mean_P_without:.4f}')
    plt.legend(fontsize=10)
    
    # Plot maximum fitness comparison
    plt.subplot(4, 1, 2)
    plt.plot(max_fitness_generations, exp_with.max_fitness_values, linewidth=2, color='red', 
             marker='o', markersize=3, label='WITH Crossover', alpha=0.8)
    plt.plot(max_fitness_generations, exp_without.max_fitness_values, linewidth=2, color='blue', 
             marker='s', markersize=3, label='WITHOUT Crossover (Mutation-Only)', alpha=0.8)
    
    plt.title('Maximum Fitness Evolution Comparison: WITH vs WITHOUT Crossover', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Maximum Fitness Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add fitness improvement information
    with_improvement = exp_with.max_fitness_values[-1] - exp_with.max_fitness_values[0]
    without_improvement = exp_without.max_fitness_values[-1] - exp_without.max_fitness_values[0]
    
    plt.text(0.02, 0.98, f'WITH Crossover Improvement: {with_improvement}', 
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
    plt.text(0.02, 0.90, f'WITHOUT Crossover Improvement: {without_improvement}', 
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2))
    
    # Plot retention rate comparison
    plt.subplot(4, 1, 3)
    retention_rates_with = [retained/generated*100 if generated > 0 else 0 
                           for retained, generated in zip(exp_with.offspring_retained_count, exp_with.offspring_generated_count)]
    retention_rates_without = [retained/generated*100 if generated > 0 else 0 
                              for retained, generated in zip(exp_without.offspring_retained_count, exp_without.offspring_generated_count)]
    
    plt.plot(generations_list, retention_rates_with, linewidth=2, color='red', 
             marker='o', markersize=3, label='WITH Crossover', alpha=0.8)
    plt.plot(generations_list, retention_rates_without, linewidth=2, color='blue', 
             marker='s', markersize=3, label='WITHOUT Crossover (Mutation-Only)', alpha=0.8)
    
    plt.title('Offspring Retention Rate Comparison: WITH vs WITHOUT Crossover', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Retention Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    
    # Plot moving average comparison
    plt.subplot(4, 1, 4)
    window_size = 10
    if len(exp_with.P_values) >= window_size:
        moving_avg_with = np.convolve(exp_with.P_values, np.ones(window_size)/window_size, mode='valid')
        moving_avg_without = np.convolve(exp_without.P_values, np.ones(window_size)/window_size, mode='valid')
        moving_avg_generations = generations_list[window_size-1:]
        
        plt.plot(moving_avg_generations, moving_avg_with, linewidth=3, color='red', 
                label=f'WITH Crossover ({window_size}-Gen Moving Avg)')
        plt.plot(moving_avg_generations, moving_avg_without, linewidth=3, color='blue', 
                label=f'WITHOUT Crossover ({window_size}-Gen Moving Avg)')
    
    # Background lines
    plt.plot(generations_list, exp_with.P_values, alpha=0.3, color='red')
    plt.plot(generations_list, exp_without.P_values, alpha=0.3, color='blue')
    
    plt.title('P Value Moving Average Trend Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('P Value (Moving Average)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, max(max(exp_with.P_values), max(exp_without.P_values)) * 1.1)
    
    plt.tight_layout()
    
    # Save the figure as vector format (PDF and SVG)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"genetic_algorithm_comparison_from_data_{timestamp}"
    
    # Save as PDF (vector format)
    plt.savefig(f"{filename_base}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot as: {filename_base}.pdf")
    
    
    # Save as high-resolution PNG for backup
    plt.savefig(f"{filename_base}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot as: {filename_base}.png")
    
    plt.show()
    
    # Print comprehensive comparison statistics
    print(f"\n" + "="*80)
    print("COMPREHENSIVE COMPARISON RESULTS (FROM LOADED DATA)")
    print("="*80)
    
    print(f"\n=== P VALUE STATISTICS ===")
    print(f"WITH Crossover    - Average: {np.mean(exp_with.P_values):.4f}, "
          f"Std: {np.std(exp_with.P_values):.4f}, "
          f"Range: [{np.min(exp_with.P_values):.4f}, {np.max(exp_with.P_values):.4f}]")
    print(f"WITHOUT Crossover - Average: {np.mean(exp_without.P_values):.4f}, "
          f"Std: {np.std(exp_without.P_values):.4f}, "
          f"Range: [{np.min(exp_without.P_values):.4f}, {np.max(exp_without.P_values):.4f}]")
    
    p_ratio = np.mean(exp_with.P_values) / np.mean(exp_without.P_values) if np.mean(exp_without.P_values) > 0 else float('inf')
    print(f"P Value Efficiency Ratio (WITH/WITHOUT): {p_ratio:.1f}x")
    
    print(f"\n=== FITNESS IMPROVEMENT STATISTICS ===")
    with_improvement = exp_with.max_fitness_values[-1] - exp_with.max_fitness_values[0]
    without_improvement = exp_without.max_fitness_values[-1] - exp_without.max_fitness_values[0]
    
    print(f"WITH Crossover    - Initial: {exp_with.max_fitness_values[0]}, "
          f"Final: {exp_with.max_fitness_values[-1]}, Improvement: {with_improvement}")
    print(f"WITHOUT Crossover - Initial: {exp_without.max_fitness_values[0]}, "
          f"Final: {exp_without.max_fitness_values[-1]}, Improvement: {without_improvement}")
    
    fitness_ratio = with_improvement / without_improvement if without_improvement > 0 else float('inf')
    print(f"Fitness Improvement Ratio (WITH/WITHOUT): {fitness_ratio:.1f}x")
    
    print(f"\n=== GENERATION STATISTICS ===")
    zero_p_with = np.sum(np.array(exp_with.P_values) == 0)
    zero_p_without = np.sum(np.array(exp_without.P_values) == 0)
    print(f"Generations with P=0 - WITH: {zero_p_with}/{len(exp_with.P_values)}, "
          f"WITHOUT: {zero_p_without}/{len(exp_without.P_values)}")
    
    print(f"\n=== RETENTION STATISTICS ===")
    total_retained_with = sum(exp_with.offspring_retained_count)
    total_generated_with = sum(exp_with.offspring_generated_count)
    retention_rate_with = total_retained_with / total_generated_with * 100
    
    total_retained_without = sum(exp_without.offspring_retained_count)
    total_generated_without = sum(exp_without.offspring_generated_count)
    retention_rate_without = total_retained_without / total_generated_without * 100
    
    print(f"WITH Crossover    - Retention Rate: {retention_rate_with:.2f}% ({total_retained_with}/{total_generated_with})")
    print(f"WITHOUT Crossover - Retention Rate: {retention_rate_without:.2f}% ({total_retained_without}/{total_generated_without})")
    
    print(f"\n=== DATA SOURCE INFORMATION ===")
    print(f"Original NPZ file: {data_loader.npz_file_path}")
    print(f"Data arrays loaded: {len(data_loader.data.files)}")
    print(f"Replicated from saved experiment data successfully!")


def linear_function(x, a, b):
    """Linear function: y = ax + b"""
    return a * x + b

def quadratic_function(x, a, b, c):
    """Quadratic function: y = ax² + bx + c"""
    return a * x**2 + b * x + c

def hyperbolic_function(x, a, b, c):
    """Hyperbolic function: y = a/x + b*x + c"""
    return a / x + b * x + c

def logarithmic_function(x, a, b, c):
    """Logarithmic function: y = a*ln(x) + b*x + c"""
    return a * np.log(x) + b * x + c

def square_root_function(x, a, b, c):
    """Square root function: y = a*sqrt(x) + b*x + c"""
    return a * np.sqrt(x) + b * x + c

def analyze_fitness_curves(exp_with, exp_without, start_generation=1000):
    """
    Analyze Maximum Fitness curves from specified generation onwards
    """
    print(f"\n" + "="*80)
    print(f"CURVE FITTING ANALYSIS: MAXIMUM FITNESS FROM GENERATION {start_generation}")
    print("="*80)
    
    # Prepare data from generation 1000 onwards
    # Note: max_fitness_values has one more element than generations (includes initial)
    max_fitness_generations = np.array(range(len(exp_with.max_fitness_values)))
    
    # Filter data from start_generation onwards
    mask = max_fitness_generations >= start_generation
    x_data = max_fitness_generations[mask]
    y_with = exp_with.max_fitness_values[mask]
    y_without = exp_without.max_fitness_values[mask]
    
    print(f"Analyzing data from generation {start_generation} to {max(x_data)}")
    print(f"Data points: {len(x_data)}")
    print(f"WITH crossover: Start={y_with[0]}, End={y_with[-1]}, Change={y_with[-1]-y_with[0]}")
    print(f"WITHOUT crossover: Start={y_without[0]}, End={y_without[-1]}, Change={y_without[-1]-y_without[0]}")
    
    # Define fitting functions and their names
    functions = [
        (linear_function, "Linear: y = ax + b", 2),
        (quadratic_function, "Quadratic: y = ax² + bx + c", 3),
        (hyperbolic_function, "Hyperbolic: y = a/x + bx + c", 3),
        (logarithmic_function, "Logarithmic: y = a*ln(x) + bx + c", 3),
        (square_root_function, "Square Root: y = a*√x + bx + c", 3)
    ]
    
    # Analyze WITH crossover
    print(f"\n" + "="*50)
    print("WITH CROSSOVER - CURVE FITTING ANALYSIS")
    print("="*50)
    
    best_r2_with = -np.inf
    best_fit_with = None
    best_params_with = None
    best_name_with = None
    
    with_results = []
    
    for func, name, param_count in functions:
        try:
            # Fit the curve
            popt, pcov = curve_fit(func, x_data, y_with, maxfev=10000)
            
            # Calculate predictions and R²
            y_pred = func(x_data, *popt)
            r2 = r2_score(y_with, y_pred)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_with - y_pred)**2))
            
            # Store results
            with_results.append((name, popt, r2, rmse, func))
            
            print(f"{name}")
            print(f"  Parameters: {popt}")
            print(f"  R² Score: {r2:.6f}")
            print(f"  RMSE: {rmse:.2f}")
            
            # Track best fit
            if r2 > best_r2_with:
                best_r2_with = r2
                best_fit_with = func
                best_params_with = popt
                best_name_with = name
            
            print()
            
        except Exception as e:
            print(f"{name}: Fitting failed - {str(e)}")
            print()
    
    # Analyze WITHOUT crossover
    print(f"\n" + "="*50)
    print("WITHOUT CROSSOVER - CURVE FITTING ANALYSIS")
    print("="*50)
    
    best_r2_without = -np.inf
    best_fit_without = None
    best_params_without = None
    best_name_without = None
    
    without_results = []
    
    for func, name, param_count in functions:
        try:
            # Fit the curve
            popt, pcov = curve_fit(func, x_data, y_without, maxfev=10000)
            
            # Calculate predictions and R²
            y_pred = func(x_data, *popt)
            r2 = r2_score(y_without, y_pred)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_without - y_pred)**2))
            
            # Store results
            without_results.append((name, popt, r2, rmse, func))
            
            print(f"{name}")
            print(f"  Parameters: {popt}")
            print(f"  R² Score: {r2:.6f}")
            print(f"  RMSE: {rmse:.2f}")
            
            # Track best fit
            if r2 > best_r2_without:
                best_r2_without = r2
                best_fit_without = func
                best_params_without = popt
                best_name_without = name
            
            print()
            
        except Exception as e:
            print(f"{name}: Fitting failed - {str(e)}")
            print()
    
    # Print best fits summary
    print(f"\n" + "="*80)
    print("BEST CURVE FITS SUMMARY")
    print("="*80)
    
    print(f"\n🏆 WITH CROSSOVER - BEST FIT:")
    print(f"  Model: {best_name_with}")
    print(f"  R² Score: {best_r2_with:.6f}")
    print(f"  Parameters: {best_params_with}")
    
    # Generate equation string for WITH crossover
    if "Linear" in best_name_with:
        eq_with = f"y = {best_params_with[0]:.4f}x + {best_params_with[1]:.2f}"
    elif "Quadratic" in best_name_with:
        eq_with = f"y = {best_params_with[0]:.8f}x² + {best_params_with[1]:.4f}x + {best_params_with[2]:.2f}"
    elif "Hyperbolic" in best_name_with:
        eq_with = f"y = {best_params_with[0]:.2f}/x + {best_params_with[1]:.4f}x + {best_params_with[2]:.2f}"
    elif "Logarithmic" in best_name_with:
        eq_with = f"y = {best_params_with[0]:.2f}*ln(x) + {best_params_with[1]:.4f}x + {best_params_with[2]:.2f}"
    elif "Square Root" in best_name_with:
        eq_with = f"y = {best_params_with[0]:.2f}*√x + {best_params_with[1]:.4f}x + {best_params_with[2]:.2f}"
    
    print(f"  📐 Equation: {eq_with}")
    
    print(f"\n🏆 WITHOUT CROSSOVER - BEST FIT:")
    print(f"  Model: {best_name_without}")
    print(f"  R² Score: {best_r2_without:.6f}")
    print(f"  Parameters: {best_params_without}")
    
    # Generate equation string for WITHOUT crossover
    if "Linear" in best_name_without:
        eq_without = f"y = {best_params_without[0]:.4f}x + {best_params_without[1]:.2f}"
    elif "Quadratic" in best_name_without:
        eq_without = f"y = {best_params_without[0]:.8f}x² + {best_params_without[1]:.4f}x + {best_params_without[2]:.2f}"
    elif "Hyperbolic" in best_name_without:
        eq_without = f"y = {best_params_without[0]:.2f}/x + {best_params_without[1]:.4f}x + {best_params_without[2]:.2f}"
    elif "Logarithmic" in best_name_without:
        eq_without = f"y = {best_params_without[0]:.2f}*ln(x) + {best_params_without[1]:.4f}x + {best_params_without[2]:.2f}"
    elif "Square Root" in best_name_without:
        eq_without = f"y = {best_params_without[0]:.2f}*√x + {best_params_without[1]:.4f}x + {best_params_without[2]:.2f}"
    
    print(f"  📐 Equation: {eq_without}")
    
    # Comparison
    print(f"\n📊 CURVE FITTING COMPARISON:")
    print(f"  WITH crossover R²:    {best_r2_with:.6f}")
    print(f"  WITHOUT crossover R²: {best_r2_without:.6f}")
    print(f"  Fitting Quality Ratio: {best_r2_with/best_r2_without:.2f}x")
    
    # Create detailed fitting plot
    create_curve_fitting_plot(x_data, y_with, y_without, with_results, without_results, 
                            best_fit_with, best_params_with, best_name_with,
                            best_fit_without, best_params_without, best_name_without,
                            start_generation)
    
    return {
        'with_results': with_results,
        'without_results': without_results,
        'best_with': (best_name_with, best_params_with, best_r2_with, eq_with),
        'best_without': (best_name_without, best_params_without, best_r2_without, eq_without)
    }

def create_curve_fitting_plot(x_data, y_with, y_without, with_results, without_results,
                            best_fit_with, best_params_with, best_name_with,
                            best_fit_without, best_params_without, best_name_without,
                            start_generation):
    """
    Create detailed curve fitting visualization
    """
    plt.figure(figsize=(20, 12))
    
    # Create extended x range for smooth curves
    x_extended = np.linspace(x_data[0], x_data[-1], 1000)
    
    # Plot 1: WITH Crossover - All Fits
    plt.subplot(2, 2, 1)
    plt.scatter(x_data, y_with, alpha=0.6, color='red', s=10, label='Data Points')
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for i, (name, params, r2, rmse, func) in enumerate(with_results):
        try:
            y_fit = func(x_extended, *params)
            plt.plot(x_extended, y_fit, color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8, label=f'{name.split(":")[0]} (R²={r2:.4f})')
        except:
            pass
    
    plt.title(f'WITH Crossover - All Curve Fits (from Gen {start_generation})', fontsize=14, fontweight='bold')
    plt.xlabel('Generation')
    plt.ylabel('Maximum Fitness')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: WITHOUT Crossover - All Fits
    plt.subplot(2, 2, 2)
    plt.scatter(x_data, y_without, alpha=0.6, color='blue', s=10, label='Data Points')
    
    for i, (name, params, r2, rmse, func) in enumerate(without_results):
        try:
            y_fit = func(x_extended, *params)
            plt.plot(x_extended, y_fit, color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8, label=f'{name.split(":")[0]} (R²={r2:.4f})')
        except:
            pass
    
    plt.title(f'WITHOUT Crossover - All Curve Fits (from Gen {start_generation})', fontsize=14, fontweight='bold')
    plt.xlabel('Generation')
    plt.ylabel('Maximum Fitness')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Best Fits Comparison
    plt.subplot(2, 1, 2)
    
    # Original data
    plt.scatter(x_data, y_with, alpha=0.5, color='red', s=15, label='WITH Crossover Data')
    plt.scatter(x_data, y_without, alpha=0.5, color='blue', s=15, label='WITHOUT Crossover Data')
    
    # Best fit curves
    try:
        y_fit_with = best_fit_with(x_extended, *best_params_with)
        plt.plot(x_extended, y_fit_with, color='red', linewidth=3, 
                label=f'WITH Best Fit: {best_name_with.split(":")[0]}')
    except:
        pass
    
    try:
        y_fit_without = best_fit_without(x_extended, *best_params_without)
        plt.plot(x_extended, y_fit_without, color='blue', linewidth=3, 
                label=f'WITHOUT Best Fit: {best_name_without.split(":")[0]}')
    except:
        pass
    
    plt.title(f'Best Curve Fits Comparison (from Generation {start_generation})', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Maximum Fitness', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the curve fitting plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"curve_fitting_analysis_{timestamp}"
    
    plt.savefig(f"{filename_base}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_base}.png", format='png', dpi=300, bbox_inches='tight')
    
    print(f"\n📈 Curve fitting plots saved:")
    print(f"  {filename_base}.pdf")
    print(f"  {filename_base}.png")
    
    plt.show()

def find_latest_npz_file():
    """
    Find the most recent NPZ file in current directory
    """
    npz_files = [f for f in os.listdir('.') if f.endswith('_raw_data.npz')]
    if not npz_files:
        return None
    
    # Sort by filename (which contains timestamp)
    npz_files.sort(reverse=True)
    return npz_files[0]


def main():
    """
    Main function: load data and recreate plots
    """
    print("=" * 80)
    print("GENETIC ALGORITHM DATA PLOTTER - LOADING FROM SAVED DATA")
    print("=" * 80)
    
    # Find latest NPZ file
    latest_npz = find_latest_npz_file()
    if latest_npz:
        print(f"Found latest NPZ file: {latest_npz}")
        npz_file_path = latest_npz
    else:
        # If no NPZ file found, use default name pattern
        npz_file_path = "genetic_algorithm_comparison_20250702_212325_raw_data.npz"
        print(f"Using specified NPZ file: {npz_file_path}")
    
    try:
        # Load data
        loader = ExperimentDataLoader(npz_file_path)
        loader.load_data()
        loader.create_mock_experiment_objects()
        
        print(f"\n" + "="*50)
        print("RECREATING PLOTS FROM SAVED DATA")
        print("="*50)
        
        # Plot results
        plot_comparison_results_from_data(loader.exp_with, loader.exp_without, loader)
        
        print(f"\n" + "="*50)
        print("SUCCESS: Plots recreated from saved data!")
        print("="*50)
        
        # Perform curve fitting analysis
        print(f"\n" + "="*50)
        print("STARTING CURVE FITTING ANALYSIS")
        print("="*50)
        
        curve_analysis = analyze_fitness_curves(loader.exp_with, loader.exp_without, start_generation=1000)
        
        print(f"\n" + "="*50)
        print("CURVE FITTING ANALYSIS COMPLETED!")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the NPZ file exists in the current directory.")
        print("Available files:")
        for f in os.listdir('.'):
            if f.endswith('.npz'):
                print(f"  {f}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
