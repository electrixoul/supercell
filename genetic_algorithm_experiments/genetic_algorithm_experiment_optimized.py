# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations

# Set matplotlib font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class GeneticAlgorithmExperimentOptimized:
    def __init__(self, initial_population_size=10, individual_length=15, 
                 value_range=(1, 30), generations=100, maintain_population_size=True):
        """
        Initialize genetic algorithm experiment with optimization options
        
        Args:
            initial_population_size: Initial population size
            individual_length: Individual length (array length)
            value_range: Value range (min, max)
            generations: Number of generations to run
            maintain_population_size: If True, keep population size constant for performance
        """
        self.initial_population_size = initial_population_size
        self.individual_length = individual_length
        self.value_range = value_range
        self.generations = generations
        self.maintain_population_size = maintain_population_size
        
        # Calculate C value: number of combinations of selecting 2 individuals from initial population
        self.C = self.initial_population_size * (self.initial_population_size - 1) // 2
        
        # Store P values and timing information for each generation
        self.P_values = []
        self.generation_times = []
        self.population_sizes = []
        
        # Initialize population
        self.population = self.initialize_population()
    
    def initialize_population(self):
        """
        Initialize population: generate initial individual arrays
        """
        population = []
        for _ in range(self.initial_population_size):
            individual = np.random.randint(
                self.value_range[0], 
                self.value_range[1] + 1, 
                size=self.individual_length
            )
            population.append(individual)
        return population
    
    def calculate_fitness(self, individual):
        """
        Calculate individual's fitness value: sum of array elements
        """
        return np.sum(individual)
    
    def crossover(self, parent1, parent2, crossover_point):
        """
        Crossover operation: perform crossover at specified cut point
        """
        offspring = np.concatenate([
            parent1[:crossover_point], 
            parent2[crossover_point:]
        ])
        return offspring
    
    def select_survivors(self, population, target_size):
        """
        Select survivors to maintain population size
        Strategy: Keep best individuals based on fitness
        """
        # Calculate fitness for all individuals
        fitness_scores = [(i, self.calculate_fitness(ind)) for i, ind in enumerate(population)]
        
        # Sort by fitness (descending order - higher fitness is better)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top individuals
        selected_indices = [idx for idx, _ in fitness_scores[:target_size]]
        
        return [population[i] for i in selected_indices]
    
    def run_one_generation(self):
        """
        Run complete process of one generation with timing
        """
        import time
        start_time = time.time()
        
        current_pop_size = len(self.population)
        
        # 1. Generate pairs - optimize for large populations
        if current_pop_size <= 50:
            # For small populations, use all combinations
            population_indices = list(range(current_pop_size))
            all_pairs = list(combinations(population_indices, 2))
            selected_pairs = random.sample(all_pairs, min(self.C, len(all_pairs)))
        else:
            # For large populations, randomly sample pairs directly to avoid memory issues
            selected_pairs = []
            for _ in range(self.C):
                idx1, idx2 = random.sample(range(current_pop_size), 2)
                selected_pairs.append((idx1, idx2))
        
        # 2. Choose same cut point for all pairs
        crossover_point = random.randint(1, self.individual_length - 1)
        
        # 3. Generate offspring and calculate fitness values
        offspring_list = []
        offspring_fitness_better_count = 0
        
        for parent1_idx, parent2_idx in selected_pairs:
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Calculate parent fitness values
            f1 = self.calculate_fitness(parent1)
            f2 = self.calculate_fitness(parent2)
            
            # Generate offspring
            offspring = self.crossover(parent1, parent2, crossover_point)
            
            # Calculate offspring fitness value
            f3 = self.calculate_fitness(offspring)
            
            # Check if offspring fitness is greater than both parent fitness values
            if f3 > f1 and f3 > f2:
                offspring_fitness_better_count += 1
            
            offspring_list.append(offspring)
        
        # 4. Calculate P value
        P = offspring_fitness_better_count / len(selected_pairs) if selected_pairs else 0
        
        # 5. Add offspring to population
        self.population.extend(offspring_list)
        
        # 6. Population size management
        if self.maintain_population_size:
            # Keep population size manageable by selecting best individuals
            target_size = min(self.initial_population_size * 20, 200)  # Limit growth
            if len(self.population) > target_size:
                self.population = self.select_survivors(self.population, target_size)
        
        # Record timing and population size
        end_time = time.time()
        generation_time = end_time - start_time
        self.generation_times.append(generation_time)
        self.population_sizes.append(len(self.population))
        
        return P
    
    def run_experiment(self):
        """
        Run complete genetic algorithm experiment with performance monitoring
        """
        print(f"Starting optimized genetic algorithm experiment...")
        print(f"Initial population size: {self.initial_population_size}")
        print(f"Individual length: {self.individual_length}")
        print(f"Value range: {self.value_range}")
        print(f"Number of generations: {self.generations}")
        print(f"Offspring per generation (C): {self.C}")
        print(f"Population size management: {'Enabled' if self.maintain_population_size else 'Disabled'}")
        print("-" * 50)
        
        # Record initial population information
        initial_fitness = [self.calculate_fitness(ind) for ind in self.population]
        print(f"Initial population fitness values: {initial_fitness}")
        print(f"Initial population average fitness: {np.mean(initial_fitness):.2f}")
        print("-" * 50)
        
        # Run generations
        for generation in range(self.generations):
            P = self.run_one_generation()
            self.P_values.append(P)
            
            # Output progress every 10 generations
            if generation % 10 == 0 or generation == self.generations - 1:
                current_fitness = [self.calculate_fitness(ind) for ind in self.population]
                avg_time = np.mean(self.generation_times[-10:]) if len(self.generation_times) >= 10 else np.mean(self.generation_times)
                print(f"Generation {generation + 1:3d}: P = {P:.4f}, "
                      f"Population size = {len(self.population):4d}, "
                      f"Average fitness = {np.mean(current_fitness):6.2f}, "
                      f"Avg time/gen = {avg_time:.4f}s")
        
        print("-" * 50)
        print("Experiment completed!")
        return self.P_values
    
    def plot_results(self):
        """
        Plot comprehensive results including performance analysis
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        generations_list = list(range(1, self.generations + 1))
        
        # Plot 1: P value changes
        ax1.plot(generations_list, self.P_values, linewidth=2, color='blue', marker='o', markersize=2)
        ax1.set_title('P Value Changes Over Generations', fontsize=12)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('P Value')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        mean_P = np.mean(self.P_values)
        ax1.axhline(y=mean_P, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average = {mean_P:.4f}')
        ax1.legend()
        
        # Plot 2: Population size over time
        ax2.plot(generations_list, self.population_sizes, linewidth=2, color='green')
        ax2.set_title('Population Size Over Generations', fontsize=12)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Population Size')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Generation time analysis
        ax3.plot(generations_list, self.generation_times, linewidth=2, color='red', alpha=0.7)
        ax3.set_title('Generation Processing Time', fontsize=12)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True, alpha=0.3)
        
        # Add moving average for generation time
        if len(self.generation_times) >= 10:
            window_size = 10
            moving_avg = np.convolve(self.generation_times, np.ones(window_size)/window_size, mode='valid')
            moving_avg_generations = generations_list[window_size-1:]
            ax3.plot(moving_avg_generations, moving_avg, linewidth=3, color='darkred', 
                    label=f'{window_size}-gen avg')
            ax3.legend()
        
        # Plot 4: Computational complexity analysis
        theoretical_complexity = [(size * (size - 1) // 2) for size in self.population_sizes]
        ax4.plot(generations_list, theoretical_complexity, linewidth=2, color='purple')
        ax4.set_title('Theoretical Computational Complexity', fontsize=12)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Number of Possible Pairs (C(n,2))')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')  # Use log scale due to exponential growth
        
        plt.tight_layout()
        plt.show()
        
        # Output comprehensive statistics
        print(f"\n=== Experiment Results & Performance Analysis ===")
        print(f"P value range: [{np.min(self.P_values):.4f}, {np.max(self.P_values):.4f}]")
        print(f"P value average: {np.mean(self.P_values):.4f}")
        print(f"P value standard deviation: {np.std(self.P_values):.4f}")
        print(f"Generations with P=0: {np.sum(np.array(self.P_values) == 0)} / {self.generations}")
        print(f"Generations with P=1: {np.sum(np.array(self.P_values) == 1)} / {self.generations}")
        
        print(f"\n=== Performance Statistics ===")
        print(f"Average generation time: {np.mean(self.generation_times):.4f}s")
        print(f"Time increase factor: {self.generation_times[-1]/self.generation_times[0]:.2f}x")
        print(f"Population size range: {min(self.population_sizes)} → {max(self.population_sizes)}")
        print(f"Final theoretical complexity: C({max(self.population_sizes)},2) = {max(self.population_sizes)*(max(self.population_sizes)-1)//2:,}")


def compare_performance():
    """
    Compare performance between original and optimized versions
    """
    print("=" * 70)
    print("PERFORMANCE COMPARISON: Original vs Optimized")
    print("=" * 70)
    
    # Set same random seed for fair comparison
    np.random.seed()
    random.seed()
    
    print("\n--- Running Original Version (Unlimited Growth) ---")
    original = GeneticAlgorithmExperimentOptimized(
        initial_population_size=10,
        individual_length=15,
        value_range=(1, 30),
        generations=50,  # Reduced generations for demo
        maintain_population_size=False
    )
    original.run_experiment()
    
    # Reset random seed
    np.random.seed()
    random.seed()
    
    print("\n--- Running Optimized Version (Population Management) ---")
    optimized = GeneticAlgorithmExperimentOptimized(
        initial_population_size=10,
        individual_length=15,
        value_range=(1, 30),
        generations=50,  # Same number of generations
        maintain_population_size=True
    )
    optimized.run_experiment()
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS:")
    print("=" * 50)
    print(f"Original - Final population: {original.population_sizes[-1]}")
    print(f"Original - Average time/gen: {np.mean(original.generation_times):.4f}s")
    print(f"Original - Time increase: {original.generation_times[-1]/original.generation_times[0]:.1f}x")
    
    print(f"Optimized - Final population: {optimized.population_sizes[-1]}")
    print(f"Optimized - Average time/gen: {np.mean(optimized.generation_times):.4f}s")
    print(f"Optimized - Time increase: {optimized.generation_times[-1]/optimized.generation_times[0]:.1f}x")
    
    speedup = np.mean(original.generation_times) / np.mean(optimized.generation_times)
    print(f"Performance improvement: {speedup:.1f}x faster")


def main():
    """
    Main function: run optimized genetic algorithm experiment
    """
    # Create optimized experiment instance
    experiment = GeneticAlgorithmExperimentOptimized(
        initial_population_size=10,
        individual_length=15,
        value_range=(1, 30),
        generations=100,
        maintain_population_size=True  # Enable optimization
    )
    
    # Run experiment
    P_values = experiment.run_experiment()
    
    # Plot comprehensive results
    experiment.plot_results()
    
    return experiment, P_values


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed()
    random.seed()
    
    print("=" * 70)
    print("OPTIMIZED Genetic Algorithm Fitness Evolution Experiment")
    print("=" * 70)
    
    # Run main experiment
    experiment, P_values = main()
    
    # Optional: Run performance comparison
    print("\n" + "="*50)
    print("Would you like to see performance comparison? (This will run additional tests)")
    print("Uncomment the line below to enable:")
    print("# compare_performance()")
