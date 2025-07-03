# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations

# Set matplotlib font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class GeneticAlgorithmExperimentWithCrossover:
    def __init__(self, initial_population_size=10, individual_length=15, 
                 value_range=(1, 30), generations=100, retention_probability=0.2, 
                 mutation_probability=0.05, population_limit=1000):
        """
        Initialize genetic algorithm experiment WITH crossover (complete genetic algorithm)
        """
        self.initial_population_size = initial_population_size
        self.individual_length = individual_length
        self.value_range = value_range
        self.generations = generations
        self.retention_probability = retention_probability
        self.mutation_probability = mutation_probability
        self.population_limit = population_limit
        
        # Calculate C value: number of combinations of selecting 2 individuals from initial population
        self.C = self.initial_population_size * (self.initial_population_size - 1) // 2
        
        # Store results
        self.P_values = []
        self.max_fitness_values = []
        self.offspring_retained_count = []
        self.offspring_generated_count = []
        self.mutation_count = []
        self.eliminated_count = []
        
        # Initialize population
        self.population = self.initialize_population()
    
    def initialize_population(self):
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
        return np.sum(individual**2)  # L2 norm squared (sum of squares)
    
    def crossover(self, parent1, parent2, crossover_point):
        offspring = np.concatenate([
            parent1[:crossover_point], 
            parent2[crossover_point:]
        ])
        return offspring
    
    def mutate(self, individual):
        mutated_individual = individual.copy()
        mutation_occurred = False
        
        if random.random() < self.mutation_probability:
            mutation_position = random.randint(0, self.individual_length - 1)
            increment = random.choice([1, -1])
            new_value = mutated_individual[mutation_position] + increment
            if new_value < 1:
                new_value = 1
            mutated_individual[mutation_position] = new_value
            mutation_occurred = True
        
        return mutated_individual, mutation_occurred
    
    def eliminate_excess_population(self):
        current_size = len(self.population)
        if current_size <= self.population_limit:
            return 0
        
        fitness_scores = [(i, self.calculate_fitness(individual)) 
                         for i, individual in enumerate(self.population)]
        fitness_scores.sort(key=lambda x: x[1])
        
        excess_count = current_size - self.population_limit
        indices_to_eliminate = [fitness_scores[i][0] for i in range(excess_count)]
        indices_to_eliminate.sort(reverse=True)
        
        for idx in indices_to_eliminate:
            self.population.pop(idx)
        
        return excess_count
    
    def select_random_pairs(self, population_size, num_pairs):
        selected_pairs = set()
        max_possible_pairs = population_size * (population_size - 1) // 2
        
        if num_pairs >= max_possible_pairs:
            return list(combinations(range(population_size), 2))
        
        while len(selected_pairs) < num_pairs:
            idx1, idx2 = random.sample(range(population_size), 2)
            pair = (min(idx1, idx2), max(idx1, idx2))
            selected_pairs.add(pair)
        
        return list(selected_pairs)
    
    def run_one_generation(self):
        current_population_size = len(self.population)
        selected_pairs = self.select_random_pairs(current_population_size, self.C)
        
        offspring_to_retain = []
        offspring_fitness_better_count = 0
        total_offspring_generated = 0
        generation_mutations = 0
        
        for parent1_idx, parent2_idx in selected_pairs:
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            f1 = self.calculate_fitness(parent1)
            f2 = self.calculate_fitness(parent2)
            
            # Generate a unique crossover point for each offspring
            crossover_point = random.randint(1, self.individual_length - 1)
            offspring = self.crossover(parent1, parent2, crossover_point)
            offspring, mutation_occurred = self.mutate(offspring)
            if mutation_occurred:
                generation_mutations += 1
            
            f3 = self.calculate_fitness(offspring)
            total_offspring_generated += 1
            
            if f3 > f1 and f3 > f2:
                offspring_fitness_better_count += 1
                offspring_to_retain.append(offspring)
            else:
                if random.random() < self.retention_probability:
                    offspring_to_retain.append(offspring)
        
        P = offspring_fitness_better_count / len(selected_pairs) if selected_pairs else 0
        self.population.extend(offspring_to_retain)
        
        self.offspring_generated_count.append(total_offspring_generated)
        self.offspring_retained_count.append(len(offspring_to_retain))
        self.mutation_count.append(generation_mutations)
        
        eliminated_count = self.eliminate_excess_population()
        self.eliminated_count.append(eliminated_count)
        
        current_fitness_values = [self.calculate_fitness(ind) for ind in self.population]
        max_fitness = max(current_fitness_values)
        self.max_fitness_values.append(max_fitness)
        
        return P
    
    def run_experiment(self):
        print(f"Starting WITH CROSSOVER experiment...")
        print(f"Initial population size: {self.initial_population_size}")
        print(f"Individual length: {self.individual_length}")
        print(f"Generations: {self.generations}, C: {self.C}")
        print(f"Retention prob: {self.retention_probability}, Mutation prob: {self.mutation_probability}")
        print(f"Population limit: {self.population_limit}")
        print("-" * 50)
        
        initial_fitness = [self.calculate_fitness(ind) for ind in self.population]
        initial_max_fitness = max(initial_fitness)
        self.max_fitness_values.append(initial_max_fitness)
        print(f"Initial fitness values: {initial_fitness}")
        print(f"Initial average: {np.mean(initial_fitness):.2f}, max: {initial_max_fitness}")
        print("-" * 50)
        
        for generation in range(self.generations):
            P = self.run_one_generation()
            self.P_values.append(P)
            
            if generation % 10 == 0 or generation == self.generations - 1:
                current_fitness = [self.calculate_fitness(ind) for ind in self.population]
                retained_count = self.offspring_retained_count[generation]
                generated_count = self.offspring_generated_count[generation]
                retention_rate = retained_count / generated_count * 100 if generated_count > 0 else 0
                
                print(f"Gen {generation + 1:3d}: P = {P:.4f}, "
                      f"Pop = {len(self.population):4d}, "
                      f"Avg = {np.mean(current_fitness):6.2f}, "
                      f"Retained: {retained_count}/{generated_count} ({retention_rate:.1f}%)")
        
        print("WITH CROSSOVER experiment completed!")
        return self.P_values


class GeneticAlgorithmExperimentNoCrossover:
    def __init__(self, initial_population_size=10, individual_length=15, 
                 value_range=(1, 30), generations=100, retention_probability=0.2, 
                 mutation_probability=0.05, population_limit=1000):
        """
        Initialize genetic algorithm experiment WITHOUT crossover (mutation-only evolution)
        """
        self.initial_population_size = initial_population_size
        self.individual_length = individual_length
        self.value_range = value_range
        self.generations = generations
        self.retention_probability = retention_probability
        self.mutation_probability = mutation_probability
        self.population_limit = population_limit
        
        self.C = self.initial_population_size * (self.initial_population_size - 1) // 2
        
        # Store results
        self.P_values = []
        self.max_fitness_values = []
        self.offspring_retained_count = []
        self.offspring_generated_count = []
        self.mutation_count = []
        self.eliminated_count = []
        
        # Initialize population
        self.population = self.initialize_population()
    
    def initialize_population(self):
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
        return np.sum(individual**2)  # L2 norm squared (sum of squares)
    
    def mutate(self, individual):
        mutated_individual = individual.copy()
        mutation_occurred = False
        
        if random.random() < self.mutation_probability:
            mutation_position = random.randint(0, self.individual_length - 1)
            increment = random.choice([1, -1])
            new_value = mutated_individual[mutation_position] + increment
            if new_value < 1:
                new_value = 1
            mutated_individual[mutation_position] = new_value
            mutation_occurred = True
        
        return mutated_individual, mutation_occurred
    
    def eliminate_excess_population(self):
        current_size = len(self.population)
        if current_size <= self.population_limit:
            return 0
        
        fitness_scores = [(i, self.calculate_fitness(individual)) 
                         for i, individual in enumerate(self.population)]
        fitness_scores.sort(key=lambda x: x[1])
        
        excess_count = current_size - self.population_limit
        indices_to_eliminate = [fitness_scores[i][0] for i in range(excess_count)]
        indices_to_eliminate.sort(reverse=True)
        
        for idx in indices_to_eliminate:
            self.population.pop(idx)
        
        return excess_count
    
    def select_random_individuals(self, population_size, num_individuals):
        return [random.randint(0, population_size - 1) for _ in range(num_individuals)]
    
    def run_one_generation(self):
        current_population_size = len(self.population)
        selected_indices = self.select_random_individuals(current_population_size, self.C)
        
        offspring_to_retain = []
        offspring_fitness_better_count = 0
        total_offspring_generated = 0
        generation_mutations = 0
        
        for parent_idx in selected_indices:
            parent = self.population[parent_idx]
            parent_fitness = self.calculate_fitness(parent)
            
            offspring = parent.copy()
            offspring, mutation_occurred = self.mutate(offspring)
            if mutation_occurred:
                generation_mutations += 1
            
            offspring_fitness = self.calculate_fitness(offspring)
            total_offspring_generated += 1
            
            if offspring_fitness > parent_fitness:
                offspring_fitness_better_count += 1
                offspring_to_retain.append(offspring)
            else:
                if random.random() < self.retention_probability:
                    offspring_to_retain.append(offspring)
        
        P = offspring_fitness_better_count / len(selected_indices) if selected_indices else 0
        self.population.extend(offspring_to_retain)
        
        self.offspring_generated_count.append(total_offspring_generated)
        self.offspring_retained_count.append(len(offspring_to_retain))
        self.mutation_count.append(generation_mutations)
        
        eliminated_count = self.eliminate_excess_population()
        self.eliminated_count.append(eliminated_count)
        
        current_fitness_values = [self.calculate_fitness(ind) for ind in self.population]
        max_fitness = max(current_fitness_values)
        self.max_fitness_values.append(max_fitness)
        
        return P
    
    def run_experiment(self):
        print(f"Starting WITHOUT CROSSOVER experiment...")
        print(f"Initial population size: {self.initial_population_size}")
        print(f"Individual length: {self.individual_length}")
        print(f"Generations: {self.generations}, C: {self.C}")
        print(f"Retention prob: {self.retention_probability}, Mutation prob: {self.mutation_probability}")
        print(f"Population limit: {self.population_limit}")
        print("*** NO CROSSOVER - MUTATION-ONLY EVOLUTION ***")
        print("-" * 50)
        
        initial_fitness = [self.calculate_fitness(ind) for ind in self.population]
        initial_max_fitness = max(initial_fitness)
        self.max_fitness_values.append(initial_max_fitness)
        print(f"Initial fitness values: {initial_fitness}")
        print(f"Initial average: {np.mean(initial_fitness):.2f}, max: {initial_max_fitness}")
        print("-" * 50)
        
        for generation in range(self.generations):
            P = self.run_one_generation()
            self.P_values.append(P)
            
            if generation % 10 == 0 or generation == self.generations - 1:
                current_fitness = [self.calculate_fitness(ind) for ind in self.population]
                retained_count = self.offspring_retained_count[generation]
                generated_count = self.offspring_generated_count[generation]
                retention_rate = retained_count / generated_count * 100 if generated_count > 0 else 0
                
                print(f"Gen {generation + 1:3d}: P = {P:.4f}, "
                      f"Pop = {len(self.population):4d}, "
                      f"Avg = {np.mean(current_fitness):6.2f}, "
                      f"Retained: {retained_count}/{generated_count} ({retention_rate:.1f}%)")
        
        print("WITHOUT CROSSOVER experiment completed!")
        return self.P_values


def plot_comparison_results(exp_with, exp_without):
    """
    Plot comparison results between with and without crossover experiments
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
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"genetic_algorithm_comparison_{timestamp}"
    
    # Save as PDF (vector format)
    plt.savefig(f"{filename_base}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot as: {filename_base}.pdf")
    
    # Save as SVG (vector format)
    plt.savefig(f"{filename_base}.svg", format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot as: {filename_base}.svg")
    
    # Save as high-resolution PNG for backup
    plt.savefig(f"{filename_base}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot as: {filename_base}.png")
    
    # Save raw data as numpy arrays
    raw_data = {
        'generations_list': np.array(generations_list),
        'max_fitness_generations': np.array(max_fitness_generations),
        'P_values_with_crossover': np.array(exp_with.P_values),
        'P_values_without_crossover': np.array(exp_without.P_values),
        'max_fitness_with_crossover': np.array(exp_with.max_fitness_values),
        'max_fitness_without_crossover': np.array(exp_without.max_fitness_values),
        'retention_rates_with_crossover': np.array(retention_rates_with),
        'retention_rates_without_crossover': np.array(retention_rates_without),
        'offspring_retained_with_crossover': np.array(exp_with.offspring_retained_count),
        'offspring_generated_with_crossover': np.array(exp_with.offspring_generated_count),
        'offspring_retained_without_crossover': np.array(exp_without.offspring_retained_count),
        'offspring_generated_without_crossover': np.array(exp_without.offspring_generated_count),
        'mutation_count_with_crossover': np.array(exp_with.mutation_count),
        'mutation_count_without_crossover': np.array(exp_without.mutation_count),
        'eliminated_count_with_crossover': np.array(exp_with.eliminated_count),
        'eliminated_count_without_crossover': np.array(exp_without.eliminated_count),
        'moving_average_with_crossover': np.array(moving_avg_with) if len(exp_with.P_values) >= window_size else np.array([]),
        'moving_average_without_crossover': np.array(moving_avg_without) if len(exp_without.P_values) >= window_size else np.array([]),
        'moving_avg_generations': np.array(moving_avg_generations) if len(exp_with.P_values) >= window_size else np.array([]),
        'experiment_parameters': {
            'initial_population_size': exp_with.initial_population_size,
            'individual_length': exp_with.individual_length,
            'generations': exp_with.generations,
            'retention_probability': exp_with.retention_probability,
            'mutation_probability': exp_with.mutation_probability,
            'population_limit': exp_with.population_limit,
            'value_range': exp_with.value_range,
            'C_value': exp_with.C
        }
    }
    
    # Save raw data as NPZ file
    np.savez_compressed(f"{filename_base}_raw_data.npz", **raw_data)
    print(f"Saved raw data as: {filename_base}_raw_data.npz")
    
    plt.show()
    
    # Print comprehensive comparison statistics
    print(f"\n" + "="*80)
    print("COMPREHENSIVE COMPARISON RESULTS")
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


def main():
    """
    Main function: run both experiments and compare results
    """
    # Set same parameters for both experiments
    params = {
        'initial_population_size': 20,
        'individual_length': 100,
        'value_range': (1, 30),
        'generations': 100000,
        'retention_probability': 0.2,
        'mutation_probability': 0.05,
        'population_limit': 1000
    }
    
    print("=" * 80)
    print("GENETIC ALGORITHM COMPARISON: WITH vs WITHOUT CROSSOVER")
    print("=" * 80)
    
    # Run experiment WITH crossover
    print("\n" + "="*50)
    print("EXPERIMENT 1: WITH CROSSOVER")
    print("="*50)
    exp_with_crossover = GeneticAlgorithmExperimentWithCrossover(**params)
    exp_with_crossover.run_experiment()
    
    print("\n" + "="*50)
    print("EXPERIMENT 2: WITHOUT CROSSOVER")
    print("="*50)
    exp_without_crossover = GeneticAlgorithmExperimentNoCrossover(**params)
    exp_without_crossover.run_experiment()
    
    # Plot comparison results
    plot_comparison_results(exp_with_crossover, exp_without_crossover)
    
    return exp_with_crossover, exp_without_crossover


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed()
    random.seed()
    
    exp_with, exp_without = main()
