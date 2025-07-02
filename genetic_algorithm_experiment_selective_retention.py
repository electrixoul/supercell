# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations

# Set matplotlib font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class GeneticAlgorithmExperimentSelectiveRetention:
    def __init__(self, initial_population_size=10, individual_length=15, 
                 value_range=(1, 30), generations=100, retention_probability=0.2):
        """
        Initialize genetic algorithm experiment with selective retention
        
        Args:
            initial_population_size: Initial population size
            individual_length: Individual length (array length)
            value_range: Value range (min, max)
            generations: Number of generations to run
            retention_probability: Probability of retaining offspring with fitness <= parents
        """
        self.initial_population_size = initial_population_size
        self.individual_length = individual_length
        self.value_range = value_range
        self.generations = generations
        self.retention_probability = retention_probability
        
        # Calculate C value: number of combinations of selecting 2 individuals from initial population
        self.C = self.initial_population_size * (self.initial_population_size - 1) // 2
        
        # Store P values for each generation
        self.P_values = []
        
        # Store maximum fitness values for each generation
        self.max_fitness_values = []
        
        # Store offspring retention statistics
        self.offspring_retained_count = []
        self.offspring_generated_count = []
        
        # Initialize population
        self.population = self.initialize_population()
    
    def initialize_population(self):
        """
        Initialize population: generate initial individual arrays, each individual is an integer array of length 15 with values between 1-30
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
        
        Args:
            parent1, parent2: Parent individuals
            crossover_point: Cut point position
        
        Returns:
            offspring: Offspring individual
        """
        # Take left part of parent1 (excluding cut point) and right part of parent2 (including cut point)
        offspring = np.concatenate([
            parent1[:crossover_point], 
            parent2[crossover_point:]
        ])
        return offspring
    
    def select_random_pairs(self, population_size, num_pairs):
        """
        Directly select num_pairs random unique pairs without generating all combinations
        
        Args:
            population_size: Current population size
            num_pairs: Number of pairs to select (C value)
        
        Returns:
            List of selected pairs as tuples (idx1, idx2)
        """
        selected_pairs = set()
        max_possible_pairs = population_size * (population_size - 1) // 2
        
        # If requesting more pairs than possible, return all possible pairs
        if num_pairs >= max_possible_pairs:
            return list(combinations(range(population_size), 2))
        
        # Efficiently select random pairs without generating all combinations
        while len(selected_pairs) < num_pairs:
            # Randomly select two different indices
            idx1, idx2 = random.sample(range(population_size), 2)
            # Standardize pair order (smaller index first)
            pair = (min(idx1, idx2), max(idx1, idx2))
            selected_pairs.add(pair)
        
        return list(selected_pairs)
    
    def run_one_generation(self):
        """
        Run complete process of one generation with selective retention
        
        Returns:
            P: Proportion of offspring with fitness values greater than both parents
        """
        # 1. Directly select C pairs of individuals (OPTIMIZED - no combinations generation)
        current_population_size = len(self.population)
        selected_pairs = self.select_random_pairs(current_population_size, self.C)
        
        # 2. Choose same cut point for each pair (both sides have non-zero length)
        crossover_point = random.randint(1, self.individual_length - 1)
        
        # 3. Generate offspring and calculate fitness values
        offspring_to_retain = []
        offspring_fitness_better_count = 0
        total_offspring_generated = 0
        
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
            
            total_offspring_generated += 1
            
            # Check if offspring fitness is greater than both parent fitness values
            if f3 > f1 and f3 > f2:
                offspring_fitness_better_count += 1
                # Always retain offspring with better fitness than both parents
                offspring_to_retain.append(offspring)
            else:
                # Selective retention: 20% probability for offspring with fitness <= parents
                if random.random() < self.retention_probability:
                    offspring_to_retain.append(offspring)
        
        # 4. Calculate P value: proportion of offspring with better fitness than parents
        P = offspring_fitness_better_count / len(selected_pairs) if selected_pairs else 0
        
        # 5. Add retained offspring to population
        self.population.extend(offspring_to_retain)
        
        # 6. Record retention statistics
        self.offspring_generated_count.append(total_offspring_generated)
        self.offspring_retained_count.append(len(offspring_to_retain))
        
        # 7. Calculate and record maximum fitness value for current generation
        current_fitness_values = [self.calculate_fitness(ind) for ind in self.population]
        max_fitness = max(current_fitness_values)
        self.max_fitness_values.append(max_fitness)
        
        return P
    
    def run_experiment(self):
        """
        Run complete genetic algorithm experiment with selective retention
        """
        print(f"Starting genetic algorithm experiment with selective retention...")
        print(f"Initial population size: {self.initial_population_size}")
        print(f"Individual length: {self.individual_length}")
        print(f"Value range: {self.value_range}")
        print(f"Number of generations: {self.generations}")
        print(f"Offspring per generation (C): {self.C}")
        print(f"Retention probability for weaker offspring: {self.retention_probability}")
        print("-" * 50)
        
        # Record initial population information
        initial_fitness = [self.calculate_fitness(ind) for ind in self.population]
        initial_max_fitness = max(initial_fitness)
        self.max_fitness_values.append(initial_max_fitness)  # Record initial max fitness
        print(f"Initial population fitness values: {initial_fitness}")
        print(f"Initial population average fitness: {np.mean(initial_fitness):.2f}")
        print(f"Initial population max fitness: {initial_max_fitness}")
        print("-" * 50)
        
        # Run generations
        for generation in range(self.generations):
            P = self.run_one_generation()
            self.P_values.append(P)
            
            # Output progress every 10 generations
            if generation % 10 == 0 or generation == self.generations - 1:
                current_fitness = [self.calculate_fitness(ind) for ind in self.population]
                retained_count = self.offspring_retained_count[generation]
                generated_count = self.offspring_generated_count[generation]
                retention_rate = retained_count / generated_count * 100 if generated_count > 0 else 0
                
                print(f"Generation {generation + 1:3d}: P = {P:.4f}, "
                      f"Population size = {len(self.population):4d}, "
                      f"Average fitness = {np.mean(current_fitness):6.2f}, "
                      f"Retained: {retained_count}/{generated_count} ({retention_rate:.1f}%)")
        
        print("-" * 50)
        print("Experiment completed!")
        
        # Print retention statistics
        self.print_retention_statistics()
        
        # Print 10 randomly selected offspring from the final population
        self.print_random_offspring()
        
        return self.P_values
    
    def print_retention_statistics(self):
        """
        Print statistics about offspring retention
        """
        total_generated = sum(self.offspring_generated_count)
        total_retained = sum(self.offspring_retained_count)
        overall_retention_rate = total_retained / total_generated * 100 if total_generated > 0 else 0
        
        print(f"\n=== Offspring Retention Statistics ===")
        print(f"Total offspring generated: {total_generated}")
        print(f"Total offspring retained: {total_retained}")
        print(f"Overall retention rate: {overall_retention_rate:.2f}%")
        print(f"Average retained per generation: {np.mean(self.offspring_retained_count):.2f}")
        print(f"Min/Max retained per generation: {np.min(self.offspring_retained_count)}/{np.max(self.offspring_retained_count)}")
        print("-" * 50)
    
    def print_random_offspring(self, num_offspring=10):
        """
        Print randomly selected offspring from the current population
        """
        print(f"\n=== Random Sample of {num_offspring} Offspring ===")
        
        # Get offspring from the population (exclude the initial 10 individuals)
        if len(self.population) > self.initial_population_size:
            offspring_population = self.population[self.initial_population_size:]
            
            # Randomly select offspring to display
            num_to_show = min(num_offspring, len(offspring_population))
            selected_offspring = random.sample(offspring_population, num_to_show)
            
            for i, offspring in enumerate(selected_offspring, 1):
                fitness = self.calculate_fitness(offspring)
                print(f"Offspring {i:2d}: {offspring.tolist()} | Fitness: {fitness}")
        else:
            print("No offspring generated yet.")
        print("-" * 50)
    
    def plot_results(self):
        """
        Plot P value changes, maximum fitness evolution, and retention statistics over generations
        """
        plt.figure(figsize=(12, 16))
        
        generations_list = list(range(1, self.generations + 1))
        # For max fitness, we include generation 0 (initial population)
        max_fitness_generations = list(range(0, self.generations + 1))
        
        # Plot P value curve
        plt.subplot(4, 1, 1)
        plt.plot(generations_list, self.P_values, linewidth=2, color='blue', marker='o', markersize=3)
        plt.title(f'P Value Changes Over Generations (Initial Population={self.initial_population_size}, C={self.C})', fontsize=14)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('P Value (Proportion of Better Offspring)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add statistical information
        mean_P = np.mean(self.P_values)
        plt.axhline(y=mean_P, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average P Value = {mean_P:.4f}')
        plt.legend()
        
        # Plot maximum fitness evolution curve
        plt.subplot(4, 1, 2)
        plt.plot(max_fitness_generations, self.max_fitness_values, linewidth=2, color='red', marker='s', markersize=3)
        plt.title(f'Maximum Fitness Evolution Over Generations', fontsize=14)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Maximum Fitness Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add fitness improvement information
        initial_max_fitness = self.max_fitness_values[0]
        final_max_fitness = self.max_fitness_values[-1]
        improvement = final_max_fitness - initial_max_fitness
        plt.axhline(y=initial_max_fitness, color='green', linestyle='--', alpha=0.7, 
                   label=f'Initial Max Fitness = {initial_max_fitness}')
        plt.axhline(y=final_max_fitness, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Final Max Fitness = {final_max_fitness}')
        plt.legend()
        
        # Plot offspring retention statistics
        plt.subplot(4, 1, 3)
        retention_rates = [retained/generated*100 if generated > 0 else 0 
                          for retained, generated in zip(self.offspring_retained_count, self.offspring_generated_count)]
        plt.plot(generations_list, retention_rates, linewidth=2, color='purple', marker='d', markersize=3)
        plt.title(f'Offspring Retention Rate Over Generations', fontsize=14)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Retention Rate (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Add average retention rate line
        avg_retention_rate = np.mean(retention_rates)
        plt.axhline(y=avg_retention_rate, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average Retention Rate = {avg_retention_rate:.1f}%')
        plt.legend()
        
        # Plot moving average of P values
        plt.subplot(4, 1, 4)
        window_size = 10
        if len(self.P_values) >= window_size:
            moving_avg = np.convolve(self.P_values, np.ones(window_size)/window_size, mode='valid')
            moving_avg_generations = generations_list[window_size-1:]
            plt.plot(moving_avg_generations, moving_avg, linewidth=2, color='green', 
                    label=f'{window_size}-Generation Moving Average')
            plt.legend()
        
        plt.plot(generations_list, self.P_values, alpha=0.5, color='lightblue')
        plt.title(f'P Value Moving Average Trend', fontsize=14)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('P Value (Moving Average)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # Output statistical information
        print(f"\n=== Experiment Results Statistics ===")
        print(f"P value range: [{np.min(self.P_values):.4f}, {np.max(self.P_values):.4f}]")
        print(f"P value average: {np.mean(self.P_values):.4f}")
        print(f"P value standard deviation: {np.std(self.P_values):.4f}")
        print(f"Generations with P=0: {np.sum(np.array(self.P_values) == 0)} / {self.generations}")
        print(f"Generations with P=1: {np.sum(np.array(self.P_values) == 1)} / {self.generations}")
        
        # Add maximum fitness statistics
        print(f"\n=== Maximum Fitness Statistics ===")
        print(f"Initial max fitness: {self.max_fitness_values[0]}")
        print(f"Final max fitness: {self.max_fitness_values[-1]}")
        print(f"Fitness improvement: {self.max_fitness_values[-1] - self.max_fitness_values[0]}")
        print(f"Max fitness range: [{np.min(self.max_fitness_values)}, {np.max(self.max_fitness_values)}]")
        print(f"Average max fitness: {np.mean(self.max_fitness_values):.2f}")


def main():
    """
    Main function: run genetic algorithm experiment with selective retention
    """
    # Create experiment instance (reduced parameters for testing)
    experiment = GeneticAlgorithmExperimentSelectiveRetention(
        initial_population_size=10,
        individual_length=100,  # Reduced for faster testing
        value_range=(1, 30),
        generations=1000,  # Reduced for faster testing
        retention_probability=0.0002  # 20% probability for weaker offspring
    )
    
    # Run experiment
    P_values = experiment.run_experiment()
    
    # Plot results
    experiment.plot_results()
    
    return experiment, P_values


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed()
    random.seed()
    
    print("=" * 60)
    print("Genetic Algorithm Experiment with Selective Retention")
    print("=" * 60)
    
    experiment, P_values = main()
