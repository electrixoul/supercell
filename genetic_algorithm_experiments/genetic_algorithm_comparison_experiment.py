# -*- coding: utf-8 -*-
"""
Genetic Algorithm Comparison Experiment
Clean Code Version - Compares GA with and without crossover
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
from itertools import combinations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# Configure matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class ExperimentConfig:
    """Configuration for genetic algorithm experiments"""
    population_size: int = 20
    individual_length: int = 100
    value_range: Tuple[int, int] = (1, 30)
    generations: int = 1000
    retention_probability: float = 0.2
    mutation_probability: float = 0.05
    population_limit: int = 1000
    
    @property
    def crossover_pairs(self) -> int:
        """Number of parent pairs for crossover"""
        return self.population_size * (self.population_size - 1) // 2


@dataclass
class GenerationStats:
    """Statistics for a single generation"""
    p_value: float
    offspring_generated: int
    offspring_retained: int
    mutations: int
    eliminated: int
    max_fitness: float


class Individual:
    """Represents a single individual in the population"""
    
    def __init__(self, genes: np.ndarray):
        self.genes = genes
    
    @classmethod
    def random(cls, length: int, value_range: Tuple[int, int]) -> 'Individual':
        """Create a random individual"""
        genes = np.random.randint(value_range[0], value_range[1] + 1, size=length)
        return cls(genes)
    
    @property
    def fitness(self) -> float:
        """Calculate fitness (sum of squares)"""
        return np.sum(self.genes ** 2)
    
    def copy(self) -> 'Individual':
        """Create a copy of this individual"""
        return Individual(self.genes.copy())
    
    def mutate(self, probability: float) -> bool:
        """Mutate this individual. Returns True if mutation occurred."""
        if random.random() < probability:
            position = random.randint(0, len(self.genes) - 1)
            change = random.choice([1, -1])
            self.genes[position] = max(1, self.genes[position] + change)
            return True
        return False
    
    def crossover_with(self, other: 'Individual') -> 'Individual':
        """Create offspring through crossover with another individual"""
        crossover_point = random.randint(1, len(self.genes) - 1)
        offspring_genes = np.concatenate([
            self.genes[:crossover_point], 
            other.genes[crossover_point:]
        ])
        return Individual(offspring_genes)


class Population:
    """Manages a population of individuals"""
    
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals
    
    @classmethod
    def random(cls, config: ExperimentConfig) -> 'Population':
        """Create a random population"""
        individuals = [
            Individual.random(config.individual_length, config.value_range)
            for _ in range(config.population_size)
        ]
        return cls(individuals)
    
    def __len__(self) -> int:
        return len(self.individuals)
    
    def __getitem__(self, index: int) -> Individual:
        return self.individuals[index]
    
    def add(self, individual: Individual):
        """Add an individual to the population"""
        self.individuals.append(individual)
    
    def extend(self, individuals: List[Individual]):
        """Add multiple individuals to the population"""
        self.individuals.extend(individuals)
    
    @property
    def fitness_values(self) -> List[float]:
        """Get fitness values for all individuals"""
        return [ind.fitness for ind in self.individuals]
    
    @property
    def max_fitness(self) -> float:
        """Get maximum fitness in population"""
        return max(self.fitness_values)
    
    @property
    def average_fitness(self) -> float:
        """Get average fitness in population"""
        return np.mean(self.fitness_values)
    
    def limit_size(self, max_size: int) -> int:
        """Remove weakest individuals to stay within size limit"""
        if len(self) <= max_size:
            return 0
        
        # Sort by fitness (ascending - weakest first)
        sorted_individuals = sorted(
            enumerate(self.individuals), 
            key=lambda x: x[1].fitness
        )
        
        # Remove weakest individuals
        excess = len(self) - max_size
        self.individuals = [ind for _, ind in sorted_individuals[excess:]]
        return excess
    
    def select_random_pairs(self, num_pairs: int) -> List[Tuple[int, int]]:
        """Select random pairs for crossover"""
        max_pairs = len(self) * (len(self) - 1) // 2
        
        if num_pairs >= max_pairs:
            return list(combinations(range(len(self)), 2))
        
        pairs = set()
        while len(pairs) < num_pairs:
            idx1, idx2 = random.sample(range(len(self)), 2)
            pairs.add((min(idx1, idx2), max(idx1, idx2)))
        
        return list(pairs)
    
    def select_random_individuals(self, count: int) -> List[int]:
        """Select random individuals"""
        return [random.randint(0, len(self) - 1) for _ in range(count)]


class BaseGeneticAlgorithm(ABC):
    """Base class for genetic algorithm experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.population = Population.random(config)
        self.stats_history: List[GenerationStats] = []
        self.max_fitness_history: List[float] = [self.population.max_fitness]
    
    @abstractmethod
    def create_offspring(self) -> Tuple[List[Individual], int, int, int]:
        """Create offspring for next generation. Returns (offspring, better_count, mutations, total_offspring)"""
        pass
    
    @abstractmethod
    def get_experiment_name(self) -> str:
        """Get the name of this experiment"""
        pass
    
    def should_retain_offspring(self, offspring: Individual, parents_fitness: List[float]) -> bool:
        """Determine if offspring should be retained"""
        offspring_fitness = offspring.fitness
        return self.should_retain_offspring_with_fitness(offspring_fitness, parents_fitness)
    
    def should_retain_offspring_with_fitness(self, offspring_fitness: float, parents_fitness: List[float]) -> bool:
        """Determine if offspring should be retained given pre-calculated fitness"""
        # Always retain if better than all parents
        if all(offspring_fitness > pf for pf in parents_fitness):
            return True
        
        # Otherwise retain with probability
        return random.random() < self.config.retention_probability
    
    def run_generation(self) -> GenerationStats:
        """Run a single generation"""
        offspring_list, better_count, mutations, total_offspring = self.create_offspring()
        
        # Add surviving offspring to population
        self.population.extend(offspring_list)
        
        # Limit population size
        eliminated = self.population.limit_size(self.config.population_limit)
        
        # Record statistics
        stats = GenerationStats(
            p_value=better_count / total_offspring if total_offspring > 0 else 0,
            offspring_generated=total_offspring,
            offspring_retained=len(offspring_list),
            mutations=mutations,
            eliminated=eliminated,
            max_fitness=self.population.max_fitness
        )
        
        self.stats_history.append(stats)
        self.max_fitness_history.append(stats.max_fitness)
        
        return stats
    
    def run_experiment(self) -> List[float]:
        """Run the complete experiment"""
        self._print_experiment_header()
        
        for generation in range(self.config.generations):
            stats = self.run_generation()
            
            if generation % 10 == 0 or generation == self.config.generations - 1:
                self._print_generation_stats(generation + 1, stats)
        
        print(f"{self.get_experiment_name()} experiment completed!")
        return [stats.p_value for stats in self.stats_history]
    
    def _print_experiment_header(self):
        """Print experiment header information"""
        print(f"Starting {self.get_experiment_name()} experiment...")
        print(f"Population: {self.config.population_size}, "
              f"Length: {self.config.individual_length}, "
              f"Generations: {self.config.generations}")
        print(f"Retention: {self.config.retention_probability}, "
              f"Mutation: {self.config.mutation_probability}")
        print("-" * 50)
        
        fitness_values = self.population.fitness_values
        print(f"Initial avg: {np.mean(fitness_values):.2f}, "
              f"max: {max(fitness_values)}")
        print("-" * 50)
    
    def _print_generation_stats(self, generation: int, stats: GenerationStats):
        """Print statistics for a generation"""
        retention_rate = (stats.offspring_retained / stats.offspring_generated * 100 
                         if stats.offspring_generated > 0 else 0)
        
        print(f"Gen {generation:3d}: P = {stats.p_value:.4f}, "
              f"Pop = {len(self.population):4d}, "
              f"Avg = {self.population.average_fitness:6.2f}, "
              f"Retained: {stats.offspring_retained}/{stats.offspring_generated} "
              f"({retention_rate:.1f}%)")


class GeneticAlgorithmWithCrossover(BaseGeneticAlgorithm):
    """Genetic algorithm with crossover operation"""
    
    def get_experiment_name(self) -> str:
        return "WITH CROSSOVER"
    
    def create_offspring(self) -> Tuple[List[Individual], int, int, int]:
        """Create offspring through crossover and mutation"""
        pairs = self.population.select_random_pairs(self.config.crossover_pairs)
        
        offspring_list = []
        better_count = 0
        mutations = 0
        total_offspring = 0
        
        for idx1, idx2 in pairs:
            parent1, parent2 = self.population[idx1], self.population[idx2]
            parents_fitness = [parent1.fitness, parent2.fitness]
            
            # Create offspring through crossover
            offspring = parent1.crossover_with(parent2)
            
            # Apply mutation
            if offspring.mutate(self.config.mutation_probability):
                mutations += 1
            
            # Calculate offspring fitness once
            offspring_fitness = offspring.fitness
            total_offspring += 1
            
            # Count if offspring is better than both parents (for ALL offspring)
            if offspring_fitness > max(parents_fitness):
                better_count += 1
            
            # Check if offspring should be retained
            if self.should_retain_offspring_with_fitness(offspring_fitness, parents_fitness):
                offspring_list.append(offspring)
        
        return offspring_list, better_count, mutations, total_offspring


class GeneticAlgorithmWithoutCrossover(BaseGeneticAlgorithm):
    """Genetic algorithm without crossover (mutation-only)"""
    
    def get_experiment_name(self) -> str:
        return "WITHOUT CROSSOVER"
    
    def create_offspring(self) -> Tuple[List[Individual], int, int, int]:
        """Create offspring through mutation only"""
        selected_indices = self.population.select_random_individuals(
            self.config.crossover_pairs
        )
        
        offspring_list = []
        better_count = 0
        mutations = 0
        total_offspring = 0
        
        for idx in selected_indices:
            parent = self.population[idx]
            parent_fitness = parent.fitness
            
            # Create offspring through copying and mutation
            offspring = parent.copy()
            
            # Apply mutation
            if offspring.mutate(self.config.mutation_probability):
                mutations += 1
            
            # Calculate offspring fitness once
            offspring_fitness = offspring.fitness
            total_offspring += 1
            
            # Count if offspring is better than parent (for ALL offspring)
            if offspring_fitness > parent_fitness:
                better_count += 1
            
            # Check if offspring should be retained
            if self.should_retain_offspring_with_fitness(offspring_fitness, [parent_fitness]):
                offspring_list.append(offspring)
        
        return offspring_list, better_count, mutations, total_offspring


class ExperimentVisualizer:
    """Handles visualization of experiment results"""
    
    def __init__(self, exp_with: BaseGeneticAlgorithm, exp_without: BaseGeneticAlgorithm):
        self.exp_with = exp_with
        self.exp_without = exp_without
    
    def plot_comparison(self):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        
        self._plot_p_values(axes[0])
        self._plot_fitness_evolution(axes[1])
        self._plot_retention_rates(axes[2])
        self._plot_moving_averages(axes[3])
        
        plt.tight_layout()
        self._save_plots()
        self._save_data()
        plt.show()
        
        self._print_statistics()
    
    def _plot_p_values(self, ax):
        """Plot P-value comparison"""
        generations = range(1, len(self.exp_with.stats_history) + 1)
        p_values_with = [s.p_value for s in self.exp_with.stats_history]
        p_values_without = [s.p_value for s in self.exp_without.stats_history]
        
        # Calculate mean values
        mean_with = np.mean(p_values_with)
        mean_without = np.mean(p_values_without)
        
        ax.plot(generations, p_values_with, 'r-o', markersize=3, 
                label='WITH Crossover', alpha=0.8)
        ax.plot(generations, p_values_without, 'b-s', markersize=3, 
                label='WITHOUT Crossover', alpha=0.8)
        
        # Add average lines
        ax.axhline(mean_with, color='red', linestyle='--', alpha=0.5)
        ax.axhline(mean_without, color='blue', linestyle='--', alpha=0.5)
        
        # Add mean value annotations
        ax.text(0.02, 0.95, f'WITH Crossover Mean: {mean_with:.4f}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.1),
                verticalalignment='top')
        
        ax.text(0.02, 0.88, f'WITHOUT Crossover Mean: {mean_without:.4f}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.1),
                verticalalignment='top')
        
        # Add efficiency ratio annotation
        ratio = mean_with / mean_without if mean_without > 0 else float('inf')
        ax.text(0.02, 0.81, f'Efficiency Ratio: {ratio:.1f}x', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.2),
                verticalalignment='top')
        
        ax.set_title('P Value Comparison: WITH vs WITHOUT Crossover', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('P Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_fitness_evolution(self, ax):
        """Plot fitness evolution comparison"""
        generations = range(len(self.exp_with.max_fitness_history))
        
        ax.plot(generations, self.exp_with.max_fitness_history, 'r-o', 
                markersize=3, label='WITH Crossover', alpha=0.8)
        ax.plot(generations, self.exp_without.max_fitness_history, 'b-s', 
                markersize=3, label='WITHOUT Crossover', alpha=0.8)
        
        # Add final fitness value annotations
        final_gen = len(self.exp_with.max_fitness_history) - 1
        final_fitness_with = self.exp_with.max_fitness_history[-1]
        final_fitness_without = self.exp_without.max_fitness_history[-1]
        
        # Annotate final fitness values at the end of curves
        ax.annotate(f'{final_fitness_with:.0f}', 
                   xy=(final_gen, final_fitness_with), 
                   xytext=(10, 10), textcoords='offset points',
                   ha='left', va='bottom', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.2),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'{final_fitness_without:.0f}', 
                   xy=(final_gen, final_fitness_without), 
                   xytext=(10, -15), textcoords='offset points',
                   ha='left', va='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.2),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_title('Maximum Fitness Evolution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Maximum Fitness')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_retention_rates(self, ax):
        """Plot retention rate comparison"""
        generations = range(1, len(self.exp_with.stats_history) + 1)
        
        rates_with = [s.offspring_retained / s.offspring_generated * 100 
                     if s.offspring_generated > 0 else 0 
                     for s in self.exp_with.stats_history]
        rates_without = [s.offspring_retained / s.offspring_generated * 100 
                        if s.offspring_generated > 0 else 0 
                        for s in self.exp_without.stats_history]
        
        ax.plot(generations, rates_with, 'r-o', markersize=3, 
                label='WITH Crossover', alpha=0.8)
        ax.plot(generations, rates_without, 'b-s', markersize=3, 
                label='WITHOUT Crossover', alpha=0.8)
        
        ax.set_title('Offspring Retention Rate', fontsize=16, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Retention Rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_moving_averages(self, ax):
        """Plot moving average trends"""
        window_size = 10
        generations = range(1, len(self.exp_with.stats_history) + 1)
        p_values_with = [s.p_value for s in self.exp_with.stats_history]
        p_values_without = [s.p_value for s in self.exp_without.stats_history]
        
        if len(p_values_with) >= window_size:
            moving_avg_with = np.convolve(p_values_with, 
                                        np.ones(window_size)/window_size, mode='valid')
            moving_avg_without = np.convolve(p_values_without, 
                                           np.ones(window_size)/window_size, mode='valid')
            moving_generations = list(generations)[window_size-1:]
            
            ax.plot(moving_generations, moving_avg_with, 'r-', linewidth=3, 
                   label=f'WITH Crossover ({window_size}-Gen Moving Avg)')
            ax.plot(moving_generations, moving_avg_without, 'b-', linewidth=3, 
                   label=f'WITHOUT Crossover ({window_size}-Gen Moving Avg)')
        
        # Background lines
        ax.plot(generations, p_values_with, alpha=0.3, color='red')
        ax.plot(generations, p_values_without, alpha=0.3, color='blue')
        
        ax.set_title('P Value Moving Average Trends', fontsize=16, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('P Value (Moving Average)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _save_plots(self):
        """Save plots in multiple formats"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"genetic_algorithm_comparison_{timestamp}"
        
        for fmt in ['pdf', 'svg', 'png']:
            plt.savefig(f"{filename_base}.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filename_base}.{fmt}")
    
    def _save_data(self):
        """Save experiment data"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"genetic_algorithm_comparison_{timestamp}_raw_data.npz"
        
        data = {
            'P_values_with_crossover': np.array([s.p_value for s in self.exp_with.stats_history]),
            'P_values_without_crossover': np.array([s.p_value for s in self.exp_without.stats_history]),
            'max_fitness_with_crossover': np.array(self.exp_with.max_fitness_history),
            'max_fitness_without_crossover': np.array(self.exp_without.max_fitness_history),
            'experiment_parameters': {
                'population_size': self.exp_with.config.population_size,
                'individual_length': self.exp_with.config.individual_length,
                'generations': self.exp_with.config.generations,
                'retention_probability': self.exp_with.config.retention_probability,
                'mutation_probability': self.exp_with.config.mutation_probability,
                'population_limit': self.exp_with.config.population_limit,
                'value_range': self.exp_with.config.value_range
            }
        }
        
        np.savez_compressed(filename, **data)
        print(f"Saved data: {filename}")
    
    def _print_statistics(self):
        """Print comprehensive comparison statistics"""
        p_values_with = [s.p_value for s in self.exp_with.stats_history]
        p_values_without = [s.p_value for s in self.exp_without.stats_history]
        
        print(f"\n" + "="*80)
        print("COMPREHENSIVE COMPARISON RESULTS")
        print("="*80)
        
        print(f"\n=== P VALUE STATISTICS ===")
        print(f"WITH Crossover    - Average: {np.mean(p_values_with):.4f}")
        print(f"WITHOUT Crossover - Average: {np.mean(p_values_without):.4f}")
        
        ratio = (np.mean(p_values_with) / np.mean(p_values_without) 
                if np.mean(p_values_without) > 0 else float('inf'))
        print(f"P Value Efficiency Ratio: {ratio:.1f}x")
        
        print(f"\n=== FITNESS IMPROVEMENT ===")
        with_improvement = (self.exp_with.max_fitness_history[-1] - 
                          self.exp_with.max_fitness_history[0])
        without_improvement = (self.exp_without.max_fitness_history[-1] - 
                             self.exp_without.max_fitness_history[0])
        
        print(f"WITH Crossover improvement: {with_improvement}")
        print(f"WITHOUT Crossover improvement: {without_improvement}")
        
        fitness_ratio = (with_improvement / without_improvement 
                        if without_improvement > 0 else float('inf'))
        print(f"Fitness Improvement Ratio: {fitness_ratio:.1f}x")


def run_comparison_experiment():
    """Run the complete comparison experiment"""
    config = ExperimentConfig()
    
    print("=" * 80)
    print("GENETIC ALGORITHM COMPARISON: WITH vs WITHOUT CROSSOVER")
    print("=" * 80)
    
    # Run experiments
    print("\n" + "="*50)
    print("EXPERIMENT 1: WITH CROSSOVER")
    print("="*50)
    exp_with = GeneticAlgorithmWithCrossover(config)
    exp_with.run_experiment()
    
    print("\n" + "="*50)
    print("EXPERIMENT 2: WITHOUT CROSSOVER")
    print("="*50)
    exp_without = GeneticAlgorithmWithoutCrossover(config)
    exp_without.run_experiment()
    
    # Visualize results
    visualizer = ExperimentVisualizer(exp_with, exp_without)
    visualizer.plot_comparison()
    
    return exp_with, exp_without


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed()
    random.seed()
    
    exp_with, exp_without = run_comparison_experiment()
