"""
Genetic Algorithm for CVRP
"""

import numpy as np
import random
import time
import threading


class GeneticAlgorithm_CVRP:
    """Genetic Algorithm for Capacitated Vehicle Routing Problem"""

    def __init__(self, cvrp, population_size=50, mutation_rate=0.1, crossover_rate=0.8, elitism=5, max_generations=100):
        """
        Initialize Genetic Algorithm for CVRP

        Parameters:
        cvrp -- CVRP object
        population_size -- Population size
        mutation_rate -- Mutation probability
        crossover_rate -- Crossover probability
        elitism -- Number of elite individuals to keep
        max_generations -- Maximum number of generations
        """
        self.cvrp = cvrp
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.max_generations = max_generations

        # Number of customers
        self.n = len(cvrp.customers)

        # Save results
        self.best_solution = None
        self.best_cost = float('inf')

        # Status for visualization
        self.current_solution = None
        self.current_cost = float('inf')

        # Cost history
        self.cost_history = []

        # Stop flag
        self.stop_flag = False

    def run(self, callback=None, step_callback=None):
        """
        Run the Genetic Algorithm

        Parameters:
        callback -- Function to call when finished
        step_callback -- Function to call after each generation
        """
        self.stop_flag = False
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []

        # Initialize population
        population = self.initialize_population()

        for generation in range(self.max_generations):
            if self.stop_flag:
                break

            # Evaluate population
            fitness_values = [self.evaluate_fitness(individual) for individual in population]

            # Find the best individual in current population
            best_idx = np.argmin(fitness_values)
            current_best_solution = self.decode_chromosome(population[best_idx])
            current_best_cost = fitness_values[best_idx]

            # Update best solution
            if current_best_cost < self.best_cost:
                self.best_solution = current_best_solution
                self.best_cost = current_best_cost

            # Save current best cost
            self.cost_history.append(self.best_cost)

            # Step callback
            if step_callback:
                self.current_solution = current_best_solution
                self.current_cost = current_best_cost

                step_data = {
                    'generation': generation,
                    'progress': (generation + 1) / self.max_generations,
                    'solution': self.current_solution,
                    'cost': self.current_cost,
                    'best_solution': self.best_solution,
                    'best_cost': self.best_cost,
                    'population': population.copy(),
                    'fitness_values': fitness_values.copy(),
                    'cost_history': self.cost_history.copy()
                }
                step_callback(step_data)

            # Create new generation
            new_population = []

            # Elitism: keep the best individuals
            sorted_indices = np.argsort(fitness_values)
            for i in range(self.elitism):
                new_population.append(population[sorted_indices[i]])

            # Create new individuals through crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.selection(population, fitness_values)
                parent2 = self.selection(population, fitness_values)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if random.random() < self.mutation_rate:
                    self.mutation(child1)
                if random.random() < self.mutation_rate:
                    self.mutation(child2)

                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            # Update population
            population = new_population

        # Callback when finished
        if callback:
            callback((self.best_solution, self.best_cost))

    def initialize_population(self):
        """
        Initialize initial population

        Returns:
        List of chromosomes
        """
        population = []

        for _ in range(self.population_size):
            # Create a random permutation of customers
            chromosome = list(range(1, self.n))
            random.shuffle(chromosome)
            population.append(chromosome)

        return population

    def decode_chromosome(self, chromosome):
        """
        Decode chromosome to CVRP solution

        Parameters:
        chromosome -- Chromosome (permutation of customers)

        Returns:
        List of routes
        """
        solution = []
        route = []
        capacity_left = self.cvrp.capacity

        for customer in chromosome:
            demand = self.cvrp.customers[customer].demand

            # If not enough capacity, start a new route
            if demand > capacity_left:
                if route:  # Only add route if not empty
                    solution.append(route)
                route = [customer]
                capacity_left = self.cvrp.capacity - demand
            else:
                route.append(customer)
                capacity_left -= demand

        # Add last route if exists
        if route:
            solution.append(route)

        return solution

    def evaluate_fitness(self, chromosome):
        """
        Evaluate fitness of a chromosome

        Parameters:
        chromosome -- Chromosome to evaluate

        Returns:
        Fitness value (cost, lower is better)
        """
        solution = self.decode_chromosome(chromosome)
        cost = self.cvrp.calculate_solution_cost(solution)
        return cost

    def selection(self, population, fitness_values):
        """
        Select an individual from the population using tournament selection

        Parameters:
        population -- Population
        fitness_values -- Corresponding fitness values

        Returns:
        Selected individual
        """
        # Tournament selection
        k = 3  # Tournament size
        selected_indices = random.sample(range(len(population)), k)
        tournament_fitness = [fitness_values[i] for i in selected_indices]

        # Select the best individual from tournament
        best_idx = selected_indices[np.argmin(tournament_fitness)]
        return population[best_idx].copy()

    def crossover(self, parent1, parent2):
        """
        Crossover two parent chromosomes

        Parameters:
        parent1, parent2 -- Two parent chromosomes

        Returns:
        Two child chromosomes
        """
        # Ordered Crossover (OX)
        size = len(parent1)

        # Select two crossover points
        point1 = random.randint(0, size - 2)
        point2 = random.randint(point1 + 1, size - 1)

        # Create mask for segment between two points
        mask = [False] * size
        for i in range(point1, point2 + 1):
            mask[i] = True

        # Create two children
        child1 = [-1] * size
        child2 = [-1] * size

        # Copy segment between two points
        for i in range(point1, point2 + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]

        # Fill the rest
        self.fill_ox(parent2, child1, mask)
        self.fill_ox(parent1, child2, mask)

        return child1, child2

    def fill_ox(self, parent, child, mask):
        """
        Fill the rest of child chromosome with values from parent

        Parameters:
        parent -- Parent chromosome
        child -- Child chromosome (partially filled)
        mask -- Mask indicating which positions are filled
        """
        size = len(parent)
        j = 0  # Index in parent

        for i in range(size):
            if not mask[i]:  # If position is not filled
                # Find next value from parent not in child
                while parent[j] in child:
                    j = (j + 1) % size

                child[i] = parent[j]
                j = (j + 1) % size

    def mutation(self, chromosome):
        """
        Mutate a chromosome

        Parameters:
        chromosome -- Chromosome to mutate
        """
        # Swap mutation
        size = len(chromosome)

        # Select two random positions to swap
        idx1 = random.randint(0, size - 1)
        idx2 = random.randint(0, size - 1)

        # Swap
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    def stop(self):
        """Stop the algorithm"""
        self.stop_flag = True