"""
Ant Colony Optimization Algorithm for CVRP
"""

import numpy as np
import random
import time
import threading


class ACO_CVRP:
    """Ant Colony Optimization for Capacitated Vehicle Routing Problem"""

    def __init__(self, cvrp, num_ants=10, alpha=1.0, beta=2.0, rho=0.5, q=100, max_iterations=100):
        """
        Initialize ACO algorithm for CVRP

        Parameters:
        cvrp -- CVRP object
        num_ants -- Number of ants
        alpha -- Pheromone importance
        beta -- Heuristic importance
        rho -- Pheromone evaporation rate
        q -- Pheromone deposit factor
        max_iterations -- Maximum number of iterations
        """
        self.cvrp = cvrp
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_iterations = max_iterations

        # Number of customers
        self.n = len(cvrp.customers)

        # Initialize pheromone matrix
        self.pheromone = np.ones((self.n, self.n))

        # Initialize heuristic matrix (inverse of distance)
        self.heuristic = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j and cvrp.distances[i, j] > 0:
                    self.heuristic[i, j] = 1.0 / cvrp.distances[i, j]

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
        Run the ACO algorithm

        Parameters:
        callback -- Function to call when finished
        step_callback -- Function to call after each iteration
        """
        self.stop_flag = False
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []

        for iteration in range(self.max_iterations):
            if self.stop_flag:
                break

            # Initialize solutions for each ant
            ant_solutions = []
            ant_costs = []

            for ant in range(self.num_ants):
                solution = self.construct_solution()
                cost = self.cvrp.calculate_solution_cost(solution)

                ant_solutions.append(solution)
                ant_costs.append(cost)

                # Update best solution
                if cost < self.best_cost:
                    self.best_solution = solution
                    self.best_cost = cost

            # Update pheromone
            self.update_pheromone(ant_solutions, ant_costs)

            # Save current best cost
            self.cost_history.append(self.best_cost)

            # Step callback
            if step_callback:
                iteration_best_idx = np.argmin(ant_costs)
                self.current_solution = ant_solutions[iteration_best_idx]
                self.current_cost = ant_costs[iteration_best_idx]

                step_data = {
                    'iteration': iteration,
                    'progress': (iteration + 1) / self.max_iterations,
                    'solution': self.current_solution,
                    'cost': self.current_cost,
                    'best_solution': self.best_solution,
                    'best_cost': self.best_cost,
                    'pheromone': self.pheromone.copy(),
                    'cost_history': self.cost_history.copy()
                }
                step_callback(step_data)

        # Callback when finished
        if callback:
            callback((self.best_solution, self.best_cost))

    def construct_solution(self):
        """
        Construct a solution using an ant

        Returns:
        List of routes (each route is a list of customers)
        """
        solution = []
        remaining = list(range(1, self.n))  # List of unvisited customers (skip depot 0)

        while remaining:
            # Start a new route from the depot
            route = []
            current_capacity = 0
            current_node = 0  # Depot

            while True:
                # Find next possible customers to visit
                candidates = []
                for node in remaining:
                    if current_capacity + self.cvrp.customers[node].demand <= self.cvrp.capacity:
                        candidates.append(node)

                if not candidates:
                    break  # No more customers can be added to current route

                # Select next customer based on ACO selection rule
                next_node = self.select_next_node(current_node, candidates)
                route.append(next_node)
                remaining.remove(next_node)

                # Update current capacity
                current_capacity += self.cvrp.customers[next_node].demand
                current_node = next_node

            if route:  # If route is not empty, add to solution
                solution.append(route)

        return solution

    def select_next_node(self, current, candidates):
        """
        Select next customer based on pheromone and heuristic

        Parameters:
        current -- Current customer
        candidates -- List of possible next customers

        Returns:
        Selected next customer
        """
        if not candidates:
            return None

        # Calculate probability for each candidate
        probabilities = np.zeros(len(candidates))

        for i, candidate in enumerate(candidates):
            pheromone = self.pheromone[current, candidate] ** self.alpha
            heuristic_value = self.heuristic[current, candidate] ** self.beta
            probabilities[i] = pheromone * heuristic_value

        # Normalize probabilities
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
        else:
            probabilities = np.ones(len(candidates)) / len(candidates)

        # Select next customer based on probability
        selected = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected]

    def update_pheromone(self, solutions, costs):
        """
        Update pheromone matrix

        Parameters:
        solutions -- List of solutions
        costs -- List of corresponding costs
        """
        # Evaporate pheromone
        self.pheromone = (1 - self.rho) * self.pheromone

        # Add new pheromone
        for solution, cost in zip(solutions, costs):
            delta = self.q / cost if cost > 0 else 0

            for route in solution:
                prev_node = 0  # Start from depot

                for node in route:
                    self.pheromone[prev_node, node] += delta
                    self.pheromone[node, prev_node] += delta  # Undirected graph
                    prev_node = node

                # Return to depot
                self.pheromone[prev_node, 0] += delta
                self.pheromone[0, prev_node] += delta

    def stop(self):
        """Stop the algorithm"""
        self.stop_flag = True