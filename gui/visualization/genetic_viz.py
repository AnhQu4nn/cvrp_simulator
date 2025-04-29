"""
Genetic Algorithm Visualization for CVRP
"""

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from .base import CVRPVisualization


class GeneticVisualization(CVRPVisualization):
    """Visualization class for Genetic Algorithm"""

    def __init__(self, parent_frame):
        """Initialize visualization for Genetic Algorithm"""
        super().__init__(parent_frame)

    def setup_axes(self):
        """Setup axes for Genetic visualization"""
        self.axs = self.fig.subplots(2, 2)
        self.fig.suptitle('Solving CVRP with Genetic Algorithm', fontsize=16)

        # Axis for current best solution
        self.axs[0, 0].set_title('Current Best Solution')
        self.axs[0, 0].set_xlabel('X')
        self.axs[0, 0].set_ylabel('Y')

        # Axis for global best solution
        self.axs[0, 1].set_title('Global Best Solution')
        self.axs[0, 1].set_xlabel('X')
        self.axs[0, 1].set_ylabel('Y')

        # Axis for fitness distribution
        self.axs[1, 0].set_title('Fitness Distribution')
        self.axs[1, 0].set_xlabel('Individual')
        self.axs[1, 0].set_ylabel('Fitness (cost)')

        # Axis for progress chart
        self.axs[1, 1].set_title('Optimization Progress')
        self.axs[1, 1].set_xlabel('Generation')
        self.axs[1, 1].set_ylabel('Best Cost')

        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Setup status text
        self.status_text = self.fig.text(0.01, 0.01, '', fontsize=8)

    def init_visualization(self, cvrp, population_size, max_generations):
        """Initialize visualization with CVRP data and parameters"""
        # Clear axes
        for ax in self.axs.flatten():
            ax.clear()

        # Reset titles
        self.axs[0, 0].set_title('Current Best Solution')
        self.axs[0, 0].set_xlabel('X')
        self.axs[0, 0].set_ylabel('Y')

        self.axs[0, 1].set_title('Global Best Solution')
        self.axs[0, 1].set_xlabel('X')
        self.axs[0, 1].set_ylabel('Y')

        self.axs[1, 0].set_title('Fitness Distribution')
        self.axs[1, 0].set_xlabel('Individual')
        self.axs[1, 0].set_ylabel('Fitness (cost)')

        self.axs[1, 1].set_title('Optimization Progress')
        self.axs[1, 1].set_xlabel('Generation')
        self.axs[1, 1].set_ylabel('Best Cost')

        # Save data
        self.cvrp = cvrp
        self.population_size = population_size
        self.max_generations = max_generations

        # Plot customers and depot
        self.plot_customers(self.axs[0, 0])
        self.plot_customers(self.axs[0, 1])

        # Initialize fitness chart
        self.fitness_bars = self.axs[1, 0].bar(range(population_size),
                                               [0] * population_size,
                                               color='skyblue')

        # Initialize progress chart
        self.progress_line, = self.axs[1, 1].plot([], [], 'r-')

        # Routes for visualizations
        self.current_routes = []
        self.best_routes = []

        self.canvas.draw()

    def update_visualization(self, data):
        """Update visualization with data from algorithm"""
        generation = data['generation']
        solution = data['solution']
        cost = data['cost']
        best_solution = data['best_solution']
        best_cost = data['best_cost']
        population = data['population']
        fitness_values = data['fitness_values']
        cost_history = data['cost_history']

        # Update current solution
        # Remove old routes
        for route in self.current_routes:
            route.remove()
        self.current_routes = []

        # Draw new routes
        colors = plt.cm.tab10(np.linspace(0, 1, len(solution)))
        for i, route in enumerate(solution):
            if not route:
                continue

            # Create point list (add depot at start and end)
            x = [self.cvrp.depot.x]
            y = [self.cvrp.depot.y]

            for node in route:
                x.append(self.cvrp.customers[node].x)
                y.append(self.cvrp.customers[node].y)

            # Return to depot
            x.append(self.cvrp.depot.x)
            y.append(self.cvrp.depot.y)

            # Draw route
            line, = self.axs[0, 0].plot(x, y, 'o-', c=colors[i % len(colors)],
                                        linewidth=1.5, markersize=5,
                                        label=f'Route {i + 1}')
            self.current_routes.append(line)

        # Add legend
        self.axs[0, 0].legend()

        # Update best solution
        # Remove old routes
        for route in self.best_routes:
            route.remove()
        self.best_routes = []

        # Draw best routes
        if best_solution:
            colors = plt.cm.tab10(np.linspace(0, 1, len(best_solution)))
            for i, route in enumerate(best_solution):
                if not route:
                    continue

                # Create point list (add depot at start and end)
                x = [self.cvrp.depot.x]
                y = [self.cvrp.depot.y]

                for node in route:
                    x.append(self.cvrp.customers[node].x)
                    y.append(self.cvrp.customers[node].y)

                # Return to depot
                x.append(self.cvrp.depot.x)
                y.append(self.cvrp.depot.y)

                # Draw route
                line, = self.axs[0, 1].plot(x, y, 'o-', c=colors[i % len(colors)],
                                            linewidth=1.5, markersize=5,
                                            label=f'Route {i + 1}')
                self.best_routes.append(line)

            # Add legend
            self.axs[0, 1].legend()

        # Update fitness chart
        sorted_fitness = sorted(fitness_values)[:min(len(fitness_values), self.population_size)]
        for i, fitness in enumerate(sorted_fitness):
            self.fitness_bars[i].set_height(fitness)

        self.axs[1, 0].set_ylim(0, max(sorted_fitness) * 1.1 if sorted_fitness else 1)

        # Update progress chart
        generations = list(range(len(cost_history)))
        self.progress_line.set_data(generations, cost_history)

        if cost_history:
            self.axs[1, 1].set_xlim(0, max(len(cost_history), 1))
            if max(cost_history) > min(cost_history):
                self.axs[1, 1].set_ylim(min(cost_history) * 0.95, max(cost_history) * 1.05)

        # Update status text
        self.status_text.set_text(
            f'Generation: {generation + 1}/{self.max_generations} | Current cost: {cost:.2f} | Best cost: {best_cost:.2f}'
        )

        self.canvas.draw()