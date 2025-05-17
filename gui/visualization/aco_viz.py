"""
ACO Visualization for CVRP
"""

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from .base import CVRPVisualization


class ACOVisualization(CVRPVisualization):
    """Visualization class for ACO algorithm"""

    def __init__(self, parent_frame):
        """Initialize visualization for ACO algorithm"""
        super().__init__(parent_frame)
        
        # Add export button to toolbar
        self.export_button = ttk.Button(self.toolbar_frame, text="Xuất kết quả", command=self.export_aco_results)
        self.export_button.pack(side=tk.RIGHT, padx=5)

    def setup_axes(self):
        """Setup axes for ACO"""
        self.axs = self.fig.subplots(2, 2)
        self.fig.suptitle('Solving CVRP with Ant Colony Optimization', fontsize=16)

        # Axis for current solution
        self.axs[0, 0].set_title('Current Solution')
        self.axs[0, 0].set_xlabel('X')
        self.axs[0, 0].set_ylabel('Y')

        # Axis for best solution
        self.axs[0, 1].set_title('Best Solution')
        self.axs[0, 1].set_xlabel('X')
        self.axs[0, 1].set_ylabel('Y')

        # Axis for pheromone
        self.axs[1, 0].set_title('Pheromone Map')
        self.axs[1, 0].set_xlabel('Customer')
        self.axs[1, 0].set_ylabel('Customer')

        # Axis for progress chart
        self.axs[1, 1].set_title('Optimization Progress')
        self.axs[1, 1].set_xlabel('Iteration')
        self.axs[1, 1].set_ylabel('Best Cost')

        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Setup status text
        self.status_text = self.fig.text(0.01, 0.01, '', fontsize=8)

    def init_visualization(self, cvrp, num_ants, iterations):
        """Initialize visualization with CVRP data and parameters"""
        # Clear axes
        for ax in self.axs.flatten():
            ax.clear()

        # Reset titles
        self.axs[0, 0].set_title('Current Solution')
        self.axs[0, 0].set_xlabel('X')
        self.axs[0, 0].set_ylabel('Y')

        self.axs[0, 1].set_title('Best Solution')
        self.axs[0, 1].set_xlabel('X')
        self.axs[0, 1].set_ylabel('Y')

        self.axs[1, 0].set_title('Pheromone Map')
        self.axs[1, 0].set_xlabel('Customer')
        self.axs[1, 0].set_ylabel('Customer')

        self.axs[1, 1].set_title('Optimization Progress')
        self.axs[1, 1].set_xlabel('Iteration')
        self.axs[1, 1].set_ylabel('Best Cost')

        # Save data
        self.cvrp = cvrp
        self.num_ants = num_ants
        self.iterations = iterations
        
        # Reset cost history
        self.cost_history = []

        # Plot customers and depot
        self.plot_customers(self.axs[0, 0])
        self.plot_customers(self.axs[0, 1])

        # Initialize pheromone heatmap
        n = len(cvrp.customers)
        self.pheromone_heatmap = self.axs[1, 0].imshow(
            np.ones((n, n)),
            cmap='viridis',
            interpolation='nearest'
        )
        self.pheromone_colorbar = self.fig.colorbar(self.pheromone_heatmap, ax=self.axs[1, 0], label='Pheromone')

        # Initialize progress chart
        self.progress_line, = self.axs[1, 1].plot([], [], 'r-')
        self.axs[1, 1].set_xlim(0, 10)
        self.axs[1, 1].set_ylim(0, 1000)
        self.axs[1, 1].grid(True, linestyle='--', alpha=0.7)

        # Routes for visualizations
        self.current_routes = []
        self.best_routes = []

        self.canvas.draw()

    def update_visualization(self, data):
        """Update visualization with data from algorithm"""
        iteration = data['iteration']
        solution = data['solution']
        cost = data['cost']
        best_solution = data['best_solution']
        best_cost = data['best_cost']
        pheromone = data['pheromone']
        cost_history = data['cost_history']
        
        # Save data for export
        self.best_solution = best_solution
        self.best_cost = best_cost
        self.cost_history = cost_history

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

        # Update pheromone heatmap
        norm_pheromone = pheromone / pheromone.max() if pheromone.max() > 0 else pheromone
        self.pheromone_heatmap.set_array(norm_pheromone)

        # Update progress chart
        iterations = list(range(len(cost_history)))
        self.progress_line.set_data(iterations, cost_history)

        if cost_history:
            self.axs[1, 1].set_xlim(0, max(len(cost_history), 1))
            if max(cost_history) > min(cost_history):
                self.axs[1, 1].set_ylim(min(cost_history) * 0.95, max(cost_history) * 1.05)

        # Update status text
        self.status_text.set_text(
            f'Iteration: {iteration + 1}/{self.iterations} | Current cost: {cost:.2f} | Best cost: {best_cost:.2f}'
        )

        self.canvas.draw()

    def update(self, cvrp, solution, cost, pheromone=None):
        """
        Update method that matches the signature in comparison_app.py
        
        Tham số:
        cvrp -- Đối tượng CVRP
        solution -- Giải pháp hiện tại
        cost -- Chi phí hiện tại
        pheromone -- Ma trận pheromone (tùy chọn)
        """
        # Save data for export
        self.best_solution = solution
        self.best_cost = cost
        
        # Ensure we have the CVRP object
        self.cvrp = cvrp
        
        # Draw solution
        for route in self.current_routes:
            route.remove()
        self.current_routes = []
        
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
        
        # Also update best solution (both plots will show the same solution in comparison mode)
        for route in self.best_routes:
            route.remove()
        self.best_routes = []
        
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
            line, = self.axs[0, 1].plot(x, y, 'o-', c=colors[i % len(colors)],
                                       linewidth=1.5, markersize=5,
                                       label=f'Route {i + 1}')
            self.best_routes.append(line)
        
        # Update status
        self.status_text.set_text(f'Current cost: {cost:.2f}')
        
        # Update pheromone if provided
        if pheromone is not None:
            norm_pheromone = pheromone / pheromone.max() if pheromone.max() > 0 else pheromone
            self.pheromone_heatmap.set_array(norm_pheromone)
        
        # Update progress chart if we have previous cost history
        if hasattr(self, 'cost_history'):
            # Add current cost to history
            self.cost_history.append(cost)
            
            # Update the progress line
            iterations = list(range(len(self.cost_history)))
            self.progress_line.set_data(iterations, self.cost_history)
            
            # Update axis limits
            self.axs[1, 1].set_xlim(0, max(len(self.cost_history), 1))
            if len(self.cost_history) > 1:
                min_cost = min(self.cost_history)
                max_cost = max(self.cost_history)
                padding = (max_cost - min_cost) * 0.1 if max_cost > min_cost else max_cost * 0.1
                self.axs[1, 1].set_ylim(max(0, min_cost - padding), max_cost + padding)
            
            # Add grid for readability
            self.axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        else:
            # Initialize cost history with current cost
            self.cost_history = [cost]
            
            # Set up the progress line
            self.progress_line.set_data([0], [cost])
            self.axs[1, 1].set_xlim(0, 10)
            self.axs[1, 1].set_ylim(0, cost * 1.5)
            self.axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        self.canvas.draw()

    def export_aco_results(self):
        """Export ACO results"""
        self.export_results("ACO")
        
    def write_algorithm_params(self, file):
        """Write ACO parameters to file"""
        file.write("THAM SỐ ACO\n")
        file.write("-" * 20 + "\n")
        
        if hasattr(self, 'algorithm') and self.algorithm:
            file.write(f"Số lượng kiến: {self.algorithm.num_ants}\n")
            file.write(f"Alpha (trọng số pheromone): {self.algorithm.alpha}\n")
            file.write(f"Beta (trọng số heuristic): {self.algorithm.beta}\n")
            file.write(f"Rho (tỷ lệ bay hơi): {self.algorithm.rho}\n")
            file.write(f"Q (hệ số pheromone): {self.algorithm.q}\n")
            file.write(f"Số vòng lặp tối đa: {self.algorithm.max_iterations}\n")
            
            if hasattr(self.algorithm, 'min_max_aco'):
                file.write(f"Min-Max ACO: {self.algorithm.min_max_aco}\n")
            
            if hasattr(self.algorithm, 'local_search'):
                file.write(f"Tìm kiếm cục bộ: {self.algorithm.local_search}\n")
                
            if hasattr(self.algorithm, 'elitist_ants'):
                file.write(f"Số kiến ưu tú: {self.algorithm.elitist_ants}\n")
        else:
            file.write("Không có thông tin tham số (thuật toán chưa được khởi tạo)\n")