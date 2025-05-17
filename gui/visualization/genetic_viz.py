"""
Genetic Algorithm Visualization for CVRP
"""

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from .base import CVRPVisualization
import random


class GeneticVisualization(CVRPVisualization):
    """Visualization class for Genetic Algorithm"""

    def __init__(self, parent_frame):
        """Initialize visualization for Genetic Algorithm"""
        super().__init__(parent_frame)
        
        # Add export button to toolbar
        self.export_button = ttk.Button(self.toolbar_frame, text="Xuất kết quả", command=self.export_ga_results)
        self.export_button.pack(side=tk.RIGHT, padx=5)

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

    def export_ga_results(self):
        """Export GA results"""
        self.export_results("GA")
        
    def write_algorithm_params(self, file):
        """Write GA parameters to file"""
        file.write("THAM SỐ DI TRUYỀN\n")
        file.write("-" * 20 + "\n")
        
        if hasattr(self, 'algorithm') and self.algorithm:
            file.write(f"Kích thước quần thể: {self.algorithm.population_size}\n")
            file.write(f"Tỷ lệ đột biến: {self.algorithm.mutation_rate}\n")
            file.write(f"Tỷ lệ lai ghép: {self.algorithm.crossover_rate}\n")
            file.write(f"Số cá thể ưu tú: {self.algorithm.elitism}\n")
            file.write(f"Số thế hệ tối đa: {self.algorithm.max_generations}\n")
            file.write(f"Phương pháp chọn lọc: {self.algorithm.selection_method}\n")
            file.write(f"Phương pháp lai ghép: {self.algorithm.crossover_method}\n")
            file.write(f"Phương pháp đột biến: {self.algorithm.mutation_method}\n")
            
            if hasattr(self.algorithm, 'tournament_size'):
                file.write(f"Kích thước giải đấu: {self.algorithm.tournament_size}\n")
                
            if hasattr(self.algorithm, 'early_stopping'):
                file.write(f"Dừng sớm: {self.algorithm.early_stopping if self.algorithm.early_stopping else 'Không'}\n")
                
            if hasattr(self.algorithm, 'local_search'):
                file.write(f"Tìm kiếm cục bộ: {self.algorithm.local_search}\n")
        else:
            file.write("Không có thông tin tham số (thuật toán chưa được khởi tạo)\n")

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
        
        # Reset cost history
        self.cost_history = []

        # Plot customers and depot
        self.plot_customers(self.axs[0, 0])
        self.plot_customers(self.axs[0, 1])

        # Initialize fitness chart
        self.fitness_bars = self.axs[1, 0].bar(range(population_size),
                                               [0] * population_size,
                                               color='skyblue')

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
        generation = data['generation']
        solution = data['solution']
        cost = data['cost']
        best_solution = data['best_solution']
        best_cost = data['best_cost']
        population = data['population']
        fitness_values = data['fitness_values']
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

    def update(self, cvrp, solution, cost, population=None):
        """
        Update method that matches the signature in comparison_app.py
        
        Tham số:
        cvrp -- Đối tượng CVRP
        solution -- Giải pháp hiện tại
        cost -- Chi phí hiện tại
        population -- Quần thể hiện tại (tùy chọn)
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
        
        # Update population fitness chart if provided
        if population is not None:
            # Tạo fitness_bars nếu chưa có
            if not hasattr(self, 'fitness_bars'):
                # Khởi tạo biểu đồ phân phối fitness
                self.axs[1, 0].clear()
                self.axs[1, 0].set_title('Fitness Distribution')
                self.axs[1, 0].set_xlabel('Individual')
                self.axs[1, 0].set_ylabel('Fitness (cost)')
                
                # Tạo các thanh biểu đồ
                pop_size = min(len(population), 50)  # Giới hạn hiển thị 50 cá thể
                self.fitness_bars = self.axs[1, 0].bar(range(pop_size), 
                                                      [0] * pop_size,
                                                      color='skyblue')
            
            # Tạo fitness values giả lập từ population
            if isinstance(population, list) and len(population) > 0:
                # Tính toán fitness values (chi phí) cho mỗi cá thể trong quần thể
                fitness_values = []
                for indiv in population[:len(self.fitness_bars)]:
                    if indiv:
                        # Giải mã cá thể thành solution
                        indiv_solution = self.decode_chromosomes_to_routes(indiv)
                        # Tính chi phí
                        indiv_cost = self.cvrp.calculate_solution_cost(indiv_solution)
                        fitness_values.append(indiv_cost)
                
                # Sắp xếp và hiển thị
                if fitness_values:
                    sorted_fitness = sorted(fitness_values)
                    max_fitness = max(sorted_fitness)
                    
                    # Cập nhật chiều cao của các thanh
                    for i, fitness in enumerate(sorted_fitness[:len(self.fitness_bars)]):
                        self.fitness_bars[i].set_height(fitness)
                    
                    # Cập nhật giới hạn trục y
                    self.axs[1, 0].set_ylim(0, max_fitness * 1.1)
                    self.axs[1, 0].set_title(f'Fitness Distribution (Avg: {np.mean(sorted_fitness):.2f})')
                else:
                    # Nếu không có dữ liệu fitness, đặt giá trị mặc định
                    for i in range(len(self.fitness_bars)):
                        self.fitness_bars[i].set_height(cost * (0.9 + 0.2 * random.random()))
                    self.axs[1, 0].set_ylim(0, cost * 1.5)
            else:
                # Nếu không có dữ liệu quần thể hợp lệ, sử dụng giá trị mặc định
                for i in range(len(self.fitness_bars)):
                    self.fitness_bars[i].set_height(cost * (0.9 + 0.2 * random.random()))
                self.axs[1, 0].set_ylim(0, cost * 1.5)
        
        # Update progress chart if we have previous cost history
        if hasattr(self, 'cost_history'):
            # Add current cost to history
            self.cost_history.append(cost)
            
            # Update the progress line
            generations = list(range(len(self.cost_history)))
            self.progress_line.set_data(generations, self.cost_history)
            
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

    def decode_chromosomes_to_routes(self, chromosome):
        """Giải mã chuỗi nhiễm sắc thể thành các tuyến đường
        
        Tham số:
        chromosome -- Chuỗi nhiễm sắc thể (danh sách các đỉnh)
        
        Trả về:
        Giải pháp (danh sách các tuyến đường)
        """
        if not chromosome or not self.cvrp:
            return []
            
        solution = []
        current_route = []
        current_load = 0
        capacity = self.cvrp.capacity
        
        for gene in chromosome:
            if gene == 0:  # Depot
                if current_route:
                    solution.append(current_route)
                    current_route = []
                    current_load = 0
                continue
                
            # Kiểm tra nếu thêm khách hàng này vượt quá sức chứa
            customer_demand = self.cvrp.customers[gene].demand
            if current_load + customer_demand > capacity:
                # Tuyến hiện tại đã đầy, bắt đầu tuyến mới
                if current_route:
                    solution.append(current_route)
                    current_route = []
                    current_load = 0
            
            # Thêm khách hàng vào tuyến hiện tại
            current_route.append(gene)
            current_load += customer_demand
            
        # Thêm tuyến cuối cùng nếu có
        if current_route:
            solution.append(current_route)
            
        return solution