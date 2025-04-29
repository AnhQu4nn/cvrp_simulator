"""
Genetic Algorithm Application
GUI for solving CVRP with Genetic Algorithm
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import random
import numpy as np

from core import CVRP, GeneticAlgorithm_CVRP
from .visualization import GeneticVisualization


class GeneticApp:
    """CVRP Application using Genetic Algorithm"""

    def __init__(self, root, selector_root=None):
        """Initialize Genetic Algorithm application"""
        self.root = root
        self.selector_root = selector_root  # Reference to selector window for returning
        self.root.title("Solve CVRP with Genetic Algorithm")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize variables
        self.cvrp = CVRP()
        self.n_customers = 15
        self.capacity = 100
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism = 5
        self.max_generations = 100
        self.algorithm = None
        self.algorithm_thread = None
        self.is_running = False

        # Create GUI
        self.create_gui()

        # Set default button
        self.select_random.invoke()

    def create_gui(self):
        """Create user interface"""
        # Main panel
        main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel on left
        self.control_frame = ttk.LabelFrame(main_panel, text="Controls")
        main_panel.add(self.control_frame, weight=1)

        # Visualization panel on right
        viz_panel = ttk.LabelFrame(main_panel, text="Visualization")
        main_panel.add(viz_panel, weight=3)

        # Create visualization
        self.visualization = GeneticVisualization(viz_panel)

        # Create control sections
        self.create_problem_generation_controls()
        self.create_algorithm_controls()
        self.create_execution_controls()
        self.create_result_display()

        # Add button to return to algorithm selector
        self.add_return_button()

    def add_return_button(self):
        """Add button to return to algorithm selector"""
        back_button = ttk.Button(
            self.control_frame,
            text="Return to Algorithm Selection",
            command=self.back_to_selector
        )
        back_button.pack(fill=tk.X, padx=5, pady=5)

    def back_to_selector(self):
        """Return to algorithm selection menu"""
        if self.is_running:
            if messagebox.askyesno("Return", "Algorithm is running. Are you sure you want to return?"):
                self.stop_algorithm()
                self.root.destroy()
                if self.selector_root:
                    self.selector_root.deiconify()  # Show selector window
        else:
            self.root.destroy()
            if self.selector_root:
                self.selector_root.deiconify()  # Show selector window

    def create_problem_generation_controls(self):
        """Create controls for CVRP problem generation"""
        problem_frame = ttk.LabelFrame(self.control_frame, text="Create CVRP Problem")
        problem_frame.pack(fill=tk.X, padx=5, pady=5)

        # Number of customers
        ttk.Label(problem_frame, text="Number of customers:").pack(anchor=tk.W, padx=5, pady=2)
        self.customer_count = tk.StringVar(value=str(self.n_customers))
        customer_count_entry = ttk.Entry(problem_frame, textvariable=self.customer_count, width=10)
        customer_count_entry.pack(fill=tk.X, padx=5, pady=2)

        # Vehicle capacity
        ttk.Label(problem_frame, text="Vehicle capacity:").pack(anchor=tk.W, padx=5, pady=2)
        self.capacity_var = tk.StringVar(value=str(self.capacity))
        capacity_entry = ttk.Entry(problem_frame, textvariable=self.capacity_var, width=10)
        capacity_entry.pack(fill=tk.X, padx=5, pady=2)

        # Problem type
        ttk.Label(problem_frame, text="Problem type:").pack(anchor=tk.W, padx=5, pady=2)

        self.problem_type = tk.StringVar(value="random")
        self.select_random = ttk.Radiobutton(problem_frame, text="Random",
                                             variable=self.problem_type, value="random")
        self.select_random.pack(anchor=tk.W, padx=5, pady=2)

        # Seed
        ttk.Label(problem_frame, text="Seed (optional):").pack(anchor=tk.W, padx=5, pady=2)
        self.seed_var = tk.StringVar()
        seed_entry = ttk.Entry(problem_frame, textvariable=self.seed_var, width=10)
        seed_entry.pack(fill=tk.X, padx=5, pady=2)

        # Generate problem button
        generate_button = ttk.Button(problem_frame, text="Generate Problem", command=self.generate_problem)
        generate_button.pack(fill=tk.X, padx=5, pady=5)

        # Save/load problem buttons
        button_frame = ttk.Frame(problem_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        save_button = ttk.Button(button_frame, text="Save", command=self.save_problem)
        save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        load_button = ttk.Button(button_frame, text="Load", command=self.load_problem)
        load_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

    def create_algorithm_controls(self):
        """Create controls for Genetic Algorithm parameters"""
        algo_frame = ttk.LabelFrame(self.control_frame, text="Algorithm Parameters")
        algo_frame.pack(fill=tk.X, padx=5, pady=5)

        # Population size
        ttk.Label(algo_frame, text="Population size:").pack(anchor=tk.W, padx=5, pady=2)
        self.population_size_var = tk.StringVar(value=str(self.population_size))
        population_size_entry = ttk.Entry(algo_frame, textvariable=self.population_size_var, width=10)
        population_size_entry.pack(fill=tk.X, padx=5, pady=2)

        # Mutation rate
        ttk.Label(algo_frame, text="Mutation rate:").pack(anchor=tk.W, padx=5, pady=2)
        self.mutation_rate_var = tk.StringVar(value=str(self.mutation_rate))
        mutation_rate_entry = ttk.Entry(algo_frame, textvariable=self.mutation_rate_var, width=10)
        mutation_rate_entry.pack(fill=tk.X, padx=5, pady=2)

        # Crossover rate
        ttk.Label(algo_frame, text="Crossover rate:").pack(anchor=tk.W, padx=5, pady=2)
        self.crossover_rate_var = tk.StringVar(value=str(self.crossover_rate))
        crossover_rate_entry = ttk.Entry(algo_frame, textvariable=self.crossover_rate_var, width=10)
        crossover_rate_entry.pack(fill=tk.X, padx=5, pady=2)

        # Elitism (number of elite individuals)
        ttk.Label(algo_frame, text="Elite individuals:").pack(anchor=tk.W, padx=5, pady=2)
        self.elitism_var = tk.StringVar(value=str(self.elitism))
        elitism_entry = ttk.Entry(algo_frame, textvariable=self.elitism_var, width=10)
        elitism_entry.pack(fill=tk.X, padx=5, pady=2)

        # Maximum generations
        ttk.Label(algo_frame, text="Maximum generations:").pack(anchor=tk.W, padx=5, pady=2)
        self.generations_var = tk.StringVar(value=str(self.max_generations))
        generations_entry = ttk.Entry(algo_frame, textvariable=self.generations_var, width=10)
        generations_entry.pack(fill=tk.X, padx=5, pady=2)

    def create_execution_controls(self):
        """Create execution controls"""
        exec_frame = ttk.LabelFrame(self.control_frame, text="Execution")
        exec_frame.pack(fill=tk.X, padx=5, pady=5)

        # Control buttons
        button_frame = ttk.Frame(exec_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_algorithm)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_algorithm)
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        self.stop_button.config(state=tk.DISABLED)

        # Progress bar
        ttk.Label(exec_frame, text="Progress:").pack(anchor=tk.W, padx=5, pady=2)
        self.progress = ttk.Progressbar(exec_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=2)

        # Simulation speed
        ttk.Label(exec_frame, text="Simulation speed:").pack(anchor=tk.W, padx=5, pady=2)
        self.speed_var = tk.DoubleVar(value=0.1)
        speed_scale = ttk.Scale(exec_frame, from_=0.01, to=1.0, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.pack(fill=tk.X, padx=5, pady=2)

    def create_result_display(self):
        """Create result display section"""
        result_frame = ttk.LabelFrame(self.control_frame, text="Results")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Best cost
        ttk.Label(result_frame, text="Best cost:").pack(anchor=tk.W, padx=5, pady=2)
        self.best_cost_var = tk.StringVar(value="...")
        ttk.Label(result_frame, textvariable=self.best_cost_var).pack(anchor=tk.W, padx=5, pady=2)

        # Number of routes
        ttk.Label(result_frame, text="Number of routes:").pack(anchor=tk.W, padx=5, pady=2)
        self.route_count_var = tk.StringVar(value="...")
        ttk.Label(result_frame, textvariable=self.route_count_var).pack(anchor=tk.W, padx=5, pady=2)

        # Best solution
        ttk.Label(result_frame, text="Best solution:").pack(anchor=tk.W, padx=5, pady=2)

        solution_frame = ttk.Frame(result_frame)
        solution_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        scrollbar = ttk.Scrollbar(solution_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.solution_text = tk.Text(solution_frame, height=6, width=20, wrap=tk.WORD,
                                     yscrollcommand=scrollbar.set)
        self.solution_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.solution_text.yview)

        # Execution time
        ttk.Label(result_frame, text="Execution time:").pack(anchor=tk.W, padx=5, pady=2)
        self.execution_time_var = tk.StringVar(value="...")
        ttk.Label(result_frame, textvariable=self.execution_time_var).pack(anchor=tk.W, padx=5, pady=2)

    def generate_problem(self):
        """Generate CVRP problem based on user preferences"""
        try:
            self.n_customers = int(self.customer_count.get())
            self.capacity = int(self.capacity_var.get())

            if self.n_customers < 5:
                messagebox.showwarning("Warning", "Need at least 5 customers")
                self.n_customers = 5
                self.customer_count.set("5")
            elif self.n_customers > 100:
                messagebox.showwarning("Warning", "Maximum 100 customers")
                self.n_customers = 100
                self.customer_count.set("100")

            if self.capacity < 50:
                messagebox.showwarning("Warning", "Minimum capacity is 50")
                self.capacity = 50
                self.capacity_var.set("50")

            # Get seed if provided
            seed_str = self.seed_var.get()
            seed = int(seed_str) if seed_str else None

            # Create CVRP problem
            self.cvrp = CVRP(capacity=self.capacity)
            self.cvrp.load_problem(self.n_customers, self.capacity, seed)

            # Update visualization
            self.update_parameters()
            self.visualization.init_visualization(self.cvrp, self.population_size, self.max_generations)

            messagebox.showinfo("Info", f"Created CVRP problem with {self.n_customers} customers")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid data: {e}")

    def save_problem(self):
        """Save CVRP problem to file"""
        if not self.cvrp or not self.cvrp.customers:
            messagebox.showwarning("Warning", "No CVRP problem to save")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save CVRP Problem"
        )

        if filename:
            success = self.cvrp.save_to_file(filename)
            if success:
                messagebox.showinfo("Info",
                                    f"Saved CVRP problem with {len(self.cvrp.customers) - 1} customers to {filename}")
            else:
                messagebox.showerror("Error", "Could not save file")

    def load_problem(self):
        """Load CVRP problem from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load CVRP Problem"
        )

        if filename:
            self.cvrp = CVRP()
            success = self.cvrp.load_from_file(filename)

            if success:
                self.n_customers = len(self.cvrp.customers) - 1  # Minus depot
                self.capacity = self.cvrp.capacity

                self.customer_count.set(str(self.n_customers))
                self.capacity_var.set(str(self.capacity))

                # Update visualization
                self.update_parameters()
                self.visualization.init_visualization(self.cvrp, self.population_size, self.max_generations)

                messagebox.showinfo("Info", f"Loaded CVRP problem with {self.n_customers} customers")
            else:
                messagebox.showerror("Error", "Could not load file")

    def update_parameters(self):
        """Update parameters from GUI"""
        try:
            self.population_size = int(self.population_size_var.get())
            self.mutation_rate = float(self.mutation_rate_var.get())
            self.crossover_rate = float(self.crossover_rate_var.get())
            self.elitism = int(self.elitism_var.get())
            self.max_generations = int(self.generations_var.get())

            # Check limits
            if self.population_size < 10:
                self.population_size = 10
                self.population_size_var.set("10")

            if self.max_generations < 1:
                self.max_generations = 1
                self.generations_var.set("1")

            if self.mutation_rate < 0 or self.mutation_rate > 1:
                self.mutation_rate = max(0, min(1, self.mutation_rate))
                self.mutation_rate_var.set(str(self.mutation_rate))

            if self.crossover_rate < 0 or self.crossover_rate > 1:
                self.crossover_rate = max(0, min(1, self.crossover_rate))
                self.crossover_rate_var.set(str(self.crossover_rate))

            if self.elitism < 0:
                self.elitism = 0
                self.elitism_var.set("0")
            elif self.elitism > self.population_size // 2:
                self.elitism = self.population_size // 2
                self.elitism_var.set(str(self.elitism))

        except ValueError:
            messagebox.showerror("Error", "Invalid parameters, using default values")
            self.population_size = 50
            self.mutation_rate = 0.1
            self.crossover_rate = 0.8
            self.elitism = 5
            self.max_generations = 100

            self.population_size_var.set(str(self.population_size))
            self.mutation_rate_var.set(str(self.mutation_rate))
            self.crossover_rate_var.set(str(self.crossover_rate))
            self.elitism_var.set(str(self.elitism))
            self.generations_var.set(str(self.max_generations))

    def start_algorithm(self):
        """Start the algorithm"""
        if not self.cvrp or not self.cvrp.customers:
            messagebox.showwarning("Warning", "Please create a CVRP problem first")
            return

        if self.is_running:
            return

        # Update parameters
        self.update_parameters()

        # Initialize progress bar
        self.progress['value'] = 0
        self.progress['maximum'] = 100

        # Initialize algorithm
        self.algorithm = GeneticAlgorithm_CVRP(
            cvrp=self.cvrp,
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elitism=self.elitism,
            max_generations=self.max_generations
        )

        # Setup visualization
        self.visualization.set_algorithm(self.algorithm)
        self.visualization.set_cvrp(self.cvrp)
        self.visualization.init_visualization(self.cvrp, self.population_size, self.max_generations)

        # Update GUI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.is_running = True
        self.start_time = time.time()

        # Start algorithm thread
        self.algorithm_thread = threading.Thread(target=self.run_algorithm)
        self.algorithm_thread.daemon = True
        self.algorithm_thread.start()

    def run_algorithm(self):
        """Run algorithm in a separate thread"""
        try:
            self.algorithm.run(
                callback=self.algorithm_finished,
                step_callback=self.algorithm_step
            )
        except Exception as e:
            # Handle error
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error running algorithm: {str(e)}"))
            self.algorithm_finished(None)

    def algorithm_step(self, data):
        """Callback for each step of algorithm"""
        # Update GUI in main thread
        self.root.after(0, lambda: self.update_visualization(data))

        # Update progress bar
        progress = data.get('progress', 0)
        self.root.after(0, lambda: self.progress.config(value=progress * 100))

        # Delay based on simulation speed
        time.sleep(1.0 - self.speed_var.get())

    def update_visualization(self, data):
        """Update GUI after each algorithm step"""
        if not self.is_running:
            return

        # Update visualization
        self.visualization.update_visualization(data)

        # Update results
        best_cost = data.get('best_cost', float('inf'))
        self.best_cost_var.set(f"{best_cost:.2f}")

        best_solution = data.get('best_solution', None)
        if best_solution:
            self.route_count_var.set(str(len(best_solution)))

            # Display solution
            solution_str = ""
            for i, route in enumerate(best_solution):
                solution_str += f"Route {i + 1}: 0 → {' → '.join(map(str, route))} → 0\n"

            self.solution_text.delete(1.0, tk.END)
            self.solution_text.insert(tk.END, solution_str)

        # Update time
        elapsed_time = time.time() - self.start_time
        self.execution_time_var.set(f"{elapsed_time:.2f} seconds")

    def algorithm_finished(self, result):
        """Callback when algorithm completes"""
        self.is_running = False

        # Update GUI
        self.root.after(0, lambda: self.finalize_ui())

        if result:
            best_solution, best_cost = result

            # Update results in main thread
            self.root.after(0, lambda: self.update_final_result(best_solution, best_cost))

    def finalize_ui(self):
        """Update GUI after algorithm completes"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Update time
        elapsed_time = time.time() - self.start_time
        self.execution_time_var.set(f"{elapsed_time:.2f} seconds")

    def update_final_result(self, best_solution, best_cost):
        """Update final results"""
        self.best_cost_var.set(f"{best_cost:.2f}")

        if best_solution:
            self.route_count_var.set(str(len(best_solution)))

            # Display solution
            solution_str = ""
            for i, route in enumerate(best_solution):
                solution_str += f"Route {i + 1}: 0 → {' → '.join(map(str, route))} → 0\n"

            self.solution_text.delete(1.0, tk.END)
            self.solution_text.insert(tk.END, solution_str)

    def stop_algorithm(self):
        """Stop the algorithm"""
        if self.algorithm and self.is_running:
            self.algorithm.stop()

    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askyesno("Exit", "Algorithm is running. Are you sure you want to exit?"):
                self.stop_algorithm()
                self.root.destroy()
                # Return to selector if it exists
                if self.selector_root:
                    self.selector_root.deiconify()
        else:
            self.root.destroy()
            # Return to selector if it exists
            if self.selector_root:
                self.selector_root.deiconify()