"""
Algorithm Comparison Application
GUI for comparing different CVRP algorithms
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import time
import random
import numpy as np

from core import CVRP, ACO_CVRP, GeneticAlgorithm_CVRP


class AlgorithmComparison:
    """Application for comparing CVRP algorithms"""

    def __init__(self, root, selector_root=None):
        """Initialize comparison application"""
        self.root = root
        self.selector_root = selector_root  # Reference to selector window for returning
        self.root.title("Compare Algorithms for CVRP")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize variables
        self.cvrp = CVRP()
        self.n_customers = 10  # Default, can be adjusted
        self.capacity = 100
        self.algorithms = {
            "ant_colony": {"selected": tk.BooleanVar(value=True), "name": "Ant Colony Optimization", "color": "blue"},
            "genetic": {"selected": tk.BooleanVar(value=True), "name": "Genetic Algorithm", "color": "green"}
        }
        self.results = {}
        self.is_running = False
        self.comparison_thread = None

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
        viz_panel = ttk.LabelFrame(main_panel, text="Comparison Results")
        main_panel.add(viz_panel, weight=3)

        # Create visualization
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        self.toolbar_frame = tk.Frame(viz_panel)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Create charts
        self.setup_plots()

        # Create control sections
        self.create_problem_generation_controls()
        self.create_algorithm_selection()
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
            if messagebox.askyesno("Return", "Comparison is running. Are you sure you want to return?"):
                self.stop_comparison()
                self.root.destroy()
                if self.selector_root:
                    self.selector_root.deiconify()  # Show selector window
        else:
            self.root.destroy()
            if self.selector_root:
                self.selector_root.deiconify()  # Show selector window

    def setup_plots(self):
        """Setup charts"""
        self.axs = self.fig.subplots(2, 1)
        self.fig.suptitle('Comparison of CVRP Algorithms', fontsize=14)

        # Execution time chart
        self.axs[0].set_title('Execution Time')
        self.axs[0].set_ylabel('Time (seconds)')
        self.axs[0].grid(True)

        # Solution cost chart
        self.axs[1].set_title('Best Solution Cost')
        self.axs[1].set_ylabel('Cost')
        self.axs[1].grid(True)

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

    def create_problem_generation_controls(self):
        """Create controls for CVRP problem generation"""
        problem_frame = ttk.LabelFrame(self.control_frame, text="Create CVRP Problem")
        problem_frame.pack(fill=tk.X, padx=5, pady=5)

        # Number of customers
        ttk.Label(problem_frame, text="Number of customers:").pack(anchor=tk.W, padx=5, pady=2)
        self.customer_count = tk.StringVar(value=str(self.n_customers))
        customer_count_entry = ttk.Entry(problem_frame, textvariable=self.customer_count, width=10)
        customer_count_entry.pack(fill=tk.X, padx=5, pady=2)

        # Warning label
        warning_label = ttk.Label(problem_frame, text="Note: Use 8-10 customers for quick comparison",
                                  foreground="red")
        warning_label.pack(anchor=tk.W, padx=5, pady=2)

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

    def create_algorithm_selection(self):
        """Create algorithm selection section"""
        algo_frame = ttk.LabelFrame(self.control_frame, text="Select Algorithms")
        algo_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create checkboxes for each algorithm
        for key, algo in self.algorithms.items():
            cb = ttk.Checkbutton(algo_frame, text=algo["name"], variable=algo["selected"])
            cb.pack(anchor=tk.W, padx=5, pady=2)

        # Select all/deselect all buttons
        button_frame = ttk.Frame(algo_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        select_all = ttk.Button(button_frame, text="Select All", command=self.select_all_algorithms)
        select_all.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        deselect_all = ttk.Button(button_frame, text="Deselect All", command=self.deselect_all_algorithms)
        deselect_all.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

    def create_execution_controls(self):
        """Create execution controls"""
        exec_frame = ttk.LabelFrame(self.control_frame, text="Execution")
        exec_frame.pack(fill=tk.X, padx=5, pady=5)

        # Run parameters
        params_frame = ttk.Frame(exec_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Number of repetitions for averaging
        ttk.Label(params_frame, text="Number of repetitions per algorithm:").pack(anchor=tk.W, padx=5, pady=2)
        self.num_runs = tk.StringVar(value="3")
        num_runs_entry = ttk.Entry(params_frame, textvariable=self.num_runs, width=5)
        num_runs_entry.pack(anchor=tk.W, padx=5, pady=2)

        # Time limit per algorithm (seconds)
        ttk.Label(params_frame, text="Time limit per algorithm (seconds):").pack(anchor=tk.W, padx=5, pady=2)
        self.time_limit = tk.StringVar(value="60")
        time_limit_entry = ttk.Entry(params_frame, textvariable=self.time_limit, width=5)
        time_limit_entry.pack(anchor=tk.W, padx=5, pady=2)

        # Control buttons
        button_frame = ttk.Frame(exec_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(button_frame, text="Start Comparison", command=self.start_comparison)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_comparison)
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        self.stop_button.config(state=tk.DISABLED)

        # Progress bar
        ttk.Label(exec_frame, text="Progress:").pack(anchor=tk.W, padx=5, pady=2)
        self.progress = ttk.Progressbar(exec_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=2)

    def create_result_display(self):
        """Create result display section"""
        result_frame = ttk.LabelFrame(self.control_frame, text="Results")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Detailed results
        ttk.Label(result_frame, text="Detailed results:").pack(anchor=tk.W, padx=5, pady=2)

        result_table_frame = ttk.Frame(result_frame)
        result_table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Create results table
        self.result_table = ttk.Treeview(result_table_frame, columns=("algorithm", "time", "cost", "routes"),
                                         show="headings")
        self.result_table.heading("algorithm", text="Algorithm")
        self.result_table.heading("time", text="Time (s)")
        self.result_table.heading("cost", text="Cost")
        self.result_table.heading("routes", text="Routes")

        self.result_table.column("algorithm", width=150)
        self.result_table.column("time", width=80)
        self.result_table.column("cost", width=80)
        self.result_table.column("routes", width=80)

        scrollbar = ttk.Scrollbar(result_table_frame, orient=tk.VERTICAL, command=self.result_table.yview)
        self.result_table.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def select_all_algorithms(self):
        """Select all algorithms"""
        for algo in self.algorithms.values():
            algo["selected"].set(True)

    def deselect_all_algorithms(self):
        """Deselect all algorithms"""
        for algo in self.algorithms.values():
            algo["selected"].set(False)

    def generate_problem(self):
        """Generate CVRP problem based on user preferences"""
        try:
            self.n_customers = int(self.customer_count.get())
            self.capacity = int(self.capacity_var.get())

            if self.n_customers < 5:
                messagebox.showwarning("Warning", "Need at least 5 customers")
                self.n_customers = 5
                self.customer_count.set("5")
            elif self.n_customers > 15:
                messagebox.showwarning("Warning", "Maximum 15 customers (for comparison)")
                self.n_customers = 15
                self.customer_count.set("15")

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

            # Update axis titles
            self.axs[0].set_title(f'Execution Time (customers: {self.n_customers})')
            self.axs[1].set_title(f'Best Solution Cost (customers: {self.n_customers})')
            self.canvas.draw()

            # Clear old results
            self.results = {}
            self.result_table.delete(*self.result_table.get_children())

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

                # Update axis titles
                self.axs[0].set_title(f'Execution Time (customers: {self.n_customers})')
                self.axs[1].set_title(f'Best Solution Cost (customers: {self.n_customers})')
                self.canvas.draw()

                # Clear old results
                self.results = {}
                self.result_table.delete(*self.result_table.get_children())

                messagebox.showinfo("Info", f"Loaded CVRP problem with {self.n_customers} customers")
            else:
                messagebox.showerror("Error", "Could not load file")

    def start_comparison(self):
        """Start algorithms comparison"""
        if not self.cvrp or not self.cvrp.customers:
            messagebox.showwarning("Warning", "Please create a CVRP problem first")
            return

        if self.is_running:
            return

        # Check if any algorithm is selected
        selected_algorithms = [key for key, algo in self.algorithms.items() if algo["selected"].get()]
        if not selected_algorithms:
            messagebox.showwarning("Warning", "Please select at least one algorithm")
            return

        try:
            self.num_runs_value = int(self.num_runs.get())
            self.time_limit_value = int(self.time_limit.get())

            if self.num_runs_value < 1:
                self.num_runs_value = 1
                self.num_runs.set("1")

            if self.time_limit_value < 1:
                self.time_limit_value = 60
                self.time_limit.set("60")
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters")
            return

        # Initialize results
        self.results = {}
        self.result_table.delete(*self.result_table.get_children())

        # Update GUI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.is_running = True

        # Clear old charts
        for ax in self.axs:
            ax.clear()

        self.axs[0].set_title(f'Execution Time (customers: {self.n_customers})')
        self.axs[0].set_ylabel('Time (seconds)')
        self.axs[0].grid(True)

        self.axs[1].set_title(f'Best Solution Cost (customers: {self.n_customers})')
        self.axs[1].set_ylabel('Cost')
        self.axs[1].grid(True)

        self.canvas.draw()

        # Initialize progress bar
        self.progress['value'] = 0
        self.progress['maximum'] = len(selected_algorithms) * self.num_runs_value

        # Start comparison thread
        self.comparison_thread = threading.Thread(target=self.run_comparison, args=(selected_algorithms,))
        self.comparison_thread.daemon = True
        self.comparison_thread.start()

    def run_comparison(self, selected_algorithms):
        """Run algorithm comparison in separate thread"""
        try:
            total_steps = len(selected_algorithms) * self.num_runs_value
            current_step = 0

            for key in selected_algorithms:
                if not self.is_running:
                    break

                algo_name = self.algorithms[key]["name"]

                # Initialize results for algorithm
                self.results[key] = {
                    "name": algo_name,
                    "times": [],
                    "costs": [],
                    "solutions": []
                }

                # Run algorithm multiple times
                for run in range(self.num_runs_value):
                    if not self.is_running:
                        break

                    # Update progress bar
                    current_step += 1
                    self.root.after(0, lambda step=current_step: self.progress.config(value=step))

                    # Show status
                    status_text = f"Running {algo_name} (run {run + 1}/{self.num_runs_value})..."
                    self.root.after(0, lambda text=status_text: self.root.title(f"Algorithm Comparison - {text}"))

                    # Run respective algorithm
                    start_time = time.time()
                    solution, cost = self.run_algorithm(key)
                    end_time = time.time()

                    execution_time = end_time - start_time

                    # Save results
                    self.results[key]["times"].append(execution_time)
                    self.results[key]["costs"].append(cost)
                    self.results[key]["solutions"].append(solution)

                    # Update charts after each run
                    self.root.after(0, self.update_charts)

            # Finish comparison
            self.root.after(0, self.finalize_comparison)

        except Exception as e:
            # Handle error
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error during comparison: {str(e)}"))
            self.root.after(0, self.finalize_comparison)

    def run_algorithm(self, algo_key):
        """Run a specific algorithm with time limit"""
        # Create a copy of CVRP to avoid interference
        cvrp_copy = CVRP(capacity=self.cvrp.capacity)

        # Copy data
        for customer in self.cvrp.customers:
            if customer.id == 0:  # Depot
                cvrp_copy.add_depot(customer.x, customer.y)
            else:
                cvrp_copy.add_customer(customer.id, customer.x, customer.y, customer.demand)

        cvrp_copy.calculate_distances()

        # Initialize respective algorithm
        algorithm = None

        if algo_key == "ant_colony":
            algorithm = ACO_CVRP(cvrp=cvrp_copy, num_ants=20, max_iterations=50)
        elif algo_key == "genetic":
            algorithm = GeneticAlgorithm_CVRP(cvrp=cvrp_copy, population_size=50, max_generations=100)

        # Variable to store result
        result = [None, float('inf')]  # [solution, cost]
        finish_event = threading.Event()

        # Callback when algorithm completes
        def on_finish(res):
            nonlocal result
            result = res
            finish_event.set()

        # Run algorithm in separate thread
        algo_thread = threading.Thread(target=algorithm.run, kwargs={"callback": on_finish})
        algo_thread.daemon = True
        algo_thread.start()

        # Wait for algorithm to finish or time limit reached
        finish_event.wait(timeout=self.time_limit_value)

        # Stop algorithm if running
        algorithm.stop()

        # Return result
        return result

    def update_charts(self):
        """Update charts with current results"""
        # Clear old charts
        for ax in self.axs:
            ax.clear()

        self.axs[0].set_title(f'Execution Time (customers: {self.n_customers})')
        self.axs[0].set_ylabel('Time (seconds)')
        self.axs[0].grid(True)

        self.axs[1].set_title(f'Best Solution Cost (customers: {self.n_customers})')
        self.axs[1].set_ylabel('Cost')
        self.axs[1].grid(True)

        # List of algorithms that have completed
        algo_keys = list(self.results.keys())
        if not algo_keys:
            return

        # Draw time chart
        times = []
        time_stds = []
        for key in algo_keys:
            if self.results[key]["times"]:
                times.append(np.mean(self.results[key]["times"]))
                time_stds.append(np.std(self.results[key]["times"]) if len(self.results[key]["times"]) > 1 else 0)
            else:
                times.append(0)
                time_stds.append(0)

        x = np.arange(len(algo_keys))
        algo_names = [self.algorithms[key]["name"] for key in algo_keys]
        colors = [self.algorithms[key]["color"] for key in algo_keys]

        # Draw bar chart with error bars
        bars = self.axs[0].bar(x, times, yerr=time_stds, align='center', alpha=0.7, color=colors, ecolor='black',
                               capsize=5)
        self.axs[0].set_xticks(x)
        self.axs[0].set_xticklabels(algo_names, rotation=45, ha='right')

        # Add values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            self.axs[0].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                             f'{times[i]:.2f}s',
                             ha='center', va='bottom', rotation=0, fontsize=8)

        # Draw cost chart
        costs = []
        cost_stds = []
        for key in algo_keys:
            if self.results[key]["costs"]:
                costs.append(np.mean(self.results[key]["costs"]))
                cost_stds.append(np.std(self.results[key]["costs"]) if len(self.results[key]["costs"]) > 1 else 0)
            else:
                costs.append(0)
                cost_stds.append(0)

        bars = self.axs[1].bar(x, costs, yerr=cost_stds, align='center', alpha=0.7, color=colors, ecolor='black',
                               capsize=5)
        self.axs[1].set_xticks(x)
        self.axs[1].set_xticklabels(algo_names, rotation=45, ha='right')

        # Add values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            self.axs[1].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                             f'{costs[i]:.2f}',
                             ha='center', va='bottom', rotation=0, fontsize=8)

        # Update results table
        self.update_result_table()

        # Redraw
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()

    def update_result_table(self):
        """Update results table"""
        # Clear old data
        self.result_table.delete(*self.result_table.get_children())

        # Add new data
        for key, result in self.results.items():
            if result["times"]:
                # Calculate averages
                mean_time = np.mean(result["times"])
                mean_cost = np.mean(result["costs"])

                # Count average number of routes
                routes_count = []
                for solution in result["solutions"]:
                    if solution:
                        routes_count.append(len(solution))

                mean_routes = np.mean(routes_count) if routes_count else 0

                # Add to table
                self.result_table.insert("", "end", values=(
                    result["name"],
                    f"{mean_time:.2f}",
                    f"{mean_cost:.2f}",
                    f"{mean_routes:.1f}"
                ))

    def stop_comparison(self):
        """Stop comparison process"""
        self.is_running = False

    def finalize_comparison(self):
        """Finalize comparison process"""
        self.is_running = False
        self.root.title("Compare Algorithms for CVRP")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        if self.results:
            # Show final results
            self.update_charts()
            # Show message
            messagebox.showinfo("Info", "Algorithm comparison completed")

    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askyesno("Exit", "Comparison is running. Are you sure you want to exit?"):
                self.stop_comparison()
                self.root.destroy()
                # Return to selector if it exists
                if self.selector_root:
                    self.selector_root.deiconify()
        else:
            self.root.destroy()
            # Return to selector if it exists
            if self.selector_root:
                self.selector_root.deiconify()