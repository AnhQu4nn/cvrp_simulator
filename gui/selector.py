"""
Algorithm Selection Interface
Allows user to select which algorithm to use
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

from .aco_app import AntColonyApp
from .genetic_app import GeneticApp
from .comparison_app import AlgorithmComparison


class AlgorithmSelector:
    """Main menu interface for selecting which algorithm to use"""

    def __init__(self, root):
        """Initialize algorithm selection interface"""
        self.root = root
        self.root.title("Select algorithm for CVRP problem")
        self.root.geometry("600x600")

        self.create_gui()

    def create_gui(self):
        """Create selection interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="CVRP Solver Application", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        subtitle_label = ttk.Label(main_frame,
                                   text="Select an algorithm to solve the Capacitated Vehicle Routing Problem",
                                   font=("Arial", 10))
        subtitle_label.pack(pady=5)

        # Frame for algorithm buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        # Create buttons for each algorithm
        algorithms = [
            ("Ant Colony Optimization", "Simulates ants finding routes", self.launch_ant_colony),
            ("Genetic Algorithm", "Evolves a population of solutions", self.launch_genetic),
        ]

        for idx, (title, desc, cmd) in enumerate(algorithms):
            frame = ttk.Frame(button_frame, padding="10", relief=tk.RAISED, borderwidth=1)
            frame.pack(fill=tk.X, pady=5)

            # Icon or illustration placeholder
            icon_frame = ttk.Frame(frame, width=50, height=50)
            icon_frame.pack(side=tk.LEFT, padx=10)

            # Algorithm info
            info_frame = ttk.Frame(frame)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

            algo_title = ttk.Label(info_frame, text=title, font=("Arial", 12, "bold"))
            algo_title.pack(anchor=tk.W)

            algo_desc = ttk.Label(info_frame, text=desc)
            algo_desc.pack(anchor=tk.W, pady=5)

            # Select button
            select_button = ttk.Button(frame, text="Select", command=cmd)
            select_button.pack(side=tk.RIGHT, padx=10)

        # Add comparison button
        compare_frame = ttk.Frame(button_frame, padding="10", relief=tk.RAISED, borderwidth=1)
        compare_frame.pack(fill=tk.X, pady=10)

        # Icon or illustration placeholder
        icon_frame = ttk.Frame(compare_frame, width=50, height=50)
        icon_frame.pack(side=tk.LEFT, padx=10)

        # Algorithm info
        info_frame = ttk.Frame(compare_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        title_label = ttk.Label(info_frame, text="Compare Algorithms", font=("Arial", 12, "bold"))
        title_label.pack(anchor=tk.W)

        desc_label = ttk.Label(info_frame, text="Compare performance of different algorithms")
        desc_label.pack(anchor=tk.W, pady=5)

        # Select button
        select_button = ttk.Button(compare_frame, text="Select", command=self.launch_comparison)
        select_button.pack(side=tk.RIGHT, padx=10)

        # Exit button
        exit_button = ttk.Button(main_frame, text="Exit", command=self.root.destroy)
        exit_button.pack(side=tk.BOTTOM, pady=10)

        # Information
        info_label = ttk.Label(main_frame, text="© 2025 - Simulation for CVRP Algorithms", font=("Arial", 8))
        info_label.pack(side=tk.BOTTOM, pady=10)

    def launch_ant_colony(self):
        """Launch Ant Colony Optimization application"""
        self.root.withdraw()  # Hide selector window
        ant_colony_root = tk.Toplevel(self.root)
        ant_colony_root.protocol("WM_DELETE_WINDOW", lambda: self.on_app_closing(ant_colony_root))
        app = AntColonyApp(ant_colony_root, self.root)

    def launch_genetic(self):
        """Launch Genetic Algorithm application"""
        self.root.withdraw()  # Hide selector window
        genetic_root = tk.Toplevel(self.root)
        genetic_root.protocol("WM_DELETE_WINDOW", lambda: self.on_app_closing(genetic_root))
        app = GeneticApp(genetic_root, self.root)

    def launch_comparison(self):
        """Launch Algorithm Comparison application"""
        self.root.withdraw()  # Hide selector window
        comparison_root = tk.Toplevel(self.root)
        comparison_root.protocol("WM_DELETE_WINDOW", lambda: self.on_app_closing(comparison_root))
        app = AlgorithmComparison(comparison_root, self.root)

    def on_app_closing(self, window):
        """Handle app window closing - return to selector"""
        window.destroy()
        self.root.deiconify()  # Show selector window again