"""
Base Visualization Class for CVRP
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import numpy as np


class CVRPVisualization:
    """Base visualization class for CVRP"""

    def __init__(self, parent_frame):
        """Initialize visualization in parent_frame"""
        self.parent = parent_frame
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        self.toolbar_frame = tk.Frame(self.parent)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Setup subplots
        self.setup_axes()

        # Running algorithm
        self.algorithm = None

        # CVRP data
        self.cvrp = None

    def setup_axes(self):
        """Setup axes - base method to override"""
        pass

    def set_algorithm(self, algorithm):
        """Set algorithm so visualization can access it"""
        self.algorithm = algorithm

    def set_cvrp(self, cvrp):
        """Set CVRP data"""
        self.cvrp = cvrp

    def clear(self):
        """Clear all display data"""
        for ax in self.fig.get_axes():
            ax.clear()
        self.canvas.draw()

    def plot_customers(self, ax):
        """Plot customers and depot on given axis"""
        if not self.cvrp:
            return

        # Plot depot
        ax.scatter(self.cvrp.depot.x, self.cvrp.depot.y, c='red', s=100, marker='*', label='Depot')

        # Plot customers
        x = [customer.x for customer in self.cvrp.customers[1:]]
        y = [customer.y for customer in self.cvrp.customers[1:]]
        demand = [customer.demand for customer in self.cvrp.customers[1:]]

        scatter = ax.scatter(x, y, c=demand, cmap='viridis', s=50, label='Customers')

        # Add customer numbers
        for i, (xi, yi) in enumerate(zip(x, y), 1):  # Start from 1 (skip depot)
            ax.annotate(f"{i}", (xi, yi), xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Add colorbar for demand
        plt.colorbar(scatter, ax=ax, label='Demand')

        # Add legend
        ax.legend()

        # Set axis limits
        all_x = [customer.x for customer in self.cvrp.customers]
        all_y = [customer.y for customer in self.cvrp.customers]

        margin = 10  # Add margin
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)