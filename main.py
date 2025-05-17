"""
CVRP Algorithm Simulator
Main entry point for the application
"""

import tkinter as tk
from gui import AlgorithmSelector
from parameter_tester import ParameterTester


def main():
    """Application entry point"""
    root = tk.Tk()
    app = AlgorithmSelector(root)
    root.mainloop()


if __name__ == "__main__":
    main()