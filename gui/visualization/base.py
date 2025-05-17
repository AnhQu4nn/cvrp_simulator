"""
Base Visualization Class for CVRP
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import os
import datetime


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
        
        # Results data
        self.best_solution = None
        self.best_cost = float('inf')
        self.execution_time = 0
        self.cost_history = []

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

        # Clear previous colorbars for this axis (to prevent duplicates)
        for cbar in self.fig.get_axes():
            if cbar.get_label() == 'Colorbar':
                if cbar.figure == ax.figure and cbar not in self.fig.axes[:4]:  # Don't remove the main 4 plot axes
                    self.fig.delaxes(cbar)

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
        ax.figure.colorbar(scatter, ax=ax, label='Demand')

        # Remove legend to make plot cleaner
        # ax.legend()

        # Set axis limits
        all_x = [customer.x for customer in self.cvrp.customers]
        all_y = [customer.y for customer in self.cvrp.customers]

        margin = 10  # Add margin
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
    def export_results(self, algorithm_name):
        """Xuất kết quả ra file"""
        if not self.best_solution or not self.cvrp:
            messagebox.showwarning("Cảnh báo", f"Chưa có kết quả {algorithm_name} để xuất. Vui lòng chạy thuật toán trước.")
            return
            
        # Tạo thư mục results nếu chưa tồn tại
        results_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Tạo tên subfolder dựa trên thời gian
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder_name = f"{algorithm_name}_{timestamp}"
        subfolder_path = os.path.join(results_dir, subfolder_name)
        
        # Tạo subfolder
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            
        # Lưu kết quả vào file
        filepath = os.path.join(subfolder_path, "results.txt")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"KẾT QUẢ THUẬT TOÁN {algorithm_name.upper()}\n")
            f.write("=" * 40 + "\n\n")
            
            # Thông tin bài toán
            f.write("THÔNG TIN BÀI TOÁN\n")
            f.write("-" * 20 + "\n")
            f.write(f"Số điểm giao hàng: {len(self.cvrp.customers) - 1}\n")
            f.write(f"Sức chứa xe: {self.cvrp.capacity}\n\n")
            
            # Thông tin thuật toán và tham số
            self.write_algorithm_params(f)
            
            # Kết quả
            f.write(f"\nKẾT QUẢ {algorithm_name.upper()}\n")
            f.write("-" * 20 + "\n")
            non_empty_routes = sum(1 for route in self.best_solution if route)
            f.write(f"Chi phí tốt nhất: {self.best_cost:.2f}\n")
            f.write(f"Số xe sử dụng: {non_empty_routes}\n")
            f.write(f"Thời gian thực thi: {self.execution_time:.2f} seconds\n\n")
            
            # Chi tiết tuyến đường
            f.write(f"CHI TIẾT TUYẾN ĐƯỜNG {algorithm_name.upper()}\n")
            f.write("-" * 20 + "\n")
            for i, route in enumerate(self.best_solution):
                if route:  # Chỉ ghi tuyến đường không rỗng
                    route_demand = sum(self.cvrp.customers[node].demand for node in route)
                    route_distance = self.cvrp.calculate_route_distance(route)
                    f.write(f"Tuyến {i+1}: {route} - Nhu cầu: {route_demand} - Khoảng cách: {route_distance:.2f}\n")
            f.write("\n")
            
            # Hội tụ
            if self.cost_history:
                f.write("THÔNG TIN HỘI TỤ\n")
                f.write("-" * 20 + "\n")
                f.write(f"Số vòng lặp: {len(self.cost_history)}\n")
                f.write(f"Chi phí ban đầu: {self.cost_history[0]:.2f}\n")
                f.write(f"Chi phí cuối cùng: {self.cost_history[-1]:.2f}\n")
                f.write(f"Cải thiện: {self.cost_history[0] - self.cost_history[-1]:.2f} ({(self.cost_history[0] - self.cost_history[-1]) / self.cost_history[0] * 100:.2f}%)\n")
                
                if len(self.cost_history) > 1:
                    # Tính tốc độ hội tụ
                    avg_improve = (self.cost_history[0] - self.cost_history[-1]) / len(self.cost_history)
                    f.write(f"Tốc độ cải thiện trung bình: {avg_improve:.2f} đơn vị/vòng lặp\n")
                    
                    # Điểm hội tụ
                    convergence_threshold = 0.01  # 1% improvement threshold
                    converged_at = 0
                    for i in range(1, len(self.cost_history)):
                        if (self.cost_history[i-1] - self.cost_history[i]) / self.cost_history[i-1] < convergence_threshold:
                            converged_at = i
                            break
                    
                    if converged_at > 0:
                        f.write(f"Hội tụ sau khoảng: {converged_at} vòng lặp\n")
                    else:
                        f.write("Không xác định được điểm hội tụ rõ ràng\n")
            
        # Lưu biểu đồ hội tụ
        try:
            plt.figure(figsize=(10, 6))
            if self.cost_history:
                plt.plot(range(len(self.cost_history)), self.cost_history, 'b-')
                plt.title(f"Hội tụ của thuật toán {algorithm_name}")
                plt.xlabel("Vòng lặp")
                plt.ylabel("Chi phí tốt nhất")
                plt.grid(True, linestyle='--', alpha=0.7)
                
                chart_filepath = os.path.join(subfolder_path, "convergence.png")
                plt.savefig(chart_filepath)
                plt.close()
        except Exception as e:
            messagebox.showwarning("Cảnh báo", f"Không thể lưu biểu đồ: {str(e)}")
            
        # Lưu bản đồ tuyến đường
        try:
            self.fig.savefig(os.path.join(subfolder_path, "routes.png"))
        except Exception as e:
            messagebox.showwarning("Cảnh báo", f"Không thể lưu bản đồ tuyến đường: {str(e)}")
            
        messagebox.showinfo("Thành công", f"Đã xuất kết quả {algorithm_name} vào thư mục {subfolder_path}")
        
    def write_algorithm_params(self, file):
        """
        Ghi các tham số thuật toán vào file - ghi đè bởi các lớp con
        
        Tham số:
        file -- file object đang mở để ghi
        """
        pass