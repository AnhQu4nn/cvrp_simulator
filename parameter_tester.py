"""
Phần mềm Thử nghiệm Tham số cho thuật toán ACO và GA
Dùng để đánh giá ảnh hưởng của các tham số đến hiệu suất thuật toán
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import threading
import os
import json
import time
import itertools
from datetime import datetime

from core.cvrp import CVRP
from core.aco import ACO_CVRP
from core.genetic import GeneticAlgorithm_CVRP

class ParameterTester(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Phần mềm Thử nghiệm Tham số CVRP")
        self.geometry("1200x800")
        
        # Biến lưu trữ
        self.cvrp = None
        self.algorithm = tk.StringVar(value="ACO")
        self.num_customers = tk.IntVar(value=20)
        self.vehicle_capacity = tk.IntVar(value=100)
        self.problem_seed = tk.IntVar(value=42)
        self.results = []
        self.running = False
        self.lock = threading.Lock()
        self.current_experiment = None
        
        # Danh sách cấu hình thử nghiệm
        self.aco_test_configs = []
        self.ga_test_configs = []
        
        # Tạo giao diện
        self.create_widgets()
        
    def create_widgets(self):
        # Panel trái cho cấu hình
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        # Panel phải cho kết quả và biểu đồ
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame cho thông tin bài toán
        problem_frame = ttk.LabelFrame(left_frame, text="Thông tin bài toán")
        problem_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(problem_frame, text="Số khách hàng:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(problem_frame, textvariable=self.num_customers, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(problem_frame, text="Tải trọng xe:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(problem_frame, textvariable=self.vehicle_capacity, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(problem_frame, text="Seed ngẫu nhiên:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(problem_frame, textvariable=self.problem_seed, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Button(problem_frame, text="Tạo bài toán", command=self.create_problem).grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        ttk.Button(problem_frame, text="Tải bài toán", command=self.load_problem).grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        # Frame cho chọn thuật toán
        algorithm_frame = ttk.LabelFrame(left_frame, text="Thuật toán")
        algorithm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(algorithm_frame, text="ACO", variable=self.algorithm, value="ACO", command=self.update_parameter_frame).pack(anchor=tk.W, padx=5, pady=5)
        ttk.Radiobutton(algorithm_frame, text="GA", variable=self.algorithm, value="GA", command=self.update_parameter_frame).pack(anchor=tk.W, padx=5, pady=5)
        
        # Frame cho cấu hình tham số
        self.parameter_frame = ttk.LabelFrame(left_frame, text="Tham số")
        self.parameter_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame cho danh sách cấu hình
        self.configs_frame = ttk.LabelFrame(left_frame, text="Danh sách cấu hình thử nghiệm")
        self.configs_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Tạo danh sách cấu hình
        self.configs_list = tk.Listbox(self.configs_frame, height=5)
        self.configs_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Thanh cuộn cho danh sách cấu hình
        scrollbar = ttk.Scrollbar(self.configs_frame, orient=tk.VERTICAL, command=self.configs_list.yview)
        self.configs_list.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame cho nút thêm/xóa cấu hình
        config_buttons_frame = ttk.Frame(self.configs_frame)
        config_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(config_buttons_frame, text="Thêm cấu hình hiện tại", command=self.add_current_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_buttons_frame, text="Xóa cấu hình đã chọn", command=self.remove_selected_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_buttons_frame, text="Xóa tất cả", command=self.clear_configs).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_buttons_frame, text="Thêm nhiều cấu hình", command=self.add_multiple_configs).pack(side=tk.LEFT, padx=5)
        
        # Frame cho các nút điều khiển
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Chạy thử nghiệm", command=self.run_experiment).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="Dừng", command=self.stop_experiment).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="Lưu kết quả", command=self.save_results).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Notebook cho biểu đồ và kết quả
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab cho biểu đồ
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Biểu đồ")
        
        # Tab cho bảng kết quả
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Kết quả")
        
        # Tạo bảng kết quả
        columns = ('id', 'algorithm', 'parameters', 'best_cost', 'avg_cost', 'time')
        self.result_tree = ttk.Treeview(self.results_frame, columns=columns, show='headings')
        
        # Đặt tên cột
        self.result_tree.heading('id', text='ID')
        self.result_tree.heading('algorithm', text='Thuật toán')
        self.result_tree.heading('parameters', text='Tham số')
        self.result_tree.heading('best_cost', text='Chi phí tốt nhất')
        self.result_tree.heading('avg_cost', text='Chi phí trung bình')
        self.result_tree.heading('time', text='Thời gian (s)')
        
        # Điều chỉnh chiều rộng cột
        self.result_tree.column('id', width=50)
        self.result_tree.column('algorithm', width=100)
        self.result_tree.column('parameters', width=300)
        self.result_tree.column('best_cost', width=120)
        self.result_tree.column('avg_cost', width=120)
        self.result_tree.column('time', width=100)
        
        # Tạo thanh cuộn
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscroll=scrollbar.set)
        
        # Sắp xếp layout
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Khởi tạo các tham số
        self.update_parameter_frame()
        
        # Tạo biểu đồ mặc định
        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_parameter_frame(self):
        # Xóa các widget cũ
        for widget in self.parameter_frame.winfo_children():
            widget.destroy()
            
        self.parameter_vars = {}
        
        if self.algorithm.get() == "ACO":
            # Tham số cơ bản ACO
            ttk.Label(self.parameter_frame, text="Số lượng kiến:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["num_ants"] = tk.IntVar(value=20)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["num_ants"], width=10).grid(row=0, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Alpha (tầm quan trọng pheromone):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["alpha"] = tk.DoubleVar(value=1.0)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["alpha"], width=10).grid(row=1, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Beta (tầm quan trọng heuristic):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["beta"] = tk.DoubleVar(value=2.0)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["beta"], width=10).grid(row=2, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Rho (tỷ lệ bay hơi pheromone):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["rho"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["rho"], width=10).grid(row=3, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Q (hệ số lượng pheromone):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["q"] = tk.DoubleVar(value=100)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["q"], width=10).grid(row=4, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Số vòng lặp tối đa:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["max_iterations"] = tk.IntVar(value=100)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["max_iterations"], width=10).grid(row=5, column=1, padx=5, pady=3)
            
            # Tham số nâng cao ACO
            ttk.Separator(self.parameter_frame, orient='horizontal').grid(row=6, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
            ttk.Label(self.parameter_frame, text="Tính năng nâng cao:", font=('Helvetica', 10, 'bold')).grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=3)
            
            self.parameter_vars["min_max_aco"] = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_frame, text="MIN-MAX ACO", variable=self.parameter_vars["min_max_aco"]).grid(row=8, column=0, columnspan=2, sticky=tk.W, padx=5, pady=3)
            
            self.parameter_vars["local_search"] = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_frame, text="Tìm kiếm cục bộ 2-opt", variable=self.parameter_vars["local_search"]).grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Số kiến ưu tú:").grid(row=10, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["elitist_ants"] = tk.IntVar(value=0)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["elitist_ants"], width=10).grid(row=10, column=1, padx=5, pady=3)
            
            # Cập nhật danh sách cấu hình
            self.update_configs_list()
            
        else:  # Thuật toán GA
            # Tham số cơ bản GA
            ttk.Label(self.parameter_frame, text="Kích thước quần thể:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["population_size"] = tk.IntVar(value=50)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["population_size"], width=10).grid(row=0, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Xác suất đột biến:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["mutation_rate"] = tk.DoubleVar(value=0.1)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["mutation_rate"], width=10).grid(row=1, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Xác suất lai ghép:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["crossover_rate"] = tk.DoubleVar(value=0.8)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["crossover_rate"], width=10).grid(row=2, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Số cá thể ưu tú:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["elitism"] = tk.IntVar(value=5)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["elitism"], width=10).grid(row=3, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Số thế hệ tối đa:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["max_generations"] = tk.IntVar(value=100)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["max_generations"], width=10).grid(row=4, column=1, padx=5, pady=3)
            
            # Tham số nâng cao GA
            ttk.Separator(self.parameter_frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
            ttk.Label(self.parameter_frame, text="Tính năng nâng cao:", font=('Helvetica', 10, 'bold')).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Phương pháp chọn lọc:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["selection_method"] = tk.StringVar(value="tournament")
            selection_combo = ttk.Combobox(self.parameter_frame, textvariable=self.parameter_vars["selection_method"], width=15)
            selection_combo['values'] = ('tournament', 'roulette', 'rank')
            selection_combo.grid(row=7, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Kích thước tournament:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["tournament_size"] = tk.IntVar(value=3)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["tournament_size"], width=10).grid(row=8, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Phương pháp lai ghép:").grid(row=9, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["crossover_method"] = tk.StringVar(value="ordered")
            crossover_combo = ttk.Combobox(self.parameter_frame, textvariable=self.parameter_vars["crossover_method"], width=15)
            crossover_combo['values'] = ('ordered', 'partially_mapped', 'cycle')
            crossover_combo.grid(row=9, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Phương pháp đột biến:").grid(row=10, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["mutation_method"] = tk.StringVar(value="swap")
            mutation_combo = ttk.Combobox(self.parameter_frame, textvariable=self.parameter_vars["mutation_method"], width=15)
            mutation_combo['values'] = ('swap', 'insert', 'inversion', 'scramble')
            mutation_combo.grid(row=10, column=1, padx=5, pady=3)
            
            ttk.Label(self.parameter_frame, text="Dừng sớm (thế hệ):").grid(row=11, column=0, sticky=tk.W, padx=5, pady=3)
            self.parameter_vars["early_stopping"] = tk.IntVar(value=20)
            ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["early_stopping"], width=10).grid(row=11, column=1, padx=5, pady=3)
            
            # Thêm tùy chọn tìm kiếm cục bộ cho GA
            self.parameter_vars["local_search"] = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.parameter_frame, text="Sử dụng tìm kiếm cục bộ 2-opt", variable=self.parameter_vars["local_search"]).grid(row=12, column=0, columnspan=2, sticky=tk.W, padx=5, pady=3)
            
            # Cập nhật danh sách cấu hình
            self.update_configs_list()
            
        # Thêm phần cấu hình thử nghiệm
        ttk.Separator(self.parameter_frame, orient='horizontal').grid(row=13, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        ttk.Label(self.parameter_frame, text="Cấu hình thử nghiệm:", font=('Helvetica', 10, 'bold')).grid(row=14, column=0, columnspan=2, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(self.parameter_frame, text="Số lần chạy mỗi cấu hình:").grid(row=15, column=0, sticky=tk.W, padx=5, pady=3)
        self.parameter_vars["num_runs"] = tk.IntVar(value=3)
        ttk.Entry(self.parameter_frame, textvariable=self.parameter_vars["num_runs"], width=10).grid(row=15, column=1, padx=5, pady=3)
        
    def update_configs_list(self):
        """Cập nhật danh sách cấu hình trong listbox"""
        self.configs_list.delete(0, tk.END)  # Xóa tất cả các mục hiện tại
        
        if self.algorithm.get() == "ACO":
            for i, config in enumerate(self.aco_test_configs):
                config_name = f"ACO {i+1}: {self.format_config_string(config)}"
                self.configs_list.insert(tk.END, config_name)
        else:  # GA
            for i, config in enumerate(self.ga_test_configs):
                config_name = f"GA {i+1}: {self.format_config_string(config)}"
                self.configs_list.insert(tk.END, config_name)
    
    def format_config_string(self, config):
        """Định dạng chuỗi hiển thị cho cấu hình"""
        if self.algorithm.get() == "ACO":
            highlights = ['alpha', 'beta', 'min_max_aco', 'local_search', 'elitist_ants']
            return ", ".join([f"{k}={v}" for k, v in config.items() if k in highlights])
        else:  # GA
            highlights = ['selection_method', 'crossover_method', 'mutation_method', 'mutation_rate', 'crossover_rate', 'local_search', 'elitism']
            return ", ".join([f"{k}={v}" for k, v in config.items() if k in highlights])
    
    def add_current_config(self):
        """Thêm cấu hình hiện tại vào danh sách thử nghiệm"""
        if not self.parameter_vars:
            messagebox.showerror("Lỗi", "Không có tham số nào được cấu hình")
            return
            
        # Lấy tham số hiện tại (loại bỏ số lần chạy)
        config = {}
        for param, var in self.parameter_vars.items():
            if param != "num_runs":
                config[param] = var.get()
        
        # Thêm vào danh sách tương ứng
        if self.algorithm.get() == "ACO":
            self.aco_test_configs.append(config)
        else:  # GA
            self.ga_test_configs.append(config)
            
        # Cập nhật hiển thị
        self.update_configs_list()
        messagebox.showinfo("Thành công", "Đã thêm cấu hình vào danh sách thử nghiệm")
    
    def remove_selected_config(self):
        """Xóa cấu hình đã chọn khỏi danh sách"""
        selected_indices = self.configs_list.curselection()
        if not selected_indices:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn cấu hình để xóa")
            return
            
        index = selected_indices[0]
        if self.algorithm.get() == "ACO":
            if 0 <= index < len(self.aco_test_configs):
                del self.aco_test_configs[index]
        else:  # GA
            if 0 <= index < len(self.ga_test_configs):
                del self.ga_test_configs[index]
                
        # Cập nhật hiển thị
        self.update_configs_list()
    
    def clear_configs(self):
        """Xóa tất cả cấu hình trong danh sách"""
        if self.algorithm.get() == "ACO":
            self.aco_test_configs = []
        else:  # GA
            self.ga_test_configs = []
            
        # Cập nhật hiển thị
        self.update_configs_list()
        
    def add_multiple_configs(self):
        """Thêm nhiều cấu hình dựa trên tổ hợp các tham số"""
        if self.algorithm.get() == "ACO":
            self.add_multiple_aco_configs()
        else:  # GA
            self.add_multiple_ga_configs()
    
    def add_multiple_aco_configs(self):
        """Thêm nhiều cấu hình ACO"""
        # Cửa sổ tham số
        config_window = tk.Toplevel(self)
        config_window.title("Tạo nhiều cấu hình ACO")
        config_window.geometry("500x400")
        
        # Tham số cơ bản giữ nguyên
        ttk.Label(config_window, text="Tham số cơ bản giữ nguyên theo cấu hình hiện tại", font=('Helvetica', 10, 'bold')).pack(pady=5)
        
        # Tạo các danh sách tham số để thử nghiệm
        ttk.Label(config_window, text="Alpha (tầm quan trọng pheromone):", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        alpha_entry = ttk.Entry(config_window)
        alpha_entry.pack(fill=tk.X, padx=10, pady=2)
        alpha_entry.insert(0, "1.0, 2.0, 3.0")
        
        ttk.Label(config_window, text="Beta (tầm quan trọng heuristic):", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        beta_entry = ttk.Entry(config_window)
        beta_entry.pack(fill=tk.X, padx=10, pady=2)
        beta_entry.insert(0, "2.0, 3.0, 5.0")
        
        ttk.Label(config_window, text="MIN-MAX ACO:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        min_max_var = tk.StringVar(value="True, False")
        min_max_entry = ttk.Entry(config_window, textvariable=min_max_var)
        min_max_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(config_window, text="Tìm kiếm cục bộ:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        local_search_var = tk.StringVar(value="True, False")
        local_search_entry = ttk.Entry(config_window, textvariable=local_search_var)
        local_search_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(config_window, text="Số kiến ưu tú:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        elitist_var = tk.StringVar(value="0, 2, 5")
        elitist_entry = ttk.Entry(config_window, textvariable=elitist_var)
        elitist_entry.pack(fill=tk.X, padx=10, pady=2)
        
        def create_configs():
            try:
                # Lấy giá trị cơ bản hiện tại
                base_config = {}
                for param, var in self.parameter_vars.items():
                    if param != "num_runs":
                        base_config[param] = var.get()
                
                # Phân tích giá trị tham số
                alpha_values = [float(x.strip()) for x in alpha_entry.get().split(",")]
                beta_values = [float(x.strip()) for x in beta_entry.get().split(",")]
                
                min_max_values = []
                for x in min_max_var.get().split(","):
                    if x.strip().lower() == "true":
                        min_max_values.append(True)
                    elif x.strip().lower() == "false":
                        min_max_values.append(False)
                
                local_search_values = []
                for x in local_search_var.get().split(","):
                    if x.strip().lower() == "true":
                        local_search_values.append(True)
                    elif x.strip().lower() == "false":
                        local_search_values.append(False)
                
                elitist_values = [int(x.strip()) for x in elitist_var.get().split(",")]
                
                # Tạo tổ hợp
                configs_count = 0
                for alpha in alpha_values:
                    for beta in beta_values:
                        for min_max in min_max_values:
                            for local_search in local_search_values:
                                for elitist in elitist_values:
                                    new_config = base_config.copy()
                                    new_config["alpha"] = alpha
                                    new_config["beta"] = beta
                                    new_config["min_max_aco"] = min_max
                                    new_config["local_search"] = local_search
                                    new_config["elitist_ants"] = elitist
                                    
                                    # Thêm cấu hình mới
                                    self.aco_test_configs.append(new_config)
                                    configs_count += 1
                
                # Cập nhật hiển thị
                self.update_configs_list()
                messagebox.showinfo("Thành công", f"Đã thêm {configs_count} cấu hình ACO")
                config_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tạo cấu hình: {str(e)}")
        
        ttk.Button(config_window, text="Tạo cấu hình", command=create_configs).pack(pady=10)
        ttk.Label(config_window, text="Lưu ý: Nhập các giá trị tham số cách nhau bởi dấu phẩy").pack(pady=5)
    
    def add_multiple_ga_configs(self):
        """Thêm nhiều cấu hình GA"""
        # Cửa sổ tham số
        config_window = tk.Toplevel(self)
        config_window.title("Tạo nhiều cấu hình GA")
        config_window.geometry("500x500")
        
        # Tham số cơ bản giữ nguyên
        ttk.Label(config_window, text="Tham số cơ bản giữ nguyên theo cấu hình hiện tại", font=('Helvetica', 10, 'bold')).pack(pady=5)
        
        # Tạo các danh sách tham số để thử nghiệm
        ttk.Label(config_window, text="Phương pháp chọn lọc:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        selection_var = tk.StringVar(value="tournament, roulette, rank")
        selection_entry = ttk.Entry(config_window, textvariable=selection_var)
        selection_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(config_window, text="Phương pháp lai ghép:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        crossover_var = tk.StringVar(value="ordered, partially_mapped, cycle")
        crossover_entry = ttk.Entry(config_window, textvariable=crossover_var)
        crossover_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(config_window, text="Phương pháp đột biến:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        mutation_var = tk.StringVar(value="swap, insert, inversion, scramble")
        mutation_entry = ttk.Entry(config_window, textvariable=mutation_var)
        mutation_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(config_window, text="Xác suất đột biến:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        mutation_rate_var = tk.StringVar(value="0.05, 0.1, 0.2")
        mutation_rate_entry = ttk.Entry(config_window, textvariable=mutation_rate_var)
        mutation_rate_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(config_window, text="Xác suất lai ghép:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        crossover_rate_var = tk.StringVar(value="0.7, 0.8, 0.9")
        crossover_rate_entry = ttk.Entry(config_window, textvariable=crossover_rate_var)
        crossover_rate_entry.pack(fill=tk.X, padx=10, pady=2)
        
        # Thêm tùy chọn tìm kiếm cục bộ
        ttk.Label(config_window, text="Tìm kiếm cục bộ:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        local_search_var = tk.StringVar(value="True, False")
        local_search_entry = ttk.Entry(config_window, textvariable=local_search_var)
        local_search_entry.pack(fill=tk.X, padx=10, pady=2)
        
        # Thêm tùy chọn cho số cá thể ưu tú
        ttk.Label(config_window, text="Số cá thể ưu tú:", anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        elitism_var = tk.StringVar(value="1, 3, 5, 10")
        elitism_entry = ttk.Entry(config_window, textvariable=elitism_var)
        elitism_entry.pack(fill=tk.X, padx=10, pady=2)
        
        def create_configs():
            try:
                # Lấy giá trị cơ bản hiện tại
                base_config = {}
                for param, var in self.parameter_vars.items():
                    if param != "num_runs":
                        base_config[param] = var.get()
                
                # Phân tích giá trị tham số
                selection_values = [x.strip() for x in selection_var.get().split(",")]
                crossover_values = [x.strip() for x in crossover_var.get().split(",")]
                mutation_values = [x.strip() for x in mutation_var.get().split(",")]
                mutation_rate_values = [float(x.strip()) for x in mutation_rate_var.get().split(",")]
                crossover_rate_values = [float(x.strip()) for x in crossover_rate_var.get().split(",")]
                
                # Phân tích các giá trị tìm kiếm cục bộ
                local_search_values = []
                for x in local_search_var.get().split(","):
                    if x.strip().lower() == "true":
                        local_search_values.append(True)
                    elif x.strip().lower() == "false":
                        local_search_values.append(False)
                
                # Phân tích các giá trị số cá thể ưu tú
                elitism_values = [int(x.strip()) for x in elitism_var.get().split(",")]
                
                # Tạo tổ hợp
                configs_count = 0
                for selection in selection_values:
                    for crossover in crossover_values:
                        for mutation in mutation_values:
                            for mutation_rate in mutation_rate_values:
                                for crossover_rate in crossover_rate_values:
                                    for local_search in local_search_values:
                                        for elitism in elitism_values:
                                            new_config = base_config.copy()
                                            new_config["selection_method"] = selection
                                            new_config["crossover_method"] = crossover
                                            new_config["mutation_method"] = mutation
                                            new_config["mutation_rate"] = mutation_rate
                                            new_config["crossover_rate"] = crossover_rate
                                            new_config["local_search"] = local_search
                                            new_config["elitism"] = elitism
                                            
                                            # Điều chỉnh tournament_size nếu cần
                                            if selection != "tournament":
                                                new_config["tournament_size"] = 3  # Giá trị mặc định, không sử dụng
                                            
                                            # Kiểm tra elitism không vượt quá kích thước quần thể
                                            population_size = new_config.get("population_size", 50)
                                            if elitism >= population_size:
                                                new_config["elitism"] = population_size - 1
                                                print(f"Cảnh báo: Giảm số cá thể ưu tú từ {elitism} xuống {population_size - 1} " +
                                                      f"vì không thể vượt quá kích thước quần thể ({population_size})")
                                            
                                            # Thêm cấu hình mới
                                            self.ga_test_configs.append(new_config)
                                            configs_count += 1
                
                # Cập nhật hiển thị
                self.update_configs_list()
                messagebox.showinfo("Thành công", f"Đã thêm {configs_count} cấu hình GA")
                config_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tạo cấu hình: {str(e)}")
        
        ttk.Button(config_window, text="Tạo cấu hình", command=create_configs).pack(pady=10)
        ttk.Label(config_window, text="Lưu ý: Nhập các giá trị tham số cách nhau bởi dấu phẩy").pack(pady=5)
    
    def create_problem(self):
        try:
            num_customers = self.num_customers.get()
            vehicle_capacity = self.vehicle_capacity.get()
            problem_seed = self.problem_seed.get()
            
            if num_customers <= 0 or vehicle_capacity <= 0:
                messagebox.showerror("Lỗi", "Số khách hàng và tải trọng xe phải lớn hơn 0")
                return
                
            self.cvrp = CVRP(capacity=vehicle_capacity)
            self.cvrp.load_problem(num_customers, vehicle_capacity, seed=problem_seed)
            
            messagebox.showinfo("Thành công", f"Đã tạo bài toán CVRP với {num_customers} khách hàng và tải trọng {vehicle_capacity}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tạo bài toán: {str(e)}")
    
    def load_problem(self):
        try:
            filename = filedialog.askopenfilename(
                title="Chọn file bài toán CVRP",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not filename:
                return
                
            self.cvrp = CVRP()
            if self.cvrp.load_from_file(filename):
                self.num_customers.set(len(self.cvrp.customers) - 1)  # Trừ depot
                self.vehicle_capacity.set(self.cvrp.capacity)
                messagebox.showinfo("Thành công", f"Đã tải bài toán CVRP với {len(self.cvrp.customers) - 1} khách hàng và tải trọng {self.cvrp.capacity}")
            else:
                messagebox.showerror("Lỗi", "Không thể tải file bài toán")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải bài toán: {str(e)}")
    
    def run_experiment(self):
        if self.running:
            messagebox.showwarning("Đang thực thi", "Một thử nghiệm đang chạy, vui lòng đợi hoặc dừng nó trước")
            return
            
        if not self.cvrp:
            messagebox.showerror("Lỗi", "Vui lòng tạo hoặc tải bài toán CVRP trước")
            return
            
        # Lấy danh sách cấu hình dựa trên thuật toán hiện tại
        configs = []
        if self.algorithm.get() == "ACO":
            configs = self.aco_test_configs
        else:
            configs = self.ga_test_configs
            
        if not configs:
            messagebox.showerror("Lỗi", "Vui lòng thêm ít nhất một cấu hình để thử nghiệm")
            return
            
        # Lấy số lần chạy cho mỗi cấu hình
        try:
            num_runs = self.parameter_vars["num_runs"].get()
            if num_runs <= 0:
                messagebox.showerror("Lỗi", "Số lần chạy phải lớn hơn 0")
                return
        except:
            messagebox.showerror("Lỗi", "Số lần chạy không hợp lệ")
            return
            
        # Xóa danh sách kết quả cũ
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # Khởi chạy luồng thử nghiệm
        self.running = True
        self.results = []
        experiment_thread = threading.Thread(target=self.run_experiment_thread, args=(configs, num_runs))
        experiment_thread.daemon = True
        experiment_thread.start()
    
    def run_experiment_thread(self, configs, num_runs):
        try:
            # Tổng số cấu hình
            total_configs = len(configs)
            
            # Lưu kết quả tổng hợp
            all_costs_history = []
            all_avg_costs_history = []
            all_iterations_history = []
            all_config_names = []
            
            for config_idx, config in enumerate(configs):
                if not self.running:
                    break
                    
                algorithm_type = self.algorithm.get()
                self.current_experiment = {
                    'algorithm': algorithm_type,
                    'config': config,
                    'results': []
                }
                
                config_name = f"{algorithm_type} {config_idx+1}"
                all_config_names.append(config_name)
                
                # Thông báo cấu hình hiện tại
                self.update_status(f"Đang chạy {algorithm_type} cấu hình {config_idx+1}/{total_configs}: {self.format_config_string(config)}")
                
                # Kiểm tra và khắc phục tham số cấu hình
                try:
                    # Tạo bản sao cấu hình để tránh thay đổi gốc
                    fixed_config = config.copy()
                    
                    # Kiểm tra phạm vi giá trị
                    if algorithm_type == "ACO":
                        if "num_ants" in fixed_config and (fixed_config["num_ants"] <= 0 or fixed_config["num_ants"] > 1000):
                            print(f"Cảnh báo: num_ants={fixed_config['num_ants']} nằm ngoài phạm vi hợp lệ (1-1000)")
                            fixed_config["num_ants"] = max(1, min(fixed_config["num_ants"], 1000))
                        if "alpha" in fixed_config and fixed_config["alpha"] < 0:
                            print(f"Cảnh báo: alpha={fixed_config['alpha']} không được âm")
                            fixed_config["alpha"] = max(0, fixed_config["alpha"])
                        if "beta" in fixed_config and fixed_config["beta"] < 0:
                            print(f"Cảnh báo: beta={fixed_config['beta']} không được âm")
                            fixed_config["beta"] = max(0, fixed_config["beta"])
                        if "rho" in fixed_config and (fixed_config["rho"] <= 0 or fixed_config["rho"] >= 1):
                            print(f"Cảnh báo: rho={fixed_config['rho']} nằm ngoài phạm vi hợp lệ (0-1)")
                            fixed_config["rho"] = max(0.001, min(fixed_config["rho"], 0.999))
                        if "q" in fixed_config and fixed_config["q"] <= 0:
                            print(f"Cảnh báo: q={fixed_config['q']} phải dương")
                            fixed_config["q"] = max(1, fixed_config["q"])
                        if "max_iterations" in fixed_config and fixed_config["max_iterations"] <= 0:
                            print(f"Cảnh báo: max_iterations={fixed_config['max_iterations']} phải dương")
                            fixed_config["max_iterations"] = max(1, fixed_config["max_iterations"])
                        if "elitist_ants" in fixed_config and fixed_config["elitist_ants"] < 0:
                            print(f"Cảnh báo: elitist_ants={fixed_config['elitist_ants']} không được âm")
                            fixed_config["elitist_ants"] = max(0, fixed_config["elitist_ants"])
                    else:  # GA
                        if "population_size" in fixed_config and fixed_config["population_size"] <= 0:
                            print(f"Cảnh báo: population_size={fixed_config['population_size']} phải dương")
                            fixed_config["population_size"] = max(10, fixed_config["population_size"])
                        if "mutation_rate" in fixed_config and (fixed_config["mutation_rate"] < 0 or fixed_config["mutation_rate"] > 1):
                            print(f"Cảnh báo: mutation_rate={fixed_config['mutation_rate']} nằm ngoài phạm vi hợp lệ (0-1)")
                            fixed_config["mutation_rate"] = max(0, min(fixed_config["mutation_rate"], 1))
                        if "crossover_rate" in fixed_config and (fixed_config["crossover_rate"] < 0 or fixed_config["crossover_rate"] > 1):
                            print(f"Cảnh báo: crossover_rate={fixed_config['crossover_rate']} nằm ngoài phạm vi hợp lệ (0-1)")
                            fixed_config["crossover_rate"] = max(0, min(fixed_config["crossover_rate"], 1))
                        if "elitism" in fixed_config and (fixed_config["elitism"] < 0):
                            print(f"Cảnh báo: elitism={fixed_config['elitism']} không được âm")
                            fixed_config["elitism"] = max(0, fixed_config["elitism"])
                        if "elitism" in fixed_config and "population_size" in fixed_config and fixed_config["elitism"] >= fixed_config["population_size"]:
                            print(f"Cảnh báo: elitism={fixed_config['elitism']} phải nhỏ hơn population_size={fixed_config['population_size']}")
                            fixed_config["elitism"] = max(0, min(fixed_config["elitism"], fixed_config["population_size"] - 1))
                        if "tournament_size" in fixed_config and fixed_config["tournament_size"] <= 0:
                            print(f"Cảnh báo: tournament_size={fixed_config['tournament_size']} phải dương")
                            fixed_config["tournament_size"] = max(2, fixed_config["tournament_size"])
                        if "selection_method" in fixed_config and fixed_config["selection_method"] not in ["tournament", "roulette", "rank"]:
                            print(f"Cảnh báo: selection_method={fixed_config['selection_method']} không hợp lệ")
                            fixed_config["selection_method"] = "tournament"
                        if "crossover_method" in fixed_config and fixed_config["crossover_method"] not in ["ordered", "partially_mapped", "cycle"]:
                            print(f"Cảnh báo: crossover_method={fixed_config['crossover_method']} không hợp lệ")
                            fixed_config["crossover_method"] = "ordered"
                        if "mutation_method" in fixed_config and fixed_config["mutation_method"] not in ["swap", "insert", "inversion", "scramble"]:
                            print(f"Cảnh báo: mutation_method={fixed_config['mutation_method']} không hợp lệ")
                            fixed_config["mutation_method"] = "swap"
                        
                        # Xử lý trường hợp đặc biệt: partially_mapped có vấn đề với scramble
                        if fixed_config.get("crossover_method") == "partially_mapped" and fixed_config.get("mutation_method") == "scramble":
                            print(f"Cảnh báo: Phát hiện tổ hợp không tương thích: crossover_method=partially_mapped và mutation_method=scramble")
                            print(f"Đang chuyển sang phương pháp đột biến an toàn hơn: swap")
                            fixed_config["mutation_method"] = "swap"
                            
                    # Nếu đã sửa đổi cấu hình, thông báo
                    if fixed_config != config:
                        print(f"Đã sửa đổi cấu hình {config_idx+1} để đảm bảo các tham số hợp lệ")
                        print(f"Cấu hình gốc: {config}")
                        print(f"Cấu hình đã sửa: {fixed_config}")
                except Exception as e:
                    print(f"Lỗi khi kiểm tra tham số cấu hình {config_idx+1}: {str(e)}")
                    fixed_config = config.copy()  # Sử dụng cấu hình gốc nếu có lỗi
                
                # Chạy nhiều lần với cấu hình hiện tại
                all_best_costs = []
                all_avg_costs = []
                all_times = []
                costs_history = []
                avg_costs_history = []
                iterations_history = []
                
                for run in range(num_runs):
                    if not self.running:
                        break
                        
                    self.update_status(f"Đang chạy {algorithm_type} cấu hình {config_idx+1}/{total_configs}, lần {run+1}/{num_runs}")
                    
                    try:
                        # Tạo và chạy thuật toán với cấu hình đã sửa đổi
                        if algorithm_type == "ACO":
                            # Tạo thuật toán ACO với cấu hình
                            algorithm = ACO_CVRP(self.cvrp, **fixed_config)
                            
                            # Lưu dữ liệu callback
                            iteration_data = []
                            
                            def step_callback(data):
                                try:
                                    # Kiểm tra dữ liệu callback hợp lệ
                                    if 'iteration' not in data or 'best_cost' not in data or 'avg_cost' not in data:
                                        print(f"Cảnh báo: Dữ liệu callback ACO không đầy đủ: {data.keys()}")
                                        return
                                    
                                    # Kiểm tra giá trị
                                    if not isinstance(data['best_cost'], (int, float)) or not isinstance(data['avg_cost'], (int, float)):
                                        print(f"Cảnh báo: Chi phí không phải là số: best_cost={data['best_cost']}, avg_cost={data['avg_cost']}")
                                        return
                                    
                                    # Thêm dữ liệu vào lịch sử
                                    iteration_data.append({
                                        'iteration': data['iteration'],
                                        'best_cost': data['best_cost'],
                                        'avg_cost': data['avg_cost'],
                                        'time': data['computation_time'] if 'computation_time' in data else 0
                                    })
                                except Exception as e:
                                    print(f"Lỗi trong callback ACO: {str(e)}")
                            
                            start_time = time.time()
                            try:
                                # Bắt lỗi chi tiết trong quá trình chạy thuật toán
                                best_solution, best_cost = algorithm.run(step_callback=step_callback)
                            except Exception as specific_e:
                                print(f"Lỗi cụ thể khi chạy ACO: {str(specific_e)}")
                                import traceback
                                traceback.print_exc()
                                continue  # Bỏ qua lần chạy này nếu xảy ra lỗi
                            end_time = time.time()
                            
                        else:  # GA
                            # Tạo thuật toán GA với cấu hình
                            try:
                                algorithm = GeneticAlgorithm_CVRP(self.cvrp, **fixed_config)
                            except Exception as e:
                                print(f"Lỗi khi khởi tạo thuật toán GA: {str(e)}")
                                continue  # Bỏ qua lần chạy này nếu không thể khởi tạo thuật toán
                            
                            # Lưu dữ liệu callback
                            iteration_data = []
                            
                            def step_callback(data):
                                try:
                                    # Kiểm tra dữ liệu callback hợp lệ
                                    if 'generation' not in data or 'best_cost' not in data or 'avg_cost' not in data:
                                        print(f"Cảnh báo: Dữ liệu callback GA không đầy đủ: {data.keys()}")
                                        return
                                    
                                    # Kiểm tra giá trị
                                    if not isinstance(data['best_cost'], (int, float)) or not isinstance(data['avg_cost'], (int, float)):
                                        print(f"Cảnh báo: Chi phí không phải là số: best_cost={data['best_cost']}, avg_cost={data['avg_cost']}")
                                        return
                                    
                                    # Thêm dữ liệu vào lịch sử
                                    iteration_data.append({
                                        'iteration': data['generation'],
                                        'best_cost': data['best_cost'],
                                        'avg_cost': data['avg_cost'],
                                        'time': data['computation_time'] if 'computation_time' in data else 0
                                    })
                                except Exception as e:
                                    print(f"Lỗi trong callback GA: {str(e)}")
                            
                            start_time = time.time()
                            try:
                                # Bắt lỗi chi tiết trong quá trình chạy thuật toán
                                best_solution, best_cost = algorithm.run(step_callback=step_callback)
                            except Exception as specific_e:
                                print(f"Lỗi cụ thể khi chạy GA: {str(specific_e)}")
                                import traceback
                                traceback.print_exc()
                                continue  # Bỏ qua lần chạy này nếu xảy ra lỗi
                            end_time = time.time()
                        
                        # Kiểm tra kết quả
                        if best_solution is None or best_cost is None or not isinstance(best_cost, (int, float)):
                            print(f"Lỗi: Kết quả không hợp lệ: best_solution={(type(best_solution))} best_cost={best_cost}")
                            continue
                            
                        # Kiểm tra chi phí bất thường
                        if best_cost <= 0 or best_cost > 10000:
                            print(f"Cảnh báo: Chi phí bất thường: {best_cost}")
                        
                        # Lưu kết quả
                        total_time = end_time - start_time
                        
                        # Kiểm tra iteration_data trống
                        if not iteration_data:
                            print(f"Cảnh báo: Không có dữ liệu lặp cho cấu hình {config_idx+1}, lần chạy {run+1}")
                            continue
                        
                        # Tính toán các giá trị thống kê
                        costs = [data['best_cost'] for data in iteration_data]
                        avg_costs = [data['avg_cost'] for data in iteration_data]
                        iterations = [data['iteration'] for data in iteration_data]
                        
                        costs_history.append(costs)
                        avg_costs_history.append(avg_costs)
                        iterations_history.append(iterations)
                        all_best_costs.append(best_cost)
                        all_avg_costs.append(np.mean(avg_costs))
                        all_times.append(total_time)
                        
                        print(f"Hoàn thành cấu hình {config_idx+1}, lần chạy {run+1}: chi phí={best_cost:.2f}, thời gian={total_time:.2f}s")
                    except Exception as e:
                        print(f"Lỗi khi chạy cấu hình {config_idx+1}, lần chạy {run+1}: {str(e)}")
                        print(f"Tham số cấu hình: {fixed_config}")
                        # Tiếp tục với lần chạy tiếp theo
                        continue
                
                # Kiểm tra nếu không có dữ liệu hợp lệ
                if not all_best_costs:
                    print(f"Cảnh báo: Không có dữ liệu hợp lệ cho cấu hình {config_idx+1}")
                    continue
                
                # Thêm vào lịch sử chung cho biểu đồ so sánh
                all_costs_history.append(costs_history)
                all_avg_costs_history.append(avg_costs_history)
                all_iterations_history.append(iterations_history)
                
                # Lưu kết quả tổng hợp cho mỗi cấu hình
                if self.running:
                    # Tính toán thống kê
                    avg_best_cost = np.mean(all_best_costs)
                    min_best_cost = np.min(all_best_costs)
                    std_best_cost = np.std(all_best_costs)
                    avg_time = np.mean(all_times)
                    
                    result = {
                        'algorithm': algorithm_type,
                        'config': fixed_config,  # Lưu cấu hình đã sửa đổi
                        'original_config': config,  # Lưu cấu hình gốc
                        'config_idx': config_idx + 1,
                        'num_runs': num_runs,
                        'best_costs': all_best_costs,
                        'avg_costs': all_avg_costs,
                        'times': all_times,
                        'avg_best_cost': avg_best_cost,
                        'min_best_cost': min_best_cost,
                        'std_best_cost': std_best_cost,
                        'avg_time': avg_time,
                        'costs_history': costs_history,
                        'avg_costs_history': avg_costs_history,
                        'iterations_history': iterations_history
                    }
                    
                    self.results.append(result)
                    
                    # Cập nhật bảng kết quả
                    self.update_result_table(config_idx+1, algorithm_type, fixed_config, avg_best_cost, np.mean(all_avg_costs), avg_time)
                    
                    # Cập nhật biểu đồ sau mỗi cấu hình
                    try:
                        self.update_comparison_chart(all_costs_history, all_iterations_history, all_config_names)
                    except Exception as e:
                        print(f"Lỗi khi cập nhật biểu đồ: {str(e)}")
            
            # Hoàn thành thử nghiệm
            if self.running:
                # Hiển thị kết quả so sánh
                try:
                    self.update_comparison_chart(all_costs_history, all_iterations_history, all_config_names)
                except Exception as e:
                    print(f"Lỗi khi cập nhật biểu đồ kết quả: {str(e)}")
                
                summary = f"Đã hoàn thành thử nghiệm {total_configs} cấu hình.\n"
                if self.results:
                    best_result = min(self.results, key=lambda x: x['avg_best_cost'])
                    summary += (
                        f"Cấu hình tốt nhất: {self.algorithm.get()} {best_result['config_idx']}\n"
                        f"Chi phí tốt nhất trung bình: {best_result['avg_best_cost']:.2f}\n"
                        f"Thời gian trung bình: {best_result['avg_time']:.2f}s\n"
                        f"Tham số: {self.format_config_string(best_result['config'])}"
                    )
                
                self.update_status(summary)
                messagebox.showinfo("Hoàn thành", summary)
                
        except Exception as e:
            error_msg = f"Lỗi trong quá trình thử nghiệm: {str(e)}"
            print(error_msg)
            # In thêm thông tin về stack trace
            import traceback
            traceback.print_exc()
            messagebox.showerror("Lỗi", error_msg)
        finally:
            self.running = False
    
    def update_status(self, message):
        # Cập nhật trên luồng giao diện
        self.after(0, lambda: self.title(f"Phần mềm Thử nghiệm Tham số CVRP - {message}"))
    
    def update_comparison_chart(self, all_costs_history, all_iterations_history, config_names):
        # Cập nhật biểu đồ trên luồng giao diện
        self.after(0, lambda: self._draw_comparison_chart(all_costs_history, all_iterations_history, config_names))
    
    def _draw_comparison_chart(self, all_costs_history, all_iterations_history, config_names):
        # Vẽ biểu đồ so sánh các cấu hình
        try:
            self.fig.clear()
            
            if not all_costs_history:
                return
            
            # Biểu đồ chi phí tốt nhất theo vòng lặp/thế hệ
            ax1 = self.fig.add_subplot(111)
            
            # Vẽ đường chi phí tốt nhất cho mỗi cấu hình
            for i, (costs_history, iterations_history, config_name) in enumerate(zip(all_costs_history, all_iterations_history, config_names)):
                # Lấy lần chạy đầu tiên của mỗi cấu hình để biểu diễn
                if costs_history and iterations_history and len(costs_history) > 0 and len(iterations_history) > 0:
                    costs = costs_history[0]  # Lần chạy đầu tiên
                    iterations = iterations_history[0]  # Lần chạy đầu tiên
                    if len(costs) > 0 and len(iterations) > 0 and len(costs) == len(iterations):
                        # Vẽ đường không có legend
                        ax1.plot(iterations, costs)
            
            ax1.set_title('So sánh hội tụ giữa các cấu hình')
            ax1.set_xlabel('Vòng lặp/Thế hệ')
            ax1.set_ylabel('Chi phí tốt nhất')
            ax1.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ: {str(e)}")
    
    def update_result_table(self, config_idx, algorithm, params, avg_best_cost, avg_avg_cost, avg_time):
        # Cập nhật bảng trên luồng giao diện
        self.after(0, lambda: self._add_result_row(config_idx, algorithm, params, avg_best_cost, avg_avg_cost, avg_time))
    
    def _add_result_row(self, config_idx, algorithm, params, avg_best_cost, avg_avg_cost, avg_time):
        try:
            # Định dạng tham số
            params_str = self.format_config_string(params)
            
            # Chèn vào bảng
            self.result_tree.insert('', 'end', values=(
                config_idx,
                algorithm,
                params_str,
                f"{avg_best_cost:.2f}",
                f"{avg_avg_cost:.2f}",
                f"{avg_time:.2f}"
            ))
        except Exception as e:
            print(f"Lỗi khi cập nhật bảng kết quả: {str(e)}")
    
    def stop_experiment(self):
        self.running = False
        self.update_status("Đã dừng thử nghiệm")
    
    def save_results(self):
        if not self.results:
            messagebox.showwarning("Cảnh báo", "Không có kết quả nào để lưu")
            return
            
        try:
            # Tạo thư mục lưu nếu chưa tồn tại
            save_dir = "ket_qua_thu_nghiem_tham_so"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # Tạo tên file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/ket_qua_{self.algorithm.get()}_{timestamp}"
            
            # Lưu dữ liệu thô dạng JSON
            with open(f"{filename}.json", "w", encoding="utf-8") as f:
                # Chuyển đổi numpy arrays sang lists
                results_json = []
                for result in self.results:
                    result_copy = result.copy()
                    for key in result_copy:
                        if isinstance(result_copy[key], np.ndarray):
                            result_copy[key] = result_copy[key].tolist()
                        elif isinstance(result_copy[key], list) and any(isinstance(item, np.ndarray) for item in result_copy[key]):
                            result_copy[key] = [item.tolist() if isinstance(item, np.ndarray) else item for item in result_copy[key]]
                    results_json.append(result_copy)
                    
                json.dump(results_json, f, indent=2)
            
            # Lưu biểu đồ
            self.fig.savefig(f"{filename}.png", dpi=300)
            
            # Lưu bảng CSV
            with open(f"{filename}.csv", "w", newline='', encoding="utf-8") as f:
                f.write("ID,Thuật toán,Tham số,Chi phí tốt nhất,Chi phí trung bình,Thời gian (s)\n")
                for item in self.result_tree.get_children():
                    values = self.result_tree.item(item)['values']
                    f.write(",".join([str(v) for v in values]) + "\n")
            
            messagebox.showinfo("Thành công", f"Đã lưu kết quả vào:\n{filename}.json\n{filename}.png\n{filename}.csv")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu kết quả: {str(e)}")


if __name__ == "__main__":
    app = ParameterTester()
    app.mainloop()
