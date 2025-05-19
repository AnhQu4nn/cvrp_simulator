"""
Ứng dụng So sánh Thuật toán
Giao diện đồ họa để so sánh và phân tích hai thuật toán ACO và GA trong bài toán CVRP
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import multiprocessing
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import os
import datetime
import psutil  # Thêm thư viện để quản lý tài nguyên hệ thống

from core import CVRP, ACO_CVRP, GeneticAlgorithm_CVRP
from .visualization import ACOVisualization, GeneticVisualization
from .tooltip import ToolTip

# Hằng số toàn cục để kiểm soát hiệu năng
MAX_CPU_USAGE_PERCENT = 95  # Giới hạn sử dụng CPU tối đa (%)
PROCESS_PRIORITY = psutil.HIGH_PRIORITY_CLASS  # Ưu tiên cao cho process

class ComparisonApp:
    """Ứng dụng so sánh các thuật toán giải CVRP"""

    def __init__(self, root, selector_root=None):
        """Khởi tạo ứng dụng so sánh"""
        self.root = root
        self.selector_root = selector_root  # Tham chiếu đến cửa sổ chọn thuật toán
        self.root.title("So sánh thuật toán giải CVRP")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Thiết lập ưu tiên process cho ứng dụng
        try:
            process = psutil.Process(os.getpid())
            process.nice(PROCESS_PRIORITY)
        except:
            pass

        # Lấy thông tin hệ thống
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_info = psutil.virtual_memory()
        self.total_memory_gb = self.memory_info.total / (1024**3)
        
        # Thiết lập multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
        
        # Khởi tạo biến cho CVRP
        self.cvrp = CVRP()
        self.n_customers = 15
        self.capacity = 100
        
        # Khởi tạo biến cho Ant Colony Optimization
        self.aco_n_ants = 20
        self.aco_alpha = 1.0
        self.aco_beta = 2.0
        self.aco_rho = 0.5
        self.aco_q = 100
        self.aco_iterations = 50
        self.aco_min_max = False
        self.aco_local_search = False
        self.aco_elitist_ants = 0
        
        # Khởi tạo biến cho Genetic Algorithm
        self.ga_population_size = 100
        self.ga_crossover_rate = 0.8
        self.ga_mutation_rate = 0.1
        self.ga_iterations = 50
        self.ga_tournament_size = 5
        self.ga_elite_size = 5
        self.ga_local_search = False
        self.ga_crossover_type = "partially_mapped"
        self.ga_mutation_type = "inversion"
        
        # Thuật toán và luồng
        self.aco_algorithm = None
        self.ga_algorithm = None
        self.aco_process = None  # Sử dụng process thay vì thread
        self.ga_process = None   # Sử dụng process thay vì thread
        self.aco_queue = None    # Queue để giao tiếp giữa process
        self.ga_queue = None     # Queue để giao tiếp giữa process
        self.is_running = False
        self.is_paused = False
        
        # Biến kiểm soát hoàn thành thuật toán
        self.aco_completed = False
        self.ga_completed = False
        self.aco_completion_time = None
        self.ga_completion_time = None
        
        # Thêm biến theo dõi thời gian tính toán thuần túy
        self.aco_pure_computation_time = 0
        self.ga_pure_computation_time = 0
        
        # Dữ liệu phân tích
        self.aco_convergence_data = []
        self.ga_convergence_data = []
        self.aco_best_solution = None
        self.ga_best_solution = None
        self.aco_best_cost = float('inf')
        self.ga_best_cost = float('inf')
        
        # Tạo giao diện
        self.create_gui()
        
        # Thiết lập mặc định
        self.select_random.invoke()
        
        # Báo cáo tài nguyên hệ thống
        messagebox.showinfo("Thông tin hệ thống", 
                           f"CPU: {self.cpu_count} cores\n"
                           f"RAM: {self.total_memory_gb:.2f} GB\n"
                           f"Ứng dụng sẽ tối ưu hóa để sử dụng tối đa tài nguyên này.")

    def create_gui(self):
        """Tạo giao diện người dùng"""
        # Notebook (tab container)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab chính
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="So sánh thuật toán")
        
        # Tab phân tích
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="Phân tích chi tiết")
        
        # Panel chính trên tab chính
        main_panel = ttk.PanedWindow(main_tab, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True)
        
        # Panel điều khiển bên trái với thanh cuộn
        control_container = ttk.Frame(main_panel)
        main_panel.add(control_container, weight=1)
        
        # Thêm thanh cuộn cho khung điều khiển
        control_scroll = ttk.Scrollbar(control_container, orient="vertical")
        control_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas để chứa các điều khiển và hỗ trợ cuộn
        control_canvas = tk.Canvas(control_container, yscrollcommand=control_scroll.set, 
                                  highlightthickness=0, bg=self.root.cget('bg'))
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_scroll.config(command=control_canvas.yview)
        
        # Frame thực sự chứa các điều khiển
        self.control_frame = ttk.LabelFrame(control_canvas, text="Điều khiển")
        control_window = control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw", 
                                                     tags="self.control_frame", width=control_container.winfo_width()-20)
        
        # Cập nhật chiều rộng của frame khi resize
        def on_container_configure(event):
            canvas_width = event.width - 20
            control_canvas.itemconfig("self.control_frame", width=canvas_width)
            
        control_container.bind("<Configure>", on_container_configure)
        
        # Cấu hình Canvas để cuộn khi kích thước của nội dung thay đổi
        def configure_scroll_region(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))
            
        self.control_frame.bind("<Configure>", configure_scroll_region)
        
        # Làm cho canvas có thể cuộn bằng chuột
        def _on_mousewheel(event):
            if str(event.widget).startswith(str(control_canvas)) or str(event.widget).startswith(str(self.control_frame)):
                control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_mousewheel(event):
            control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
        def _unbind_mousewheel(event):
            control_canvas.unbind_all("<MouseWheel>")
            
        control_canvas.bind("<Enter>", _bind_mousewheel)
        control_canvas.bind("<Leave>", _unbind_mousewheel)
        self.control_frame.bind("<Enter>", _bind_mousewheel)
        self.control_frame.bind("<Leave>", _unbind_mousewheel)
        
        # Panel trực quan hóa bên phải với thanh cuộn
        viz_container = ttk.Frame(main_panel)
        main_panel.add(viz_container, weight=3)
        
        # Thêm thanh cuộn cho khung trực quan hóa
        viz_scroll = ttk.Scrollbar(viz_container, orient="vertical")
        viz_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas để chứa các trực quan hóa và hỗ trợ cuộn
        viz_canvas = tk.Canvas(viz_container, yscrollcommand=viz_scroll.set,
                               highlightthickness=0, bg=self.root.cget('bg'))
        viz_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        viz_scroll.config(command=viz_canvas.yview)
        
        # Frame chứa các trực quan hóa
        viz_panel = ttk.Frame(viz_canvas)
        viz_window = viz_canvas.create_window((0, 0), window=viz_panel, anchor="nw",
                                              tags="viz_panel")
        
        # Cấu hình Canvas để cuộn khi kích thước của nội dung thay đổi
        def viz_configure_scroll_region(event):
            viz_canvas.configure(scrollregion=viz_canvas.bbox("all"))
            # Đảm bảo chiều rộng đầy đủ
            viz_canvas.itemconfig("viz_panel", width=viz_canvas.winfo_width())
            
        viz_panel.bind("<Configure>", viz_configure_scroll_region)
        
        # Làm cho canvas có thể cuộn bằng chuột
        def _viz_on_mousewheel(event):
            if str(event.widget).startswith(str(viz_canvas)) or str(event.widget).startswith(str(viz_panel)):
                viz_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _viz_bind_mousewheel(event):
            viz_canvas.bind_all("<MouseWheel>", _viz_on_mousewheel)
            
        def _viz_unbind_mousewheel(event):
            viz_canvas.unbind_all("<MouseWheel>")
            
        viz_canvas.bind("<Enter>", _viz_bind_mousewheel)
        viz_canvas.bind("<Leave>", _viz_unbind_mousewheel)
        viz_panel.bind("<Enter>", _viz_bind_mousewheel)
        viz_panel.bind("<Leave>", _viz_unbind_mousewheel)
        
        viz_container.bind("<Configure>", lambda e: viz_canvas.itemconfig("viz_panel", width=viz_canvas.winfo_width()))
        
        # Chia panel trực quan hóa thành hai phần: ACO và GA
        viz_top_panel = ttk.LabelFrame(viz_panel, text="Ant Colony Optimization")
        viz_top_panel.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        viz_bottom_panel = ttk.LabelFrame(viz_panel, text="Genetic Algorithm")
        viz_bottom_panel.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Tạo trực quan hóa cho cả hai thuật toán
        self.aco_visualization = ACOVisualization(viz_top_panel)
        self.ga_visualization = GeneticVisualization(viz_bottom_panel)
        
        # Tạo các phần điều khiển
        self.create_problem_generation_controls()
        self.create_algorithm_controls()
        self.create_execution_controls()
        self.create_result_display()
        
        # Tạo giao diện phân tích trên tab phân tích
        self.create_analysis_tab(analysis_tab)
        
        # Thêm nút quay lại
        self.add_return_button()

    def add_return_button(self):
        """Thêm nút quay lại màn hình chọn thuật toán"""
        return_frame = ttk.Frame(self.root)
        return_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        return_button = ttk.Button(return_frame, text="Quay lại màn hình chọn", command=self.back_to_selector)
        return_button.pack(side=tk.LEFT, padx=5)
        
        # Thông tin tác giả
        info_label = ttk.Label(return_frame, text="© 2025 - Nguyễn Anh Quân - Vũ Quốc Long - DAA", font=("Arial", 8))
        info_label.pack(side=tk.RIGHT, padx=5)
        
    def back_to_selector(self):
        """Quay lại màn hình chọn thuật toán"""
        if self.is_running:
            if messagebox.askyesno("Xác nhận", "Các thuật toán đang chạy. Bạn có chắc muốn hủy và quay lại?"):
                self.stop_algorithm()
                self.root.destroy()
                self.selector_root.deiconify()
        else:
            self.root.destroy()
            self.selector_root.deiconify()
            
    def create_problem_generation_controls(self):
        """Tạo các điều khiển cho việc tạo bài toán CVRP"""
        problem_frame = ttk.LabelFrame(self.control_frame, text="Tạo bài toán CVRP")
        problem_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Nhóm các radio button để chọn loại bài toán
        generation_frame = ttk.Frame(problem_frame)
        generation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        generation_type = tk.StringVar()
        
        self.select_random = ttk.Radiobutton(
            generation_frame, text="Bài toán ngẫu nhiên", variable=generation_type, value="random"
        )
        self.select_random.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.select_from_file = ttk.Radiobutton(
            generation_frame, text="Tải bài toán từ file", variable=generation_type, value="file"
        )
        self.select_from_file.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tham số cho bài toán ngẫu nhiên
        params_frame = ttk.Frame(problem_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Số điểm cần giao hàng:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.customer_count = ttk.Spinbox(params_frame, from_=5, to=100, increment=5, width=10)
        self.customer_count.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.customer_count.set(self.n_customers)
        
        ttk.Label(params_frame, text="Sức chứa xe:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.vehicle_capacity = ttk.Spinbox(params_frame, from_=50, to=500, increment=50, width=10)
        self.vehicle_capacity.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.vehicle_capacity.set(self.capacity)
        
        # Nút tạo bài toán
        button_frame = ttk.Frame(problem_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Tạo bài toán", command=self.generate_problem).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Lưu bài toán", command=self.save_problem).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Tải bài toán", command=self.load_problem).pack(side=tk.LEFT, padx=5)
        
    def create_algorithm_controls(self):
        """Tạo các điều khiển cho việc cài đặt tham số của cả hai thuật toán"""
        # Frame cho cả hai thuật toán
        algorithm_frame = ttk.LabelFrame(self.control_frame, text="Cài đặt thuật toán")
        algorithm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Tab cho từng thuật toán
        algo_notebook = ttk.Notebook(algorithm_frame)
        algo_notebook.pack(fill=tk.X, padx=5, pady=5)
        
        # Tab Ant Colony Optimization
        aco_tab = ttk.Frame(algo_notebook)
        algo_notebook.add(aco_tab, text="Ant Colony")
        
        # Tab Genetic Algorithm
        ga_tab = ttk.Frame(algo_notebook)
        algo_notebook.add(ga_tab, text="Genetic Algorithm")
        
        # Tham số cho ACO
        aco_params = ttk.LabelFrame(aco_tab, text="Tham số ACO")
        aco_params.pack(fill=tk.X, padx=5, pady=5)
        
        # Số lượng kiến
        ttk.Label(aco_params, text="Số lượng kiến:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_ant_count = ttk.Spinbox(aco_params, from_=5, to=100, increment=5, width=10)
        self.aco_ant_count.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.aco_ant_count.set(self.aco_n_ants)
        
        # Alpha (trọng số pheromone)
        ttk.Label(aco_params, text="Alpha (trọng số pheromone):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_alpha_value = ttk.Spinbox(aco_params, from_=0.1, to=5.0, increment=0.1, width=10)
        self.aco_alpha_value.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.aco_alpha_value.set(self.aco_alpha)
        
        # Beta (trọng số heuristic)
        ttk.Label(aco_params, text="Beta (trọng số heuristic):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_beta_value = ttk.Spinbox(aco_params, from_=0.1, to=10.0, increment=0.1, width=10)
        self.aco_beta_value.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.aco_beta_value.set(self.aco_beta)
        
        # Rho (tỷ lệ bay hơi)
        ttk.Label(aco_params, text="Rho (tỷ lệ bay hơi):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_rho_value = ttk.Spinbox(aco_params, from_=0.1, to=0.9, increment=0.1, width=10)
        self.aco_rho_value.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        self.aco_rho_value.set(self.aco_rho)
        
        # Q (hệ số cường độ pheromone)
        ttk.Label(aco_params, text="Q (hệ số pheromone):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_q_value = ttk.Spinbox(aco_params, from_=10, to=1000, increment=10, width=10)
        self.aco_q_value.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        self.aco_q_value.set(self.aco_q)
        
        # Số lượng vòng lặp
        ttk.Label(aco_params, text="Số vòng lặp:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_iteration_count = ttk.Spinbox(aco_params, from_=10, to=1000, increment=10, width=10)
        self.aco_iteration_count.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        self.aco_iteration_count.set(self.aco_iterations)
        
        # Tùy chọn nâng cao ACO
        aco_advanced = ttk.LabelFrame(aco_tab, text="Tùy chọn nâng cao")
        aco_advanced.pack(fill=tk.X, padx=5, pady=5)
        
        # Min-Max ACO
        self.aco_minmax_var = tk.BooleanVar(value=self.aco_min_max)
        ttk.Checkbutton(aco_advanced, text="Sử dụng Min-Max ACO", variable=self.aco_minmax_var).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        
        # Tìm kiếm cục bộ
        self.aco_localsearch_var = tk.BooleanVar(value=self.aco_local_search)
        ttk.Checkbutton(aco_advanced, text="Sử dụng tìm kiếm cục bộ", variable=self.aco_localsearch_var).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        
        # Số kiến ưu tú
        ttk.Label(aco_advanced, text="Số kiến ưu tú:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_elitist_count = ttk.Spinbox(aco_advanced, from_=0, to=20, increment=1, width=10)
        self.aco_elitist_count.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.aco_elitist_count.set(self.aco_elitist_ants)
        
        # Tham số cho GA
        ga_params = ttk.LabelFrame(ga_tab, text="Tham số GA")
        ga_params.pack(fill=tk.X, padx=5, pady=5)
        
        # Kích thước quần thể
        ttk.Label(ga_params, text="Kích thước quần thể:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_pop_size = ttk.Spinbox(ga_params, from_=20, to=500, increment=10, width=10)
        self.ga_pop_size.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.ga_pop_size.set(self.ga_population_size)
        
        # Tỷ lệ lai ghép
        ttk.Label(ga_params, text="Tỷ lệ lai ghép:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_crossover_prob = ttk.Spinbox(ga_params, from_=0.1, to=1.0, increment=0.1, width=10)
        self.ga_crossover_prob.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.ga_crossover_prob.set(self.ga_crossover_rate)
        
        # Tỷ lệ đột biến
        ttk.Label(ga_params, text="Tỷ lệ đột biến:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_mutation_prob = ttk.Spinbox(ga_params, from_=0.01, to=0.5, increment=0.01, width=10)
        self.ga_mutation_prob.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.ga_mutation_prob.set(self.ga_mutation_rate)
        
        # Số lượng vòng lặp
        ttk.Label(ga_params, text="Số thế hệ:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_generation_count = ttk.Spinbox(ga_params, from_=10, to=1000, increment=10, width=10)
        self.ga_generation_count.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        self.ga_generation_count.set(self.ga_iterations)
        
        # Kích thước giải đấu
        ttk.Label(ga_params, text="Kích thước giải đấu:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_tournament_size_value = ttk.Spinbox(ga_params, from_=2, to=10, increment=1, width=10)
        self.ga_tournament_size_value.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        self.ga_tournament_size_value.set(self.ga_tournament_size)
        
        # Số cá thể ưu tú
        ttk.Label(ga_params, text="Số cá thể ưu tú:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_elite_count = ttk.Spinbox(ga_params, from_=0, to=20, increment=1, width=10)
        self.ga_elite_count.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        self.ga_elite_count.set(self.ga_elite_size)
        
        # Tùy chọn nâng cao GA
        ga_advanced = ttk.LabelFrame(ga_tab, text="Tùy chọn nâng cao")
        ga_advanced.pack(fill=tk.X, padx=5, pady=5)
        
        # Loại lai ghép
        ttk.Label(ga_advanced, text="Loại lai ghép:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_crossover_type_var = tk.StringVar(value=self.ga_crossover_type)
        crossover_combo = ttk.Combobox(ga_advanced, textvariable=self.ga_crossover_type_var, width=15)
        crossover_combo['values'] = ('partially_mapped', 'ordered', 'cycle')
        crossover_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        crossover_combo.current(0)
        
        # Loại đột biến
        ttk.Label(ga_advanced, text="Loại đột biến:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_mutation_type_var = tk.StringVar(value=self.ga_mutation_type)
        mutation_combo = ttk.Combobox(ga_advanced, textvariable=self.ga_mutation_type_var, width=15)
        mutation_combo['values'] = ('inversion', 'swap', 'insert', 'scramble')
        mutation_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        mutation_combo.current(0)
        
        # Tìm kiếm cục bộ
        self.ga_localsearch_var = tk.BooleanVar(value=self.ga_local_search)
        ttk.Checkbutton(ga_advanced, text="Sử dụng tìm kiếm cục bộ", variable=self.ga_localsearch_var).grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        
    def create_execution_controls(self):
        """Tạo các điều khiển cho việc thực thi thuật toán"""
        execution_frame = ttk.LabelFrame(self.control_frame, text="Thực thi thuật toán")
        execution_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Thêm thông tin CPU và bộ nhớ
        system_frame = ttk.Frame(execution_frame)
        system_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(system_frame, text=f"CPU: {self.cpu_count} cores | RAM: {self.total_memory_gb:.2f} GB", 
                 font=("Arial", 9, "italic")).pack(side=tk.LEFT, padx=5)
        
        # Thêm options để tối ưu
        optimize_frame = ttk.Frame(execution_frame)
        optimize_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Checkbox sử dụng đa nhân
        self.use_multiprocessing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(optimize_frame, text=f"Dùng đa nhân ({self.cpu_count} cores)", 
                       variable=self.use_multiprocessing_var).pack(side=tk.LEFT, padx=5)
        
        # Mức độ ưu tiên CPU
        ttk.Label(optimize_frame, text="Mức ưu tiên CPU:").pack(side=tk.LEFT, padx=(10, 0))
        self.cpu_priority_var = tk.StringVar(value="high")
        priority_combo = ttk.Combobox(optimize_frame, textvariable=self.cpu_priority_var, width=10, state="readonly")
        priority_combo['values'] = ('low', 'normal', 'high', 'realtime')
        priority_combo.current(2)  # high by default
        priority_combo.pack(side=tk.LEFT, padx=5)
        
        # Nút điều khiển
        button_frame = ttk.Frame(execution_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Bắt đầu so sánh", command=self.start_comparison)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(button_frame, text="Tạm dừng", command=self.pause_algorithms, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Dừng lại", command=self.stop_algorithm, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Thanh tiến trình
        progress_frame = ttk.Frame(execution_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Tiến trình:").pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=200)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress_text_var = tk.StringVar(value="0%")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_text_var)
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
    def create_result_display(self):
        """Tạo phần hiển thị kết quả so sánh"""
        result_frame = ttk.LabelFrame(self.control_frame, text="Kết quả so sánh")
        result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Kết quả ACO
        aco_result_frame = ttk.LabelFrame(result_frame, text="Ant Colony Optimization")
        aco_result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(aco_result_frame, text="Chi phí tốt nhất:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_best_cost_var = tk.StringVar(value="---")
        ttk.Label(aco_result_frame, textvariable=self.aco_best_cost_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(aco_result_frame, text="Số xe sử dụng:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_vehicles_var = tk.StringVar(value="---")
        ttk.Label(aco_result_frame, textvariable=self.aco_vehicles_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(aco_result_frame, text="Thời gian thực thi:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.aco_time_var = tk.StringVar(value="---")
        ttk.Label(aco_result_frame, textvariable=self.aco_time_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Kết quả GA
        ga_result_frame = ttk.LabelFrame(result_frame, text="Genetic Algorithm")
        ga_result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ga_result_frame, text="Chi phí tốt nhất:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_best_cost_var = tk.StringVar(value="---")
        ttk.Label(ga_result_frame, textvariable=self.ga_best_cost_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(ga_result_frame, text="Số xe sử dụng:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_vehicles_var = tk.StringVar(value="---")
        ttk.Label(ga_result_frame, textvariable=self.ga_vehicles_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(ga_result_frame, text="Thời gian thực thi:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.ga_time_var = tk.StringVar(value="---")
        ttk.Label(ga_result_frame, textvariable=self.ga_time_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # So sánh kết quả
        comparison_frame = ttk.LabelFrame(result_frame, text="So sánh")
        comparison_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(comparison_frame, text="Chênh lệch chi phí:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.cost_diff_var = tk.StringVar(value="---")
        ttk.Label(comparison_frame, textvariable=self.cost_diff_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(comparison_frame, text="Chênh lệch thời gian:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.time_diff_var = tk.StringVar(value="---")
        ttk.Label(comparison_frame, textvariable=self.time_diff_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(comparison_frame, text="Thuật toán tốt hơn:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.better_algo_var = tk.StringVar(value="---")
        ttk.Label(comparison_frame, textvariable=self.better_algo_var, font=("Arial", 9, "bold")).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Nút xuất kết quả
        export_button = ttk.Button(result_frame, text="Xuất kết quả", command=self.export_results)
        export_button.pack(padx=5, pady=5)

    def create_analysis_tab(self, parent):
        """Tạo tab phân tích chi tiết"""
        # Chia tab thành các phần
        analysis_pane = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        analysis_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel trên cho biểu đồ hội tụ
        convergence_frame = ttk.LabelFrame(analysis_pane, text="So sánh hội tụ")
        analysis_pane.add(convergence_frame, weight=2)
        
        # Tạo vùng cho biểu đồ matplotlib
        self.fig_convergence = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_convergence = self.fig_convergence.add_subplot(111)
        self.canvas_convergence = FigureCanvasTkAgg(self.fig_convergence, convergence_frame)
        self.canvas_convergence.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_convergence.set_title("So sánh hội tụ của hai thuật toán")
        self.ax_convergence.set_xlabel("Vòng lặp")
        self.ax_convergence.set_ylabel("Chi phí tốt nhất")
        
        # Panel dưới cho chi tiết so sánh
        detail_frame = ttk.LabelFrame(analysis_pane, text="Chi tiết so sánh")
        analysis_pane.add(detail_frame, weight=1)
        
        # Tạo grid cho chi tiết
        detail_grid = ttk.Frame(detail_frame)
        detail_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Header
        ttk.Label(detail_grid, text="Tiêu chí", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(detail_grid, text="ACO", font=("Arial", 10, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(detail_grid, text="GA", font=("Arial", 10, "bold")).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(detail_grid, text="So sánh", font=("Arial", 10, "bold")).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Chi phí tốt nhất
        ttk.Label(detail_grid, text="Chi phí tốt nhất:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.detail_aco_best_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_aco_best_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.detail_ga_best_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_ga_best_var).grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.detail_best_diff_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_best_diff_var).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Thời gian thực hiện
        ttk.Label(detail_grid, text="Thời gian thực hiện:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.detail_aco_time_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_aco_time_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.detail_ga_time_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_ga_time_var).grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.detail_time_diff_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_time_diff_var).grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Tốc độ hội tụ
        ttk.Label(detail_grid, text="Tốc độ hội tụ:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.detail_aco_conv_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_aco_conv_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        self.detail_ga_conv_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_ga_conv_var).grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)
        self.detail_conv_diff_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_conv_diff_var).grid(row=3, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Độ ổn định
        ttk.Label(detail_grid, text="Độ ổn định:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.detail_aco_stab_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_aco_stab_var).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        self.detail_ga_stab_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_ga_stab_var).grid(row=4, column=2, sticky=tk.W, padx=5, pady=2)
        self.detail_stab_diff_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_stab_diff_var).grid(row=4, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Số xe sử dụng
        ttk.Label(detail_grid, text="Số xe sử dụng:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.detail_aco_veh_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_aco_veh_var).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        self.detail_ga_veh_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_ga_veh_var).grid(row=5, column=2, sticky=tk.W, padx=5, pady=2)
        self.detail_veh_diff_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_veh_diff_var).grid(row=5, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Đánh giá tổng quan
        ttk.Label(detail_grid, text="Đánh giá tổng quan:", font=("Arial", 10, "bold")).grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.detail_evaluation_var = tk.StringVar(value="---")
        ttk.Label(detail_grid, textvariable=self.detail_evaluation_var, wraplength=500).grid(row=6, column=1, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Nút xuất báo cáo phân tích
        ttk.Button(detail_frame, text="Xuất báo cáo phân tích", command=self.export_analysis).pack(side=tk.BOTTOM, padx=5, pady=5)
        
    def generate_problem(self):
        """Tạo bài toán CVRP ngẫu nhiên"""
        try:
            self.n_customers = int(self.customer_count.get())
            self.capacity = int(self.vehicle_capacity.get())
            
            # Đặt lại các biến giữ kết quả
            self.reset_results()
            
            # Tạo bài toán mới
            self.cvrp = CVRP(capacity=self.capacity)
            self.cvrp.load_problem(self.n_customers, self.capacity)
            
            # Hiển thị bài toán trên cả hai trực quan hóa
            self.aco_visualization.init_visualization(self.cvrp, self.aco_n_ants, self.aco_iterations)
            self.ga_visualization.init_visualization(self.cvrp, self.ga_population_size, self.ga_iterations)
            
            messagebox.showinfo("Thành công", f"Đã tạo bài toán CVRP với {self.n_customers} điểm và sức chứa xe {self.capacity}")
            
        except ValueError as e:
            messagebox.showerror("Lỗi", f"Lỗi khi tạo bài toán: {str(e)}")
    
    def reset_results(self):
        """Đặt lại kết quả và chuẩn bị cho phiên so sánh mới"""
        # Thiết lập lại dữ liệu phân tích
        self.aco_convergence_data = []
        self.ga_convergence_data = []
        self.aco_best_solution = None
        self.ga_best_solution = None
        self.aco_best_cost = float('inf')
        self.ga_best_cost = float('inf')
        
        # Thiết lập lại biến thời gian
        if hasattr(self, 'aco_execution_time'):
            delattr(self, 'aco_execution_time')
        if hasattr(self, 'ga_execution_time'):
            delattr(self, 'ga_execution_time')
        if hasattr(self, 'aco_start_time'):
            delattr(self, 'aco_start_time')
        if hasattr(self, 'ga_start_time'):
            delattr(self, 'ga_start_time')
        if hasattr(self, 'aco_end_time'):
            delattr(self, 'aco_end_time')
        if hasattr(self, 'ga_end_time'):
            delattr(self, 'ga_end_time')
            
        # Đặt lại thời gian tính toán thuần túy
        self.aco_pure_computation_time = 0
        self.ga_pure_computation_time = 0
        
        # Thiết lập lại trạng thái hoàn thành
        self.aco_completed = False
        self.ga_completed = False
        self.aco_completion_time = None
        self.ga_completion_time = None
        
        # Thiết lập lại hiển thị kết quả
        self.aco_best_cost_var.set("---")
        self.aco_vehicles_var.set("---")
        self.aco_time_var.set("---")
        
        self.ga_best_cost_var.set("---")
        self.ga_vehicles_var.set("---")
        self.ga_time_var.set("---")
        
        self.cost_diff_var.set("---")
        self.time_diff_var.set("---")
        self.better_algo_var.set("---")
        
        # Thiết lập lại hiển thị chi tiết
        self.detail_aco_best_var.set("---")
        self.detail_aco_time_var.set("---")
        self.detail_aco_conv_var.set("---")
        self.detail_aco_stab_var.set("---")
        self.detail_aco_veh_var.set("---")
        
        self.detail_ga_best_var.set("---")
        self.detail_ga_time_var.set("---")
        self.detail_ga_conv_var.set("---")
        self.detail_ga_stab_var.set("---")
        self.detail_ga_veh_var.set("---")
        
        self.detail_best_diff_var.set("---")
        self.detail_time_diff_var.set("---")
        self.detail_veh_diff_var.set("---")
        self.detail_conv_diff_var.set("---")
        self.detail_evaluation_var.set("---")
        
        # Đặt lại trực quan hóa - kiểm tra phương thức có tồn tại hay không
        if hasattr(self, 'aco_visualization'):
            if hasattr(self.aco_visualization, 'reset'):
                self.aco_visualization.reset()
            # Nếu không có phương thức reset, thì có thể khởi tạo lại đối tượng hoặc để nguyên
        
        if hasattr(self, 'ga_visualization'):
            if hasattr(self.ga_visualization, 'reset'):
                self.ga_visualization.reset()
            # Nếu không có phương thức reset, thì có thể khởi tạo lại đối tượng hoặc để nguyên
            
        # Thiết lập lại biểu đồ hội tụ
        if hasattr(self, 'ax_convergence'):
            self.ax_convergence.clear()
            self.ax_convergence.set_title("So sánh hội tụ của hai thuật toán")
            self.ax_convergence.set_xlabel("Vòng lặp")
            self.ax_convergence.set_ylabel("Chi phí tốt nhất")
            self.fig_convergence.canvas.draw()
        
        # Đặt lại tiến trình
        self.progress_var.set(0)
        self.progress_text_var.set("0%")
        
    def save_problem(self):
        """Lưu bài toán CVRP hiện tại vào file"""
        if not self.cvrp or not self.cvrp.customers:
            messagebox.showwarning("Cảnh báo", "Không có bài toán để lưu")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Lưu bài toán CVRP"
        )
        
        if file_path:
            try:
                self.cvrp.save_to_json(file_path)
                messagebox.showinfo("Thành công", f"Đã lưu bài toán vào file {file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi lưu bài toán: {str(e)}")
    
    def load_problem(self):
        """Tải bài toán CVRP từ file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Tải bài toán CVRP"
        )
        
        if file_path:
            try:
                # Đặt lại các biến giữ kết quả
                self.reset_results()
                
                # Tải bài toán từ file
                self.cvrp.load_from_json(file_path)
                self.n_customers = len(self.cvrp.customers) - 1  # Trừ depot
                self.capacity = self.cvrp.capacity
                
                # Cập nhật giao diện
                self.customer_count.set(self.n_customers)
                self.vehicle_capacity.set(self.capacity)
                
                # Hiển thị bài toán trên cả hai trực quan hóa
                self.aco_visualization.init_visualization(self.cvrp, self.aco_n_ants, self.aco_iterations)
                self.ga_visualization.init_visualization(self.cvrp, self.ga_population_size, self.ga_iterations)
                
                messagebox.showinfo("Thành công", f"Đã tải bài toán từ file {file_path}")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi tải bài toán: {str(e)}")
    
    def update_parameters(self):
        """Cập nhật các tham số từ giao diện người dùng"""
        try:
            # Cập nhật tham số CVRP (chung cho cả hai)
            # Lấy giá trị từ các biến StringVar hoặc trực tiếp từ widget
            # Ví dụ: self.n_customers_var đã được tạo trong create_problem_generation_controls
            # và self.capacity_var cũng vậy.
            self.n_customers = int(self.customer_count.get()) # Sửa từ self.n_customers_var thành self.customer_count
            self.capacity = int(self.vehicle_capacity.get()) # Sửa từ self.capacity_var thành self.vehicle_capacity

            # Cập nhật tham số ACO
            self.aco_n_ants = int(self.aco_ant_count.get()) # Sửa tên biến get
            self.aco_alpha = float(self.aco_alpha_value.get()) # Sửa tên biến get
            self.aco_beta = float(self.aco_beta_value.get())   # Sửa tên biến get
            self.aco_rho = float(self.aco_rho_value.get())     # Sửa tên biến get
            self.aco_q = float(self.aco_q_value.get())         # Sửa tên biến get
            self.aco_iterations = int(self.aco_iteration_count.get()) # Sửa tên biến get
            self.aco_min_max = self.aco_minmax_var.get()
            self.aco_local_search = self.aco_localsearch_var.get()
            self.aco_elitist_ants = int(self.aco_elitist_count.get()) # Sửa tên biến get
            
            # Cập nhật tham số GA
            self.ga_population_size = int(self.ga_pop_size.get()) # Sửa tên biến get
            self.ga_crossover_rate = float(self.ga_crossover_prob.get())   # Sửa tên biến get
            self.ga_mutation_rate = float(self.ga_mutation_prob.get())     # Sửa tên biến get
            self.ga_iterations = int(self.ga_generation_count.get())           # Sửa tên biến get
            self.ga_tournament_size = int(self.ga_tournament_size_value.get()) # Sửa tên biến get
            self.ga_elite_size = int(self.ga_elite_count.get())           # Sửa tên biến get
            self.ga_local_search = self.ga_localsearch_var.get()
            self.ga_crossover_type = self.ga_crossover_type_var.get()
            self.ga_mutation_type = self.ga_mutation_type_var.get()

            # Kiểm tra các giá trị không hợp lệ chung
            if self.n_customers <= 0:
                messagebox.showerror("Lỗi tham số CVRP", "Số lượng khách hàng phải lớn hơn 0.")
                return False
            if self.capacity <= 0:
                messagebox.showerror("Lỗi tham số CVRP", "Sức chứa của xe phải lớn hơn 0.")
                return False

            # Kiểm tra các giá trị không hợp lệ cho ACO
            if self.aco_n_ants <= 0:
                messagebox.showerror("Lỗi tham số ACO", "Số lượng kiến phải lớn hơn 0.")
                return False
            if not (0 <= self.aco_rho <= 1):
                messagebox.showerror("Lỗi tham số ACO", "Tỷ lệ bay hơi (rho) phải nằm trong khoảng [0, 1].")
                return False
            if self.aco_iterations <= 0:
                messagebox.showerror("Lỗi tham số ACO", "Số vòng lặp ACO phải lớn hơn 0.")
                return False
            if self.aco_elitist_ants < 0:
                messagebox.showerror("Lỗi tham số ACO", "Số kiến ưu tú ACO không được âm.")
                return False

            # Kiểm tra các giá trị không hợp lệ cho GA
            if self.ga_population_size <= 0:
                messagebox.showerror("Lỗi tham số GA", "Kích thước quần thể GA phải lớn hơn 0.")
                return False
            if not (0 <= self.ga_mutation_rate <= 1):
                messagebox.showerror("Lỗi tham số GA", "Tỷ lệ đột biến GA phải nằm trong khoảng [0, 1].")
                return False
            if not (0 <= self.ga_crossover_rate <= 1):
                messagebox.showerror("Lỗi tham số GA", "Tỷ lệ lai ghép GA phải nằm trong khoảng [0, 1].")
                return False
            if self.ga_elite_size < 0 or self.ga_elite_size > self.ga_population_size:
                messagebox.showerror("Lỗi tham số GA", "Số cá thể ưu tú GA phải không âm và không lớn hơn kích thước quần thể.")
                return False
            if self.ga_iterations <= 0:
                messagebox.showerror("Lỗi tham số GA", "Số thế hệ GA tối đa phải lớn hơn 0.")
                return False
            if self.ga_tournament_size <= 0:
                messagebox.showerror("Lỗi tham số GA", "Kích thước tournament GA phải lớn hơn 0.")
                return False
            
            return True
            
        except ValueError as e: # Giữ lại biến e để có thể debug nếu cần
            messagebox.showerror("Lỗi đầu vào", f"Vui lòng nhập giá trị số hợp lệ cho các tham số. Lỗi: {str(e)}")
            return False
            
    def _set_process_priority(self):
        """Thiết lập mức độ ưu tiên cho process hiện tại"""
        try:
            process = psutil.Process(os.getpid())
            priority_level = self.cpu_priority_var.get()
            
            if priority_level == "low":
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            elif priority_level == "normal":
                process.nice(psutil.NORMAL_PRIORITY_CLASS)
            elif priority_level == "high":
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            elif priority_level == "realtime":
                process.nice(psutil.REALTIME_PRIORITY_CLASS)
        except:
            pass

    def start_comparison(self):
        """Bắt đầu chạy so sánh hai thuật toán"""
        if not self.cvrp or not self.cvrp.customers:
            messagebox.showwarning("Cảnh báo", "Vui lòng tạo hoặc tải bài toán trước")
            return
            
        if not self.update_parameters():
            return
            
        # Đặt lại dữ liệu phân tích
        self.reset_results()
        
        # Thiết lập mức độ ưu tiên CPU
        self._set_process_priority()
        
        # Cập nhật giao diện cho trạng thái đang chạy
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        
        # Khởi tạo các thuật toán
        self.aco_algorithm = ACO_CVRP(
            self.cvrp,
            num_ants=self.aco_n_ants,
            alpha=self.aco_alpha,
            beta=self.aco_beta,
            rho=self.aco_rho,
            q=self.aco_q,
            max_iterations=self.aco_iterations,
            min_max_aco=self.aco_min_max,
            local_search=self.aco_local_search,
            elitist_ants=self.aco_elitist_ants
        )
        
        self.ga_algorithm = GeneticAlgorithm_CVRP(
            self.cvrp,
            population_size=self.ga_population_size,
            crossover_rate=self.ga_crossover_rate,
            mutation_rate=self.ga_mutation_rate,
            tournament_size=self.ga_tournament_size,
            elitism=self.ga_elite_size,
            max_generations=self.ga_iterations,
            selection_method="tournament",
            crossover_method=self.ga_crossover_type,
            mutation_method=self.ga_mutation_type,
            local_search=self.ga_local_search
        )
        
        # Ghi lại thời gian bắt đầu
        self.aco_start_time = time.time()
        self.ga_start_time = time.time()
        
        # Thiết lập trạng thái
        self.is_running = True
        self.is_paused = False
        
        # Sử dụng multithreading với thread có độ ưu tiên cao để tận dụng tối đa CPU
        # Tkinter không hoạt động tốt với multiprocessing do vấn đề with pickle
        if self.use_multiprocessing_var.get():
            # Tối ưu hóa threading
            self.aco_thread = threading.Thread(target=self.run_aco_algorithm_optimized)
            self.ga_thread = threading.Thread(target=self.run_ga_algorithm_optimized)
            
            # Đặt độ ưu tiên thread cao nhất
            self.aco_thread.daemon = True
            self.ga_thread.daemon = True
            
            self.aco_thread.start()
            self.ga_thread.start()
        else:
            # Fallback to simple threading
            self.aco_thread = threading.Thread(target=self.run_aco_algorithm)
            self.ga_thread = threading.Thread(target=self.run_ga_algorithm)
            
            self.aco_thread.start()
            self.ga_thread.start()
    
    def run_aco_algorithm_optimized(self):
        """Phiên bản tối ưu hóa của ACO để tận dụng tối đa CPU"""
        try:
            # Đặt độ ưu tiên cho thread hiện tại
            import os
            try:
                process = psutil.Process(os.getpid())
                priority_level = self.cpu_priority_var.get()
                if priority_level == "realtime":
                    process.nice(psutil.REALTIME_PRIORITY_CLASS)
                elif priority_level == "high":
                    process.nice(psutil.HIGH_PRIORITY_CLASS)
            except:
                pass
                
            print("\n[INFO] === THUẬT TOÁN ACO BẮT ĐẦU CHẠY ===")
            print(f"[INFO] Số kiến: {self.aco_n_ants}, Alpha: {self.aco_alpha}, Beta: {self.aco_beta}")
            print(f"[INFO] Số vòng lặp tối đa: {self.aco_iterations}")
            
            # Định nghĩa các callbacks
            def step_callback(data):
                # Tính toán thời gian tính toán thuần túy
                if 'computation_time' in data:
                    print(f"[ACO] Vòng lặp {data.get('iteration', 0)}/{self.aco_iterations} - Thời gian tính: {data['computation_time']:.4f}s")
                
                # In thông tin tiến trình
                iteration = data.get('iteration', 0)
                if iteration % 5 == 0 or iteration == 1:  # In mỗi 5 vòng lặp để tránh quá nhiều output
                    print(f"[ACO] Vòng lặp {iteration}/{self.aco_iterations} - Chi phí hiện tại: {data['best_cost']:.2f}")
                
                # Xử lý dữ liệu trong thread chính
                self.root.after(0, lambda: self._handle_aco_step_data(data))
                # Kiểm tra dừng
                return not self.is_running
            
            def finish_callback(result):
                # Xử lý kết quả trong thread chính
                self.root.after(0, lambda: self._handle_aco_finish(result))
                
            # Chạy thuật toán với process priority cao
            self.aco_start_time = time.time()
            self.aco_algorithm.run(callback=finish_callback, step_callback=step_callback)
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi chạy ACO: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Lỗi ACO", f"Lỗi khi chạy thuật toán ACO: {str(e)}"))
    
    def _handle_aco_step_data(self, data):
        """Xử lý dữ liệu bước từ ACO trong thread chính"""
        if not self.is_running:
            return
            
        # Cập nhật dữ liệu
        self.aco_best_solution = data['best_solution']
        self.aco_best_cost = data['best_cost']
        self.aco_convergence_data.append(data['best_cost'])
        
        # Cộng dồn thời gian tính toán thuần túy
        if 'computation_time' in data:
            self.aco_pure_computation_time += data['computation_time']
        
        # Cập nhật giao diện
        self.update_aco_visualization(data)
        progress = data['progress'] * 100
        self.update_progress(progress)
        self.update_convergence_chart()
    
    def _handle_aco_finish(self, result):
        """Xử lý kết quả cuối cùng từ ACO trong thread chính"""
        if not self.is_running:
            return
            
        # Cập nhật kết quả
        self.aco_best_solution, self.aco_best_cost = result
        
        # Tính toán thời gian
        self.aco_end_time = time.time()
        self.aco_execution_time = self.aco_end_time - self.aco_start_time
        
        # Đánh dấu hoàn thành
        self.aco_completed = True
        self.aco_completion_time = time.time()
        
        # In kết quả cuối cùng
        print(f"\n[INFO] === THUẬT TOÁN ACO HOÀN THÀNH ===")
        print(f"[INFO] Chi phí tốt nhất ACO: {self.aco_best_cost:.2f}")
        print(f"[INFO] Thời gian chạy ACO: {self.aco_execution_time:.2f} giây")
        print(f"[INFO] Số tuyến đường: {sum(1 for route in self.aco_best_solution if route)}")
        
        # Cập nhật giao diện
        self.update_aco_final_result()
    
    def run_ga_algorithm_optimized(self):
        """Phiên bản tối ưu hóa của GA để tận dụng tối đa CPU"""
        try:
            # Đặt độ ưu tiên cho thread hiện tại
            import os
            try:
                process = psutil.Process(os.getpid())
                priority_level = self.cpu_priority_var.get()
                if priority_level == "realtime":
                    process.nice(psutil.REALTIME_PRIORITY_CLASS)
                elif priority_level == "high":
                    process.nice(psutil.HIGH_PRIORITY_CLASS)
            except:
                pass
                
            print("\n[INFO] === THUẬT TOÁN GA BẮT ĐẦU CHẠY ===")
            print(f"[INFO] Kích thước quần thể: {self.ga_population_size}, Tỷ lệ lai ghép: {self.ga_crossover_rate}")
            print(f"[INFO] Tỷ lệ đột biến: {self.ga_mutation_rate}, Số thế hệ tối đa: {self.ga_iterations}")
            
            # Định nghĩa các callbacks
            def step_callback(data):
                # Tính toán thời gian tính toán thuần túy
                if 'computation_time' in data:
                    print(f"[GA] Thế hệ {data.get('generation', 0)}/{self.ga_iterations} - Thời gian tính: {data['computation_time']:.4f}s")
                
                # In thông tin tiến trình
                generation = data.get('generation', 0)
                if generation % 5 == 0 or generation == 1:  # In mỗi 5 thế hệ để tránh quá nhiều output
                    print(f"[GA] Thế hệ {generation}/{self.ga_iterations} - Chi phí hiện tại: {data['best_cost']:.2f}")
                
                # Xử lý dữ liệu trong thread chính
                self.root.after(0, lambda: self._handle_ga_step_data(data))
                # Kiểm tra dừng
                return not self.is_running
            
            def finish_callback(result):
                # Xử lý kết quả trong thread chính
                self.root.after(0, lambda: self._handle_ga_finish(result))
                
            # Chạy thuật toán với process priority cao
            self.ga_start_time = time.time()
            self.ga_algorithm.run(callback=finish_callback, step_callback=step_callback)
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi chạy GA: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Lỗi GA", f"Lỗi khi chạy thuật toán GA: {str(e)}"))
    
    def _handle_ga_step_data(self, data):
        """Xử lý dữ liệu bước từ GA trong thread chính"""
        if not self.is_running:
            return
            
        # Cập nhật dữ liệu
        self.ga_best_solution = data['best_solution']
        self.ga_best_cost = data['best_cost']
        self.ga_convergence_data.append(data['best_cost'])
        
        # Cộng dồn thời gian tính toán thuần túy
        if 'computation_time' in data:
            self.ga_pure_computation_time += data['computation_time']
        
        # Cập nhật giao diện
        self.update_ga_visualization(data)
        self.update_convergence_chart()
    
    def _handle_ga_finish(self, result):
        """Xử lý kết quả cuối cùng từ GA trong thread chính"""
        if not self.is_running:
            return
            
        # Cập nhật kết quả
        self.ga_best_solution, self.ga_best_cost = result
        
        # Tính toán thời gian
        self.ga_end_time = time.time()
        self.ga_execution_time = self.ga_end_time - self.ga_start_time
        
        # Đánh dấu hoàn thành
        self.ga_completed = True
        self.ga_completion_time = time.time()
        
        # In kết quả cuối cùng
        print(f"\n[INFO] === THUẬT TOÁN GA HOÀN THÀNH ===")
        print(f"[INFO] Chi phí tốt nhất GA: {self.ga_best_cost:.2f}")
        print(f"[INFO] Thời gian chạy GA: {self.ga_execution_time:.2f} giây")
        print(f"[INFO] Số tuyến đường: {sum(1 for route in self.ga_best_solution if route)}")
        
        # Cập nhật giao diện
        self.update_ga_final_result()
    
    def update_aco_visualization(self, data):
        """Cập nhật trực quan hóa ACO"""
        if not self.is_running:
            return
            
        self.aco_visualization.update(
            self.cvrp,
            data['best_solution'],
            data['best_cost'],
            data.get('pheromone', None)
        )
        
    def update_ga_visualization(self, data):
        """Cập nhật trực quan hóa GA"""
        if not self.is_running:
            return
            
        self.ga_visualization.update(
            self.cvrp,
            data['best_solution'],
            data['best_cost'],
            data.get('population', None)
        )

    def update_convergence_chart(self):
        """Cập nhật biểu đồ hội tụ"""
        if not self.is_running:
            return
            
        # Xóa nội dung hiện tại của biểu đồ
        self.ax_convergence.clear()
        
        # Kiểm tra xem có dữ liệu không trước khi vẽ
        has_aco_data = len(self.aco_convergence_data) > 0
        has_ga_data = len(self.ga_convergence_data) > 0
        
        # Vẽ dữ liệu ACO nếu có
        if has_aco_data:
            aco_x = list(range(len(self.aco_convergence_data)))
            self.ax_convergence.plot(aco_x, self.aco_convergence_data, 'r-', label="ACO")
        
        # Vẽ dữ liệu GA nếu có
        if has_ga_data:
            ga_x = list(range(len(self.ga_convergence_data)))
            self.ax_convergence.plot(ga_x, self.ga_convergence_data, 'b-', label="GA")
        
        # Thiết lập tiêu đề và nhãn
        self.ax_convergence.set_title("So sánh hội tụ của hai thuật toán")
        self.ax_convergence.set_xlabel("Vòng lặp")
        self.ax_convergence.set_ylabel("Chi phí tốt nhất")
        
        # Thiết lập giới hạn trục dựa trên dữ liệu hiện có
        if has_aco_data or has_ga_data:
            # Thiết lập giới hạn trục x
            max_x = max(len(self.aco_convergence_data), len(self.ga_convergence_data))
            self.ax_convergence.set_xlim(0, max(10, max_x))
            
            # Thiết lập giới hạn trục y
            if has_aco_data and has_ga_data:
                min_y = min(min(self.aco_convergence_data), min(self.ga_convergence_data))
                max_y = max(max(self.aco_convergence_data), max(self.ga_convergence_data))
            elif has_aco_data:
                min_y = min(self.aco_convergence_data)
                max_y = max(self.aco_convergence_data)
            else:
                min_y = min(self.ga_convergence_data)
                max_y = max(self.ga_convergence_data)
                
            # Thêm một chút padding để đồ thị dễ nhìn hơn
            y_padding = (max_y - min_y) * 0.1 if max_y > min_y else max_y * 0.1
            self.ax_convergence.set_ylim(max(0, min_y - y_padding), max_y + y_padding)
        else:
            # Giá trị mặc định nếu không có dữ liệu
            self.ax_convergence.set_xlim(0, 10)
            self.ax_convergence.set_ylim(0, 1000)
        
        # Thêm legend
        if has_aco_data or has_ga_data:
            self.ax_convergence.legend()
            
        # Vẽ lưới để dễ đọc hơn
        self.ax_convergence.grid(True, linestyle='--', alpha=0.7)
        
        # Vẽ lại biểu đồ
        self.canvas_convergence.draw()
        
    def update_aco_final_result(self):
        """Cập nhật kết quả cuối cùng của thuật toán ACO"""
        if not self.is_running:
            return
            
        if self.aco_best_solution is None or not self.aco_convergence_data:
            return
            
        try:
            # Count non-empty routes
            non_empty_routes = sum(1 for route in self.aco_best_solution if route)
                
            self.aco_best_cost_var.set(f"{self.aco_best_cost:.2f}")
            self.aco_vehicles_var.set(f"{non_empty_routes}")
            self.aco_time_var.set(f"{self.aco_execution_time:.2f}s (CPU: {self.aco_pure_computation_time:.2f}s)")
            
            self.detail_aco_best_var.set(f"{self.aco_best_cost:.2f}")
            self.detail_aco_time_var.set(f"{self.aco_execution_time:.2f}s (CPU: {self.aco_pure_computation_time:.2f}s)")
            
            self.detail_aco_conv_var.set(f"{self.aco_convergence_data[-1]:.2f}")
            self.detail_aco_stab_var.set("Stable")
            self.detail_aco_veh_var.set(f"{non_empty_routes}")
            
            if len(self.aco_convergence_data) >= 2:
                self.detail_best_diff_var.set(f"{self.aco_best_cost - self.aco_convergence_data[-1]:.2f}")
                self.detail_conv_diff_var.set(f"{self.aco_convergence_data[-1] - self.aco_convergence_data[-2]:.2f}")
            
            # Kiểm tra xem cả hai thuật toán đã hoàn thành chưa   
            if self.aco_completed and self.ga_completed:
                self.update_comparison_results()
                
                # Hiển thị thông tin thuật toán nào kết thúc trước
                completion_message = ""
                if self.aco_completion_time < self.ga_completion_time:
                    time_diff = self.ga_completion_time - self.aco_completion_time
                    completion_message = f"ACO kết thúc trước GA: {time_diff:.2f} giây"
                else:
                    time_diff = self.aco_completion_time - self.ga_completion_time
                    completion_message = f"GA kết thúc trước ACO: {time_diff:.2f} giây"
                    
                # Hiển thị thông báo
                messagebox.showinfo("Kết quả so sánh", f"Cả hai thuật toán đã hoàn thành.\n{completion_message}")
                
        except Exception as e:
            print(f"Lỗi khi cập nhật kết quả ACO: {str(e)}")
        
    def update_ga_final_result(self):
        """Cập nhật kết quả cuối cùng của thuật toán GA"""
        if not self.is_running:
            return
            
        if self.ga_best_solution is None or not self.ga_convergence_data:
            return
            
        try:
            # Count non-empty routes
            non_empty_routes = sum(1 for route in self.ga_best_solution if route)
                
            self.ga_best_cost_var.set(f"{self.ga_best_cost:.2f}")
            self.ga_vehicles_var.set(f"{non_empty_routes}")
            self.ga_time_var.set(f"{self.ga_execution_time:.2f}s (CPU: {self.ga_pure_computation_time:.2f}s)")
            
            self.detail_ga_best_var.set(f"{self.ga_best_cost:.2f}")
            self.detail_ga_time_var.set(f"{self.ga_execution_time:.2f}s (CPU: {self.ga_pure_computation_time:.2f}s)")
            
            self.detail_ga_conv_var.set(f"{self.ga_convergence_data[-1]:.2f}")
            self.detail_ga_stab_var.set("Stable")
            self.detail_ga_veh_var.set(f"{non_empty_routes}")
            
            if len(self.ga_convergence_data) >= 2:
                self.detail_best_diff_var.set(f"{self.ga_best_cost - self.ga_convergence_data[-1]:.2f}")
                self.detail_conv_diff_var.set(f"{self.ga_convergence_data[-1] - self.ga_convergence_data[-2]:.2f}")
                
            # Kiểm tra xem cả hai thuật toán đã hoàn thành chưa
            if self.aco_completed and self.ga_completed:
                self.update_comparison_results()
                
                # Thông báo đã được hiển thị trong update_aco_final_result để tránh hiển thị hai lần
        except Exception as e:
            print(f"Lỗi khi cập nhật kết quả GA: {str(e)}")
        
    def update_comparison_results(self):
        """Cập nhật kết quả so sánh giữa hai thuật toán"""
        # Kiểm tra xem cả hai thuật toán đã hoàn thành chưa
        if not self.aco_completed or not self.ga_completed:
            return
            
        if self.aco_best_solution is None or self.ga_best_solution is None:
            return
            
        if not self.aco_convergence_data or not self.ga_convergence_data:
            return
            
        # Tính toán chênh lệch chi phí
        cost_diff = self.aco_best_cost - self.ga_best_cost
        time_diff = self.aco_execution_time - self.ga_execution_time
        
        # Thêm chênh lệch thời gian tính toán thuần túy
        comp_time_diff = self.aco_pure_computation_time - self.ga_pure_computation_time
        
        # Cập nhật hiển thị chênh lệch chi phí
        self.cost_diff_var.set(f"{abs(cost_diff):.2f}")
        
        # Cập nhật hiển thị chênh lệch thời gian
        self.time_diff_var.set(f"{abs(time_diff):.2f}s (CPU: {abs(comp_time_diff):.2f}s)")
        
        # Xác định thuật toán tốt hơn
        better_algo = "ACO" if self.aco_best_cost < self.ga_best_cost else "GA"
        self.better_algo_var.set(better_algo)
        
        # Cập nhật chi tiết phân tích
        cost_pct = abs(cost_diff) / max(self.aco_best_cost, self.ga_best_cost) * 100
        time_pct = abs(time_diff) / max(self.aco_execution_time, self.ga_execution_time) * 100
        comp_time_pct = abs(comp_time_diff) / max(self.aco_pure_computation_time, self.ga_pure_computation_time) * 100 if max(self.aco_pure_computation_time, self.ga_pure_computation_time) > 0 else 0
        
        # Cập nhật chênh lệch
        self.detail_best_diff_var.set(f"{abs(cost_diff):.2f} ({cost_pct:.2f}%)")
        self.detail_time_diff_var.set(f"{abs(time_diff):.2f}s ({time_pct:.2f}%) | CPU: {abs(comp_time_diff):.2f}s ({comp_time_pct:.2f}%)")
        
        # So sánh số xe
        try:
            aco_vehicles = sum(1 for route in self.aco_best_solution if route)
            ga_vehicles = sum(1 for route in self.ga_best_solution if route)
            vehicle_diff = aco_vehicles - ga_vehicles
            better_vehicles = "GA" if vehicle_diff > 0 else "ACO" if vehicle_diff < 0 else "Bằng nhau"
            self.detail_veh_diff_var.set(f"{abs(vehicle_diff)} xe ({better_vehicles} tốt hơn)")
        except Exception:
            self.detail_veh_diff_var.set("Lỗi khi phân tích")
        
        # Đánh giá tốc độ hội tụ
        try:
            if len(self.aco_convergence_data) > 10 and len(self.ga_convergence_data) > 10:
                aco_improve_rate = (self.aco_convergence_data[0] - self.aco_convergence_data[-1]) / len(self.aco_convergence_data)
                ga_improve_rate = (self.ga_convergence_data[0] - self.ga_convergence_data[-1]) / len(self.ga_convergence_data)
                better_conv = "ACO" if aco_improve_rate > ga_improve_rate else "GA"
                self.detail_conv_diff_var.set(f"{better_conv} nhanh hơn")
            else:
                self.detail_conv_diff_var.set("Không đủ dữ liệu")
        except Exception:
            self.detail_conv_diff_var.set("Lỗi khi phân tích")
        
        # Đánh giá tổng quát
        try:
            evaluation = f"{better_algo} là thuật toán tốt hơn cho bài toán này, "
            if better_algo == "ACO":
                if self.aco_pure_computation_time < self.ga_pure_computation_time:
                    evaluation += f"với chi phí thấp hơn {abs(cost_diff):.2f} ({cost_pct:.2f}%) và thời gian tính toán nhanh hơn {abs(comp_time_diff):.2f}s ({comp_time_pct:.2f}%)."
                else:
                    evaluation += f"với chi phí thấp hơn {abs(cost_diff):.2f} ({cost_pct:.2f}%) mặc dù thời gian tính toán chậm hơn {abs(comp_time_diff):.2f}s ({comp_time_pct:.2f}%)."
            else:  # GA better
                if self.ga_pure_computation_time < self.aco_pure_computation_time:
                    evaluation += f"với chi phí thấp hơn {abs(cost_diff):.2f} ({cost_pct:.2f}%) và thời gian tính toán nhanh hơn {abs(comp_time_diff):.2f}s ({comp_time_pct:.2f}%)."
                else:
                    evaluation += f"với chi phí thấp hơn {abs(cost_diff):.2f} ({cost_pct:.2f}%) mặc dù thời gian tính toán chậm hơn {abs(comp_time_diff):.2f}s ({comp_time_pct:.2f}%)."
            
            self.detail_evaluation_var.set(evaluation)
        except Exception as e:
            self.detail_evaluation_var.set(f"Lỗi khi đánh giá: {str(e)}")
        
    def update_progress(self, progress):
        """Cập nhật tiến trình thực thi"""
        self.progress_var.set(progress)
        self.progress_text_var.set(f"{progress:.0f}%")
        
    def export_results(self):
        """Xuất kết quả so sánh ra file"""
        if not self.aco_best_solution or not self.ga_best_solution:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để xuất. Vui lòng chạy so sánh trước.")
            return
            
        # Tạo thư mục results nếu chưa tồn tại
        results_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Tạo tên subfolder dựa trên thời gian
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder_name = f"comparison_{timestamp}"
        subfolder_path = os.path.join(results_dir, subfolder_name)
        
        # Tạo subfolder
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            
        # Lưu kết quả vào file
        filepath = os.path.join(subfolder_path, "results.txt")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("KẾT QUẢ SO SÁNH THUẬT TOÁN CVRP\n")
            f.write("=" * 40 + "\n\n")
            
            # Thông tin bài toán
            f.write("THÔNG TIN BÀI TOÁN\n")
            f.write("-" * 20 + "\n")
            f.write(f"Số điểm giao hàng: {self.n_customers}\n")
            f.write(f"Sức chứa xe: {self.capacity}\n\n")
            
            # Tham số ACO
            f.write("THAM SỐ ACO\n")
            f.write("-" * 20 + "\n")
            f.write(f"Số lượng kiến: {self.aco_n_ants}\n")
            f.write(f"Alpha: {self.aco_alpha}\n")
            f.write(f"Beta: {self.aco_beta}\n")
            f.write(f"Rho: {self.aco_rho}\n")
            f.write(f"Q: {self.aco_q}\n")
            f.write(f"Số vòng lặp: {self.aco_iterations}\n")
            f.write(f"Min-Max ACO: {self.aco_min_max}\n")
            f.write(f"Tìm kiếm cục bộ: {self.aco_local_search}\n")
            f.write(f"Số kiến ưu tú: {self.aco_elitist_ants}\n\n")
            
            # Tham số GA
            f.write("THAM SỐ GA\n")
            f.write("-" * 20 + "\n")
            f.write(f"Kích thước quần thể: {self.ga_population_size}\n")
            f.write(f"Tỷ lệ lai ghép: {self.ga_crossover_rate}\n")
            f.write(f"Tỷ lệ đột biến: {self.ga_mutation_rate}\n")
            f.write(f"Số thế hệ: {self.ga_iterations}\n")
            f.write(f"Kích thước giải đấu: {self.ga_tournament_size}\n")
            f.write(f"Số cá thể ưu tú: {self.ga_elite_size}\n")
            f.write(f"Loại lai ghép: {self.ga_crossover_type}\n")
            f.write(f"Loại đột biến: {self.ga_mutation_type}\n")
            f.write(f"Tìm kiếm cục bộ: {self.ga_local_search}\n\n")
            
            # Kết quả ACO
            f.write("KẾT QUẢ ACO\n")
            f.write("-" * 20 + "\n")
            non_empty_aco_routes = sum(1 for route in self.aco_best_solution if route)
            f.write(f"Chi phí tốt nhất: {self.aco_best_cost:.2f}\n")
            f.write(f"Số xe sử dụng: {non_empty_aco_routes}\n")
            f.write(f"Thời gian thực thi: {self.aco_execution_time:.2f} seconds\n")
            f.write(f"Thời gian tính toán thuần túy: {self.aco_pure_computation_time:.2f} seconds\n\n")
            
            # Kết quả GA
            f.write("KẾT QUẢ GA\n")
            f.write("-" * 20 + "\n")
            non_empty_ga_routes = sum(1 for route in self.ga_best_solution if route)
            f.write(f"Chi phí tốt nhất: {self.ga_best_cost:.2f}\n")
            f.write(f"Số xe sử dụng: {non_empty_ga_routes}\n")
            f.write(f"Thời gian thực thi: {self.ga_execution_time:.2f} seconds\n")
            f.write(f"Thời gian tính toán thuần túy: {self.ga_pure_computation_time:.2f} seconds\n\n")
            
            # Chi tiết tuyến đường ACO
            f.write("CHI TIẾT TUYẾN ĐƯỜNG ACO\n")
            f.write("-" * 20 + "\n")
            for i, route in enumerate(self.aco_best_solution):
                if route:  # Chỉ ghi tuyến đường không rỗng
                    route_demand = sum(self.cvrp.customers[node].demand for node in route)
                    route_distance = self.cvrp.calculate_route_distance(route)
                    f.write(f"Tuyến {i+1}: {route} - Nhu cầu: {route_demand} - Khoảng cách: {route_distance:.2f}\n")
            f.write("\n")
            
            # Chi tiết tuyến đường GA
            f.write("CHI TIẾT TUYẾN ĐƯỜNG GA\n")
            f.write("-" * 20 + "\n")
            for i, route in enumerate(self.ga_best_solution):
                if route:  # Chỉ ghi tuyến đường không rỗng
                    route_demand = sum(self.cvrp.customers[node].demand for node in route)
                    route_distance = self.cvrp.calculate_route_distance(route)
                    f.write(f"Tuyến {i+1}: {route} - Nhu cầu: {route_demand} - Khoảng cách: {route_distance:.2f}\n")
            f.write("\n")
            
            # So sánh
            f.write("SO SÁNH KẾT QUẢ\n")
            f.write("-" * 20 + "\n")
            cost_diff = self.aco_best_cost - self.ga_best_cost
            time_diff = self.aco_execution_time - self.ga_execution_time
            comp_time_diff = self.aco_pure_computation_time - self.ga_pure_computation_time
            better_algo = "ACO" if self.aco_best_cost < self.ga_best_cost else "GA"
            f.write(f"Chênh lệch chi phí: {abs(cost_diff):.2f} ({better_algo} tốt hơn)\n")
            f.write(f"Chênh lệch thời gian thực thi: {abs(time_diff):.2f} seconds\n")
            f.write(f"Chênh lệch thời gian tính toán thuần túy: {abs(comp_time_diff):.2f} seconds\n")
            f.write(f"Thuật toán tốt hơn: {better_algo}\n")
            
        # Lưu biểu đồ hội tụ
        try:
            plt.figure(figsize=(10, 6))
            if len(self.aco_convergence_data) > 0:
                plt.plot(range(len(self.aco_convergence_data)), self.aco_convergence_data, 'r-', label="ACO")
            if len(self.ga_convergence_data) > 0:
                plt.plot(range(len(self.ga_convergence_data)), self.ga_convergence_data, 'b-', label="GA")
                
            plt.title("So sánh hội tụ của hai thuật toán")
            plt.xlabel("Vòng lặp")
            plt.ylabel("Chi phí tốt nhất")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            chart_filepath = os.path.join(subfolder_path, "comparison_chart.png")
            plt.savefig(chart_filepath)
            plt.close()
        except Exception as e:
            messagebox.showwarning("Cảnh báo", f"Không thể lưu biểu đồ: {str(e)}")
            
        # Lưu bản đồ tuyến đường ACO
        try:
            # Lưu một hình ảnh riêng chỉ với tuyến đường ACO
            plt.figure(figsize=(10, 8))
            plt.title("Tuyến đường tối ưu của ACO")
            
            # Vẽ depot
            plt.scatter(self.cvrp.depot.x, self.cvrp.depot.y, c='red', s=100, marker='*', label='Depot')
            
            # Vẽ khách hàng
            x_customers = [customer.x for customer in self.cvrp.customers[1:]]
            y_customers = [customer.y for customer in self.cvrp.customers[1:]]
            demand = [customer.demand for customer in self.cvrp.customers[1:]]
            scatter = plt.scatter(x_customers, y_customers, c=demand, cmap='viridis', 
                               s=50, alpha=0.8, edgecolors='black')
            plt.colorbar(scatter, label='Nhu cầu')
            
            # Vẽ từng tuyến
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.aco_best_solution)))
            for i, route in enumerate(self.aco_best_solution):
                if not route:
                    continue
                    
                # Tạo danh sách điểm (thêm depot vào đầu và cuối)
                x = [self.cvrp.depot.x]
                y = [self.cvrp.depot.y]
                
                for node in route:
                    x.append(self.cvrp.customers[node].x)
                    y.append(self.cvrp.customers[node].y)
                
                # Quay lại depot
                x.append(self.cvrp.depot.x)
                y.append(self.cvrp.depot.y)
                
                # Vẽ tuyến, bỏ label để không hiển thị legend
                plt.plot(x, y, 'o-', c=colors[i % len(colors)], linewidth=2, markersize=6)
            
            # Bỏ tạo legend
            # plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('X')
            plt.ylabel('Y')
            
            # Lưu hình ảnh
            plt.savefig(os.path.join(subfolder_path, "aco_routes.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            messagebox.showwarning("Cảnh báo", f"Không thể lưu bản đồ ACO: {str(e)}")
            
        # Lưu bản đồ tuyến đường GA
        try:
            # Lưu một hình ảnh riêng chỉ với tuyến đường GA
            plt.figure(figsize=(10, 8))
            plt.title("Tuyến đường tối ưu của GA")
            
            # Vẽ depot
            plt.scatter(self.cvrp.depot.x, self.cvrp.depot.y, c='red', s=100, marker='*', label='Depot')
            
            # Vẽ khách hàng
            x_customers = [customer.x for customer in self.cvrp.customers[1:]]
            y_customers = [customer.y for customer in self.cvrp.customers[1:]]
            demand = [customer.demand for customer in self.cvrp.customers[1:]]
            scatter = plt.scatter(x_customers, y_customers, c=demand, cmap='viridis', 
                               s=50, alpha=0.8, edgecolors='black')
            plt.colorbar(scatter, label='Nhu cầu')
            
            # Vẽ từng tuyến
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.ga_best_solution)))
            for i, route in enumerate(self.ga_best_solution):
                if not route:
                    continue
                    
                # Tạo danh sách điểm (thêm depot vào đầu và cuối)
                x = [self.cvrp.depot.x]
                y = [self.cvrp.depot.y]
                
                for node in route:
                    x.append(self.cvrp.customers[node].x)
                    y.append(self.cvrp.customers[node].y)
                
                # Quay lại depot
                x.append(self.cvrp.depot.x)
                y.append(self.cvrp.depot.y)
                
                # Vẽ tuyến, bỏ label để không hiển thị legend
                plt.plot(x, y, 'o-', c=colors[i % len(colors)], linewidth=2, markersize=6)
            
            # Bỏ tạo legend
            # plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('X')
            plt.ylabel('Y')
            
            # Lưu hình ảnh
            plt.savefig(os.path.join(subfolder_path, "ga_routes.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            messagebox.showwarning("Cảnh báo", f"Không thể lưu biểu đồ GA: {str(e)}")
            
        messagebox.showinfo("Thành công", f"Đã xuất kết quả so sánh vào thư mục {subfolder_path}")
        
    def export_analysis(self):
        """Xuất báo cáo phân tích ra file"""
        if not self.aco_best_solution or not self.ga_best_solution:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để xuất. Vui lòng chạy so sánh trước.")
            return
            
        # Tạo thư mục results nếu chưa tồn tại
        results_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Tạo tên subfolder dựa trên thời gian
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder_name = f"analysis_{timestamp}"
        subfolder_path = os.path.join(results_dir, subfolder_name)
        
        # Tạo subfolder
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            
        # Tạo tên file
        filepath = os.path.join(subfolder_path, "analysis.txt")
        
        # Lưu phân tích vào file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("PHÂN TÍCH CHI TIẾT THUẬT TOÁN CVRP\n")
            f.write("=" * 40 + "\n\n")
            
            # Thông tin bài toán
            f.write("THÔNG TIN BÀI TOÁN\n")
            f.write("-" * 20 + "\n")
            f.write(f"Số điểm giao hàng: {self.n_customers}\n")
            f.write(f"Sức chứa xe: {self.capacity}\n\n")
            
            # Phân tích ACO
            f.write("PHÂN TÍCH ACO\n")
            f.write("-" * 20 + "\n")
            non_empty_aco_routes = sum(1 for route in self.aco_best_solution if route)
            f.write(f"Chi phí tốt nhất: {self.aco_best_cost:.2f}\n")
            f.write(f"Thời gian thực thi: {self.aco_execution_time:.2f} seconds\n")
            f.write(f"Số vòng lặp hội tụ: {len(self.aco_convergence_data)}\n")
            f.write(f"Chi phí cuối cùng: {self.aco_convergence_data[-1]:.2f}\n")
            f.write(f"Chênh lệch với giá trị tốt nhất: {self.aco_best_cost - self.aco_convergence_data[-1]:.2f}\n")
            if len(self.aco_convergence_data) > 1:
                f.write(f"Độ cải thiện ở vòng lặp cuối: {self.aco_convergence_data[-2] - self.aco_convergence_data[-1]:.2f}\n")
            f.write(f"Số xe sử dụng: {non_empty_aco_routes}\n\n")
            
            # Phân tích GA
            f.write("PHÂN TÍCH GA\n")
            f.write("-" * 20 + "\n")
            non_empty_ga_routes = sum(1 for route in self.ga_best_solution if route)
            f.write(f"Chi phí tốt nhất: {self.ga_best_cost:.2f}\n")
            f.write(f"Thời gian thực thi: {self.ga_execution_time:.2f} seconds\n")
            f.write(f"Số thế hệ hội tụ: {len(self.ga_convergence_data)}\n")
            f.write(f"Chi phí cuối cùng: {self.ga_convergence_data[-1]:.2f}\n")
            f.write(f"Chênh lệch với giá trị tốt nhất: {self.ga_best_cost - self.ga_convergence_data[-1]:.2f}\n")
            if len(self.ga_convergence_data) > 1:
                f.write(f"Độ cải thiện ở thế hệ cuối: {self.ga_convergence_data[-2] - self.ga_convergence_data[-1]:.2f}\n")
            f.write(f"Số xe sử dụng: {non_empty_ga_routes}\n\n")
            
            # So sánh chi tiết
            f.write("SO SÁNH CHI TIẾT\n")
            f.write("-" * 20 + "\n")
            cost_diff = self.aco_best_cost - self.ga_best_cost
            time_diff = self.aco_execution_time - self.ga_execution_time
            comp_time_diff = self.aco_pure_computation_time - self.ga_pure_computation_time
            cost_pct = abs(cost_diff) / max(self.aco_best_cost, self.ga_best_cost) * 100
            time_pct = abs(time_diff) / max(self.aco_execution_time, self.ga_execution_time) * 100
            comp_time_pct = abs(comp_time_diff) / max(self.aco_pure_computation_time, self.ga_pure_computation_time) * 100 if max(self.aco_pure_computation_time, self.ga_pure_computation_time) > 0 else 0
            
            better_cost = "ACO" if self.aco_best_cost < self.ga_best_cost else "GA"
            better_time = "ACO" if self.aco_execution_time < self.ga_execution_time else "GA"
            better_comp_time = "ACO" if self.aco_pure_computation_time < self.ga_pure_computation_time else "GA"
            
            f.write(f"Chênh lệch chi phí: {abs(cost_diff):.2f} ({cost_pct:.2f}%) - {better_cost} tốt hơn\n")
            f.write(f"Chênh lệch thời gian: {abs(time_diff):.2f} seconds ({time_pct:.2f}%) - {better_time} nhanh hơn\n")
            f.write(f"Chênh lệch thời gian tính toán thuần túy: {abs(comp_time_diff):.2f}s ({comp_time_pct:.2f}%) - {better_comp_time} nhanh hơn\n")
            
            # Đánh giá tốc độ hội tụ
            if len(self.aco_convergence_data) > 10 and len(self.ga_convergence_data) > 10:
                aco_improve_rate = (self.aco_convergence_data[0] - self.aco_convergence_data[-1]) / len(self.aco_convergence_data)
                ga_improve_rate = (self.ga_convergence_data[0] - self.ga_convergence_data[-1]) / len(self.ga_convergence_data)
                better_conv = "ACO" if aco_improve_rate > ga_improve_rate else "GA"
                f.write(f"Tốc độ hội tụ: {better_conv} hội tụ nhanh hơn\n")
                f.write(f"Tốc độ cải thiện ACO: {aco_improve_rate:.2f} đơn vị/vòng lặp\n")
                f.write(f"Tốc độ cải thiện GA: {ga_improve_rate:.2f} đơn vị/thế hệ\n")
            
            # Kết luận
            f.write("\nKẾT LUẬN\n")
            f.write("-" * 20 + "\n")
            if self.aco_best_cost < self.ga_best_cost:
                f.write(f"ACO cho kết quả tốt hơn với chi phí thấp hơn {abs(cost_diff):.2f} ({cost_pct:.2f}%)\n")
            else:
                f.write(f"GA cho kết quả tốt hơn với chi phí thấp hơn {abs(cost_diff):.2f} ({cost_pct:.2f}%)\n")
                
            if self.aco_execution_time < self.ga_execution_time:
                f.write(f"ACO chạy nhanh hơn GA {abs(time_diff):.2f} seconds ({time_pct:.2f}%)\n")
            else:
                f.write(f"GA chạy nhanh hơn ACO {abs(time_diff):.2f} seconds ({time_pct:.2f}%)\n")
            
            if self.aco_pure_computation_time < self.ga_pure_computation_time:
                f.write(f"ACO có thời gian tính toán nhanh hơn GA {abs(comp_time_diff):.2f}s ({comp_time_pct:.2f}%)\n")
            else:
                f.write(f"GA có thời gian tính toán nhanh hơn ACO {abs(comp_time_diff):.2f}s ({comp_time_pct:.2f}%)\n")
            
            f.write("\nTùy vào yêu cầu về thời gian thực thi và chất lượng giải pháp, ")
            if self.aco_best_cost < self.ga_best_cost and self.aco_execution_time < self.ga_execution_time:
                f.write("ACO là lựa chọn tốt hơn trong mọi trường hợp.\n")
            elif self.ga_best_cost < self.aco_best_cost and self.ga_execution_time < self.aco_execution_time:
                f.write("GA là lựa chọn tốt hơn trong mọi trường hợp.\n")
            elif self.aco_best_cost < self.ga_best_cost:
                f.write("ACO là lựa chọn tốt hơn nếu ưu tiên chất lượng giải pháp.\n")
            else:
                f.write("GA là lựa chọn tốt hơn nếu ưu tiên chất lượng giải pháp.\n")
                
        # Lưu biểu đồ hội tụ so sánh
        try:
            plt.figure(figsize=(10, 6))
            
            if len(self.aco_convergence_data) > 0 and len(self.ga_convergence_data) > 0:
                # Chuẩn hóa trục x để so sánh dễ dàng hơn
                aco_x = [i / len(self.aco_convergence_data) for i in range(len(self.aco_convergence_data))]
                ga_x = [i / len(self.ga_convergence_data) for i in range(len(self.ga_convergence_data))]
                
                plt.plot(aco_x, self.aco_convergence_data, 'r-', label="ACO")
                plt.plot(ga_x, self.ga_convergence_data, 'b-', label="GA")
                
                plt.title("So sánh tốc độ hội tụ (chuẩn hóa)")
                plt.xlabel("Tỷ lệ hoàn thành (%)")
                plt.ylabel("Chi phí tốt nhất")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                chart_filepath = os.path.join(subfolder_path, "normalized_comparison.png")
                plt.savefig(chart_filepath)
                plt.close()
            
        except Exception as e:
            messagebox.showwarning("Cảnh báo", f"Không thể lưu biểu đồ chuẩn hóa: {str(e)}")
            
        messagebox.showinfo("Thành công", f"Đã xuất báo cáo phân tích vào thư mục {subfolder_path}")
        
    def on_closing(self):
        """Xử lý sự kiện đóng cửa sổ"""
        # Dừng tất cả các processes khi đóng ứng dụng
        self.stop_algorithm()
        
        if self.is_running:
            if messagebox.askyesno("Xác nhận", "Các thuật toán đang chạy. Bạn có chắc muốn hủy và quay lại?"):
                self.root.destroy()
                self.selector_root.deiconify()
        else:
            self.root.destroy()
            self.selector_root.deiconify()
        
    def stop_algorithm(self):
        """Dừng các thuật toán và đặt lại trạng thái chạy"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Dừng các thuật toán
        if self.aco_algorithm:
            self.aco_algorithm.stop_flag = True
        
        if self.ga_algorithm:
            self.ga_algorithm.stop_flag = True
        
        # Cập nhật giao diện
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        
        # Đặt lại trạng thái hoàn thành
        self.aco_completed = False
        self.ga_completed = False
        self.aco_completion_time = None
        self.ga_completion_time = None
        
        # Hiển thị thông báo
        messagebox.showinfo("Thông báo", "So sánh thuật toán đã bị dừng.")
        
    def pause_algorithms(self):
        """Tạm dừng thuật toán"""
        self.is_paused = True
        if hasattr(self, 'aco_algorithm') and self.aco_algorithm:
            self.aco_algorithm.pause()
        if hasattr(self, 'ga_algorithm') and self.ga_algorithm:
            self.ga_algorithm.pause()
        
        # Cập nhật giao diện
        self.pause_button.config(text="Tiếp tục", command=self.resume_algorithms)
        
    def resume_algorithms(self):
        """Tiếp tục thuật toán sau khi tạm dừng"""
        self.is_paused = False
        if hasattr(self, 'aco_algorithm') and self.aco_algorithm:
            self.aco_algorithm.resume()
        if hasattr(self, 'ga_algorithm') and self.ga_algorithm:
            self.ga_algorithm.resume()
        
        # Cập nhật giao diện
        self.pause_button.config(text="Tạm dừng", command=self.pause_algorithms) 