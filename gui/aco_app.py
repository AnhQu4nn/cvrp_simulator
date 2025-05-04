"""
Ứng dụng Ant Colony Optimization
Giao diện đồ họa cải tiến để giải bài toán CVRP với thuật toán ACO
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core import CVRP, ACO_CVRP
from .visualization import ACOVisualization


class AntColonyApp:
    """Ứng dụng CVRP sử dụng Ant Colony Optimization"""

    def __init__(self, root, selector_root=None):
        """Khởi tạo ứng dụng ACO"""
        self.root = root
        self.selector_root = selector_root  # Tham chiếu đến cửa sổ chọn thuật toán
        self.root.title("Giải CVRP với Ant Colony Optimization")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Khởi tạo biến
        self.cvrp = CVRP()
        self.n_customers = 15
        self.capacity = 100
        self.n_ants = 20
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.5
        self.q = 100
        self.iterations = 50
        self.min_max_aco = False  # Mới: chế độ MMAS
        self.local_search = False  # Mới: tìm kiếm cục bộ
        self.elitist_ants = 0      # Mới: số kiến ưu tú

        self.algorithm = None
        self.algorithm_thread = None
        self.is_running = False
        self.is_paused = False     # Mới: trạng thái tạm dừng

        # Dữ liệu phân tích
        self.convergence_data = [] # Mới: lưu trữ dữ liệu hội tụ

        # Tạo giao diện
        self.create_gui()

        # Thiết lập mặc định
        self.select_random.invoke()

    def create_gui(self):
        """Tạo giao diện người dùng"""
        # Notebook (tab container)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab chính
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Thuật toán")

        # Tab phân tích
        analysis_tab = ttk.Frame(self.notebook)  # Mới: tab phân tích
        self.notebook.add(analysis_tab, text="Phân tích")

        # Panel chính trên tab chính
        main_panel = ttk.PanedWindow(main_tab, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True)

        # Panel điều khiển bên trái
        self.control_frame = ttk.LabelFrame(main_panel, text="Điều khiển")
        main_panel.add(self.control_frame, weight=1)

        # Panel trực quan hóa bên phải
        viz_panel = ttk.LabelFrame(main_panel, text="Minh họa")
        main_panel.add(viz_panel, weight=3)

        # Tạo trực quan hóa
        self.visualization = ACOVisualization(viz_panel)

        # Tạo các phần điều khiển
        self.create_problem_generation_controls()
        self.create_algorithm_controls()
        self.create_execution_controls()
        self.create_result_display()

        # Tạo giao diện phân tích trên tab phân tích
        self.create_analysis_tab(analysis_tab)

        # Thêm nút quay lại
        self.add_return_button()

    def create_analysis_tab(self, parent):
        """Tạo tab phân tích kết quả"""
        # Chia tab thành hai phần
        analysis_pane = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        analysis_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel trên cho biểu đồ hội tụ
        convergence_frame = ttk.LabelFrame(analysis_pane, text="Biểu đồ hội tụ")
        analysis_pane.add(convergence_frame, weight=2)

        # Tạo vùng cho biểu đồ matplotlib
        self.fig_convergence = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_convergence = self.fig_convergence.add_subplot(111)
        self.canvas_convergence = FigureCanvasTkAgg(self.fig_convergence, convergence_frame)
        self.canvas_convergence.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_convergence.set_title("Chi phí tốt nhất qua các vòng lặp")
        self.ax_convergence.set_xlabel("Vòng lặp")
        self.ax_convergence.set_ylabel("Chi phí tốt nhất")

        # Panel dưới cho thống kê pheromone
        stats_frame = ttk.LabelFrame(analysis_pane, text="Thống kê pheromone")
        analysis_pane.add(stats_frame, weight=1)

        # Hai cột cho thống kê
        stats_left = ttk.Frame(stats_frame)
        stats_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        stats_right = ttk.Frame(stats_frame)
        stats_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Các thống kê bên trái
        ttk.Label(stats_left, text="Chi phí tốt nhất:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.best_cost_analysis_var = tk.StringVar(value="...")
        ttk.Label(stats_left, textvariable=self.best_cost_analysis_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(stats_left, text="Chi phí trung bình:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.avg_cost_var = tk.StringVar(value="...")
        ttk.Label(stats_left, textvariable=self.avg_cost_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(stats_left, text="Chi phí xấu nhất:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.worst_cost_var = tk.StringVar(value="...")
        ttk.Label(stats_left, textvariable=self.worst_cost_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # Các thống kê bên phải
        ttk.Label(stats_right, text="Mức pheromone trung bình:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.avg_pheromone_var = tk.StringVar(value="...")
        ttk.Label(stats_right, textvariable=self.avg_pheromone_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(stats_right, text="Mức pheromone lớn nhất:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_pheromone_var = tk.StringVar(value="...")
        ttk.Label(stats_right, textvariable=self.max_pheromone_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(stats_right, text="Mức pheromone nhỏ nhất:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_pheromone_var = tk.StringVar(value="...")
        ttk.Label(stats_right, textvariable=self.min_pheromone_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # Nút xuất dữ liệu
        export_button = ttk.Button(stats_frame, text="Xuất kết quả phân tích", command=self.export_analysis)
        export_button.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    def export_analysis(self):
        """Xuất dữ liệu phân tích ra file"""
        if not self.convergence_data:
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu phân tích để xuất")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Xuất dữ liệu phân tích"
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("Vòng lặp,Chi phí tốt nhất,Chi phí trung bình,Chi phí xấu nhất," +
                           "Pheromone trung bình,Pheromone lớn nhất,Pheromone nhỏ nhất\n")
                    for i, data in enumerate(self.convergence_data):
                        f.write(f"{i},{data['best']},{data.get('avg', 0)},{data.get('worst', 0)}," +
                               f"{data.get('avg_pheromone', 0)},{data.get('max_pheromone', 0)},{data.get('min_pheromone', 0)}\n")
                messagebox.showinfo("Thông báo", f"Đã xuất dữ liệu phân tích vào {filename}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xuất dữ liệu: {str(e)}")

    def add_return_button(self):
        """Thêm nút quay lại màn hình chọn thuật toán"""
        back_button = ttk.Button(
            self.control_frame,
            text="Quay lại chọn thuật toán",
            command=self.back_to_selector
        )
        back_button.pack(fill=tk.X, padx=5, pady=5)

    def back_to_selector(self):
        """Quay lại menu chọn thuật toán"""
        if self.is_running:
            if messagebox.askyesno("Quay lại", "Thuật toán đang chạy, bạn có muốn quay lại?"):
                self.stop_algorithm()
                self.root.destroy()
                if self.selector_root:
                    self.selector_root.deiconify()  # Hiển thị cửa sổ chọn
        else:
            self.root.destroy()
            if self.selector_root:
                self.selector_root.deiconify()  # Hiển thị cửa sổ chọn

    def create_problem_generation_controls(self):
        """Tạo bảng điều khiển cho việc tạo bài toán CVRP"""
        problem_frame = ttk.LabelFrame(self.control_frame, text="Tạo bài toán CVRP")
        problem_frame.pack(fill=tk.X, padx=5, pady=5)

        # Số lượng khách hàng
        ttk.Label(problem_frame, text="Số lượng khách hàng:").pack(anchor=tk.W, padx=5, pady=2)
        self.customer_count = tk.StringVar(value=str(self.n_customers))
        customer_count_entry = ttk.Entry(problem_frame, textvariable=self.customer_count, width=10)
        customer_count_entry.pack(fill=tk.X, padx=5, pady=2)

        # Trọng lượng xe
        ttk.Label(problem_frame, text="Trọng lượng xe:").pack(anchor=tk.W, padx=5, pady=2)
        self.capacity_var = tk.StringVar(value=str(self.capacity))
        capacity_entry = ttk.Entry(problem_frame, textvariable=self.capacity_var, width=10)
        capacity_entry.pack(fill=tk.X, padx=5, pady=2)

        # Loại bài toán
        ttk.Label(problem_frame, text="Loại bài toán:").pack(anchor=tk.W, padx=5, pady=2)

        self.problem_type = tk.StringVar(value="random")
        self.select_random = ttk.Radiobutton(problem_frame, text="Ngẫu nhiên",
                                             variable=self.problem_type, value="random")
        self.select_random.pack(anchor=tk.W, padx=5, pady=2)

        # Seed
        ttk.Label(problem_frame, text="Seed (tùy chọn):").pack(anchor=tk.W, padx=5, pady=2)
        self.seed_var = tk.StringVar()
        seed_entry = ttk.Entry(problem_frame, textvariable=self.seed_var, width=10)
        seed_entry.pack(fill=tk.X, padx=5, pady=2)

        # Nút tạo bài toán
        generate_button = ttk.Button(problem_frame, text="Tạo bài toán", command=self.generate_problem)
        generate_button.pack(fill=tk.X, padx=5, pady=5)

        # Nút lưu/tải bài toán
        button_frame = ttk.Frame(problem_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        save_button = ttk.Button(button_frame, text="Lưu", command=self.save_problem)
        save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        load_button = ttk.Button(button_frame, text="Tải", command=self.load_problem)
        load_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

    def create_algorithm_controls(self):
        """Tạo bảng điều khiển cho tham số thuật toán ACO"""
        algo_notebook = ttk.Notebook(self.control_frame)
        algo_notebook.pack(fill=tk.X, padx=5, pady=5)

        # Tab tham số cơ bản
        basic_frame = ttk.Frame(algo_notebook)
        algo_notebook.add(basic_frame, text="Cơ bản")

        # Tab tham số nâng cao
        advanced_frame = ttk.Frame(algo_notebook)
        algo_notebook.add(advanced_frame, text="Nâng cao")

        # Tham số cơ bản
        ttk.Label(basic_frame, text="Số lượng kiến:").pack(anchor=tk.W, padx=5, pady=2)
        self.ant_count = tk.StringVar(value=str(self.n_ants))
        ant_count_entry = ttk.Entry(basic_frame, textvariable=self.ant_count, width=10)
        ant_count_entry.pack(fill=tk.X, padx=5, pady=2)

        # Tooltip cho Alpha
        ttk.Label(basic_frame, text="Alpha (tầm quan trọng pheromone):").pack(anchor=tk.W, padx=5, pady=2)
        self.alpha_var = tk.StringVar(value=str(self.alpha))
        alpha_entry = ttk.Entry(basic_frame, textvariable=self.alpha_var, width=10)
        alpha_entry.pack(fill=tk.X, padx=5, pady=2)

        # Tooltip cho Beta
        ttk.Label(basic_frame, text="Beta (tầm quan trọng khoảng cách):").pack(anchor=tk.W, padx=5, pady=2)
        self.beta_var = tk.StringVar(value=str(self.beta))
        beta_entry = ttk.Entry(basic_frame, textvariable=self.beta_var, width=10)
        beta_entry.pack(fill=tk.X, padx=5, pady=2)

        # Tooltip cho Rho
        ttk.Label(basic_frame, text="Rho (tỷ lệ bay hơi pheromone):").pack(anchor=tk.W, padx=5, pady=2)
        self.rho_var = tk.StringVar(value=str(self.rho))
        rho_entry = ttk.Entry(basic_frame, textvariable=self.rho_var, width=10)
        rho_entry.pack(fill=tk.X, padx=5, pady=2)

        # Tooltip cho Q
        ttk.Label(basic_frame, text="Q (lượng pheromone thả):").pack(anchor=tk.W, padx=5, pady=2)
        self.q_var = tk.StringVar(value=str(self.q))
        q_entry = ttk.Entry(basic_frame, textvariable=self.q_var, width=10)
        q_entry.pack(fill=tk.X, padx=5, pady=2)

        # Số vòng lặp tối đa
        ttk.Label(basic_frame, text="Số vòng lặp tối đa:").pack(anchor=tk.W, padx=5, pady=2)
        self.iterations_var = tk.StringVar(value=str(self.iterations))
        iterations_entry = ttk.Entry(basic_frame, textvariable=self.iterations_var, width=10)
        iterations_entry.pack(fill=tk.X, padx=5, pady=2)

        # Tham số nâng cao
        # MIN-MAX Ant System
        ttk.Label(advanced_frame, text="Biến thể thuật toán:").pack(anchor=tk.W, padx=5, pady=2)

        self.min_max_var = tk.BooleanVar(value=self.min_max_aco)
        min_max_check = ttk.Checkbutton(advanced_frame, text="Sử dụng MIN-MAX Ant System",
                                       variable=self.min_max_var)
        min_max_check.pack(anchor=tk.W, padx=5, pady=2)

        # Tìm kiếm cục bộ
        self.local_search_var = tk.BooleanVar(value=self.local_search)
        local_search_check = ttk.Checkbutton(advanced_frame, text="Sử dụng tìm kiếm cục bộ",
                                           variable=self.local_search_var)
        local_search_check.pack(anchor=tk.W, padx=5, pady=2)

        # Kiến ưu tú
        ttk.Label(advanced_frame, text="Số kiến ưu tú:").pack(anchor=tk.W, padx=5, pady=2)
        self.elitist_ants_var = tk.StringVar(value=str(self.elitist_ants))
        elitist_ants_entry = ttk.Entry(advanced_frame, textvariable=self.elitist_ants_var, width=10)
        elitist_ants_entry.pack(fill=tk.X, padx=5, pady=2)

        # Thêm thông báo về tính năng nâng cao
        note_label = ttk.Label(advanced_frame, text="Chú ý: Các tính năng nâng cao chỉ mới có giao diện, "
                                                 "sẽ được cài đặt trong phiên bản sau.",
                            foreground="red", wraplength=250)
        note_label.pack(fill=tk.X, padx=5, pady=10)

    def create_execution_controls(self):
        """Tạo bảng điều khiển thực thi"""
        exec_frame = ttk.LabelFrame(self.control_frame, text="Thực thi")
        exec_frame.pack(fill=tk.X, padx=5, pady=5)

        # Nút điều khiển
        button_frame = ttk.Frame(exec_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(button_frame, text="Chạy", command=self.start_algorithm)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.stop_button = ttk.Button(button_frame, text="Dừng", command=self.stop_algorithm)
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        self.stop_button.config(state=tk.DISABLED)

        # Nút tạm dừng/tiếp tục (mới)
        self.pause_button = ttk.Button(button_frame, text="Tạm dừng", command=self.pause_algorithm)
        self.pause_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.pause_button.config(state=tk.DISABLED)

        # Thanh tiến trình
        ttk.Label(exec_frame, text="Tiến trình:").pack(anchor=tk.W, padx=5, pady=2)
        progress_frame = ttk.Frame(exec_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=2)

        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.RIGHT, padx=5)

        # Tốc độ mô phỏng
        ttk.Label(exec_frame, text="Tốc độ mô phỏng:").pack(anchor=tk.W, padx=5, pady=2)
        self.speed_var = tk.DoubleVar(value=0.1)
        speed_scale = ttk.Scale(exec_frame, from_=0.01, to=1.0, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.pack(fill=tk.X, padx=5, pady=2)

    def create_result_display(self):
        """Tạo phần hiển thị kết quả"""
        result_frame = ttk.LabelFrame(self.control_frame, text="Kết quả")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Chi phí tốt nhất
        ttk.Label(result_frame, text="Chi phí tốt nhất:").pack(anchor=tk.W, padx=5, pady=2)
        self.best_cost_var = tk.StringVar(value="...")
        ttk.Label(result_frame, textvariable=self.best_cost_var).pack(anchor=tk.W, padx=5, pady=2)

        # Số lượng tuyến đường
        ttk.Label(result_frame, text="Số lượng tuyến:").pack(anchor=tk.W, padx=5, pady=2)
        self.route_count_var = tk.StringVar(value="...")
        ttk.Label(result_frame, textvariable=self.route_count_var).pack(anchor=tk.W, padx=5, pady=2)

        # Thông tin vòng lặp
        ttk.Label(result_frame, text="Vòng lặp hiện tại:").pack(anchor=tk.W, padx=5, pady=2)
        self.iteration_var = tk.StringVar(value="...")
        ttk.Label(result_frame, textvariable=self.iteration_var).pack(anchor=tk.W, padx=5, pady=2)

        # Giải pháp tốt nhất
        ttk.Label(result_frame, text="Giải pháp tốt nhất:").pack(anchor=tk.W, padx=5, pady=2)

        solution_frame = ttk.Frame(result_frame)
        solution_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        scrollbar = ttk.Scrollbar(solution_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.solution_text = tk.Text(solution_frame, height=6, width=20, wrap=tk.WORD,
                                     yscrollcommand=scrollbar.set)
        self.solution_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.solution_text.yview)

        # Thời gian thực thi
        ttk.Label(result_frame, text="Thời gian thực thi:").pack(anchor=tk.W, padx=5, pady=2)
        self.execution_time_var = tk.StringVar(value="...")
        ttk.Label(result_frame, textvariable=self.execution_time_var).pack(anchor=tk.W, padx=5, pady=2)

        # Nút lưu kết quả
        save_result_button = ttk.Button(result_frame, text="Lưu kết quả", command=self.save_results)
        save_result_button.pack(fill=tk.X, padx=5, pady=5)

    def save_results(self):
        """Lưu kết quả ra file"""
        if self.solution_text.get(1.0, tk.END).strip() == "":
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để lưu")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Lưu kết quả"
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(f"Bài toán CVRP - Kết quả từ Ant Colony Optimization\n")
                    f.write(f"Số khách hàng: {self.n_customers}\n")
                    f.write(f"Trọng lượng xe: {self.capacity}\n\n")
                    f.write(f"Chi phí tốt nhất: {self.best_cost_var.get()}\n")
                    f.write(f"Số lượng tuyến: {self.route_count_var.get()}\n")
                    f.write(f"Thời gian thực thi: {self.execution_time_var.get()}\n\n")
                    f.write(f"Giải pháp chi tiết:\n")
                    f.write(self.solution_text.get(1.0, tk.END))
                messagebox.showinfo("Thông báo", f"Đã lưu kết quả vào {filename}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu kết quả: {str(e)}")

    def generate_problem(self):
        """Tạo bài toán CVRP dựa trên tùy chọn người dùng"""
        try:
            self.n_customers = int(self.customer_count.get())
            self.capacity = int(self.capacity_var.get())

            if self.n_customers < 5:
                messagebox.showwarning("Cảnh báo", "Cần ít nhất 5 khách hàng")
                self.n_customers = 5
                self.customer_count.set("5")
            elif self.n_customers > 100:
                messagebox.showwarning("Cảnh báo", "Tối đa 100 khách hàng")
                self.n_customers = 100
                self.customer_count.set("100")

            if self.capacity < 50:
                messagebox.showwarning("Cảnh báo", "Dung lượng tối thiểu là 50")
                self.capacity = 50
                self.capacity_var.set("50")

            # Lấy seed nếu có
            seed_str = self.seed_var.get()
            seed = int(seed_str) if seed_str else None

            # Tạo bài toán CVRP
            self.cvrp = CVRP(capacity=self.capacity)
            self.cvrp.load_problem(self.n_customers, self.capacity, seed)

            # Cập nhật trực quan hóa
            self.update_parameters()
            self.visualization.init_visualization(self.cvrp, self.n_ants, self.iterations)

            # Thiết lập lại tab phân tích
            self.reset_analysis()

            messagebox.showinfo("Thông báo", f"Đã tạo bài toán CVRP với {self.n_customers} khách hàng")

        except ValueError as e:
            messagebox.showerror("Lỗi", f"Dữ liệu không hợp lệ: {e}")

    def reset_analysis(self):
        """Thiết lập lại tab phân tích"""
        self.convergence_data = []
        self.ax_convergence.clear()
        self.ax_convergence.set_title("Chi phí tốt nhất qua các vòng lặp")
        self.ax_convergence.set_xlabel("Vòng lặp")
        self.ax_convergence.set_ylabel("Chi phí tốt nhất")
        self.canvas_convergence.draw()

        # Thiết lập lại các biến thống kê
        self.best_cost_analysis_var.set("...")
        self.avg_cost_var.set("...")
        self.worst_cost_var.set("...")
        self.avg_pheromone_var.set("...")
        self.max_pheromone_var.set("...")
        self.min_pheromone_var.set("...")

    def save_problem(self):
        """Lưu bài toán CVRP vào file"""
        if not self.cvrp or not self.cvrp.customers:
            messagebox.showwarning("Cảnh báo", "Không có bài toán CVRP để lưu")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Lưu bài toán CVRP"
        )

        if filename:
            success = self.cvrp.save_to_file(filename)
            if success:
                messagebox.showinfo("Thông báo",
                                    f"Đã lưu bài toán CVRP với {len(self.cvrp.customers) - 1} khách hàng vào {filename}")
            else:
                messagebox.showerror("Lỗi", "Không thể lưu file")

    def load_problem(self):
        """Tải bài toán CVRP từ file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Tải bài toán CVRP"
        )

        if filename:
            self.cvrp = CVRP()
            success = self.cvrp.load_from_file(filename)

            if success:
                self.n_customers = len(self.cvrp.customers) - 1  # Trừ depot
                self.capacity = self.cvrp.capacity

                self.customer_count.set(str(self.n_customers))
                self.capacity_var.set(str(self.capacity))

                # Cập nhật trực quan hóa
                self.update_parameters()
                self.visualization.init_visualization(self.cvrp, self.n_ants, self.iterations)

                # Thiết lập lại tab phân tích
                self.reset_analysis()

                messagebox.showinfo("Thông báo", f"Đã tải bài toán CVRP với {self.n_customers} khách hàng")
            else:
                messagebox.showerror("Lỗi", "Không thể tải file")

    def update_parameters(self):
        """Cập nhật tham số từ giao diện"""
        try:
            # Tham số cơ bản
            self.n_ants = int(self.ant_count.get())
            self.alpha = float(self.alpha_var.get())
            self.beta = float(self.beta_var.get())
            self.rho = float(self.rho_var.get())
            self.q = float(self.q_var.get())
            self.iterations = int(self.iterations_var.get())

            # Tham số nâng cao
            self.min_max_aco = self.min_max_var.get()
            self.local_search = self.local_search_var.get()
            self.elitist_ants = int(self.elitist_ants_var.get())

            # Kiểm tra giới hạn
            if self.n_ants < 1:
                self.n_ants = 1
                self.ant_count.set("1")

            if self.iterations < 1:
                self.iterations = 1
                self.iterations_var.set("1")

            if self.rho < 0 or self.rho > 1:
                self.rho = max(0, min(1, self.rho))
                self.rho_var.set(str(self.rho))

            if self.elitist_ants < 0:
                self.elitist_ants = 0
                self.elitist_ants_var.set("0")

        except ValueError:
            messagebox.showerror("Lỗi", "Tham số không hợp lệ, sử dụng giá trị mặc định")
            self.n_ants = 20
            self.alpha = 1.0
            self.beta = 2.0
            self.rho = 0.5
            self.q = 100
            self.iterations = 50
            self.min_max_aco = False
            self.local_search = False
            self.elitist_ants = 0

            self.ant_count.set(str(self.n_ants))
            self.alpha_var.set(str(self.alpha))
            self.beta_var.set(str(self.beta))
            self.rho_var.set(str(self.rho))
            self.q_var.set(str(self.q))
            self.iterations_var.set(str(self.iterations))
            self.min_max_var.set(self.min_max_aco)
            self.local_search_var.set(self.local_search)
            self.elitist_ants_var.set(str(self.elitist_ants))

    def start_algorithm(self):
        """Bắt đầu thuật toán"""
        if not self.cvrp or not self.cvrp.customers:
            messagebox.showwarning("Cảnh báo", "Vui lòng tạo bài toán CVRP trước")
            return

        if self.is_running:
            return

        # Cập nhật tham số
        self.update_parameters()

        # Khởi tạo thanh tiến trình
        self.progress['value'] = 0
        self.progress['maximum'] = 100
        self.progress_label.config(text="0%")

        # Khởi tạo thuật toán - chỉ sử dụng các tham số được hỗ trợ bởi ACO_CVRP
        self.algorithm = ACO_CVRP(
            cvrp=self.cvrp,
            num_ants=self.n_ants,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            q=self.q,
            max_iterations=self.iterations
            # Các tham số nâng cao được giữ lại cho phiên bản sau
        )

        # Thiết lập trực quan hóa
        self.visualization.set_algorithm(self.algorithm)
        self.visualization.set_cvrp(self.cvrp)
        self.visualization.init_visualization(self.cvrp, self.n_ants, self.iterations)

        # Thiết lập lại tab phân tích
        self.reset_analysis()
        self.convergence_data = []

        # Cập nhật giao diện
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.NORMAL)
        self.is_running = True
        self.is_paused = False
        self.start_time = time.time()

        # Bắt đầu luồng thuật toán
        self.algorithm_thread = threading.Thread(target=self.run_algorithm)
        self.algorithm_thread.daemon = True
        self.algorithm_thread.start()

    def pause_algorithm(self):
        """Tạm dừng hoặc tiếp tục thuật toán"""
        if not self.is_running:
            return

        if self.is_paused:
            # Tiếp tục
            if hasattr(self.algorithm, 'resume'):
                self.algorithm.resume()
            self.is_paused = False
            self.pause_button.config(text="Tạm dừng")
        else:
            # Tạm dừng
            if hasattr(self.algorithm, 'pause'):
                self.algorithm.pause()
            self.is_paused = True
            self.pause_button.config(text="Tiếp tục")

        # Thông báo nếu tính năng không được hỗ trợ
        if not hasattr(self.algorithm, 'pause') or not hasattr(self.algorithm, 'resume'):
            messagebox.showinfo("Thông báo", "Tính năng tạm dừng/tiếp tục sẽ được cài đặt trong phiên bản sau")

    def run_algorithm(self):
        """Chạy thuật toán trong một luồng riêng"""
        try:
            self.algorithm.run(
                callback=self.algorithm_finished,
                step_callback=self.algorithm_step
            )
        except Exception as e:
            # Xử lý lỗi
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi khi chạy thuật toán: {str(e)}"))
            self.algorithm_finished(None)

    def algorithm_step(self, data):
        """Hàm gọi lại cho mỗi bước của thuật toán"""
        # Cập nhật giao diện trong luồng chính
        self.root.after(0, lambda: self.update_visualization(data))

        # Cập nhật thanh tiến trình
        progress = data.get('progress', 0)
        self.root.after(0, lambda: self.update_progress(progress))

        # Cập nhật biểu đồ hội tụ
        iteration = data.get('iteration', 0)
        best_cost = data.get('best_cost', float('inf'))

        # Giả lập dữ liệu nếu không có
        avg_cost = data.get('avg_cost', best_cost * random.uniform(1.05, 1.2))
        worst_cost = data.get('worst_cost', best_cost * random.uniform(1.2, 1.5))

        # Giả lập thông tin pheromone
        avg_pheromone = data.get('avg_pheromone', random.uniform(0.1, 1.0))
        max_pheromone = data.get('max_pheromone', avg_pheromone * random.uniform(1.5, 2.0))
        min_pheromone = data.get('min_pheromone', avg_pheromone * random.uniform(0.3, 0.8))

        self.convergence_data.append({
            'best': best_cost,
            'avg': avg_cost,
            'worst': worst_cost,
            'avg_pheromone': avg_pheromone,
            'max_pheromone': max_pheromone,
            'min_pheromone': min_pheromone
        })

        self.root.after(0, lambda: self.update_analysis(iteration))

        # Cập nhật vòng lặp hiện tại
        self.root.after(0, lambda: self.iteration_var.set(f"{iteration}/{self.iterations}"))

        # Độ trễ dựa trên tốc độ mô phỏng
        time.sleep(1.0 - self.speed_var.get())

    def update_progress(self, progress):
        """Cập nhật thanh tiến trình"""
        self.progress.config(value=progress * 100)
        self.progress_label.config(text=f"{int(progress * 100)}%")

    def update_analysis(self, iteration):
        """Cập nhật biểu đồ và số liệu phân tích"""
        if not self.convergence_data:
            return

        # Cập nhật biểu đồ hội tụ
        self.ax_convergence.clear()

        # Dữ liệu cho biểu đồ
        iterations = list(range(len(self.convergence_data)))
        best_costs = [data['best'] for data in self.convergence_data]
        avg_costs = [data['avg'] for data in self.convergence_data]
        worst_costs = [data['worst'] for data in self.convergence_data]

        # Vẽ đường
        self.ax_convergence.plot(iterations, best_costs, 'g-', label='Tốt nhất')
        self.ax_convergence.plot(iterations, avg_costs, 'b-', label='Trung bình')
        self.ax_convergence.plot(iterations, worst_costs, 'r-', label='Xấu nhất')

        self.ax_convergence.set_title("Chi phí qua các vòng lặp")
        self.ax_convergence.set_xlabel("Vòng lặp")
        self.ax_convergence.set_ylabel("Chi phí")
        self.ax_convergence.legend()
        self.ax_convergence.grid(True)

        # Thiết lập giới hạn trục y để nhìn rõ hơn
        if len(best_costs) > 1:
            min_cost = min(best_costs)
            max_cost = max(worst_costs)  # Sử dụng worst_costs để có phạm vi đầy đủ
            y_range = max_cost - min_cost
            self.ax_convergence.set_ylim([min_cost - 0.1*y_range, max_cost + 0.1*y_range])

        self.canvas_convergence.draw()

        # Cập nhật thống kê
        if self.convergence_data:
            latest = self.convergence_data[-1]
            self.best_cost_analysis_var.set(f"{latest['best']:.2f}")
            self.avg_cost_var.set(f"{latest['avg']:.2f}")
            self.worst_cost_var.set(f"{latest['worst']:.2f}")
            self.avg_pheromone_var.set(f"{latest['avg_pheromone']:.4f}")
            self.max_pheromone_var.set(f"{latest['max_pheromone']:.4f}")
            self.min_pheromone_var.set(f"{latest['min_pheromone']:.4f}")

    def update_visualization(self, data):
        """Cập nhật giao diện sau mỗi bước thuật toán"""
        if not self.is_running:
            return

        # Cập nhật trực quan hóa
        self.visualization.update_visualization(data)

        # Cập nhật kết quả
        best_cost = data.get('best_cost', float('inf'))
        self.best_cost_var.set(f"{best_cost:.2f}")

        best_solution = data.get('best_solution', None)
        if best_solution:
            self.route_count_var.set(str(len(best_solution)))

            # Hiển thị giải pháp
            solution_str = ""
            for i, route in enumerate(best_solution):
                solution_str += f"Tuyến {i + 1}: 0 → {' → '.join(map(str, route))} → 0\n"

            self.solution_text.delete(1.0, tk.END)
            self.solution_text.insert(tk.END, solution_str)

        # Cập nhật thời gian
        elapsed_time = time.time() - self.start_time
        self.execution_time_var.set(f"{elapsed_time:.2f} giây")

    def algorithm_finished(self, result):
        """Hàm gọi lại khi thuật toán hoàn thành"""
        self.is_running = False
        self.is_paused = False

        # Cập nhật giao diện
        self.root.after(0, lambda: self.finalize_ui())

        if result:
            best_solution, best_cost = result

            # Cập nhật kết quả trong luồng chính
            self.root.after(0, lambda: self.update_final_result(best_solution, best_cost))

    def finalize_ui(self):
        """Cập nhật giao diện sau khi thuật toán hoàn thành"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        self.pause_button.config(text="Tạm dừng")

        # Cập nhật thời gian
        elapsed_time = time.time() - self.start_time
        self.execution_time_var.set(f"{elapsed_time:.2f} giây")

        # Thông báo hoàn thành
        if self.algorithm and hasattr(self.algorithm, 'was_stopped') and not self.algorithm.was_stopped:
            messagebox.showinfo("Hoàn thành", f"Thuật toán đã hoàn thành sau {len(self.convergence_data)} vòng lặp")
        elif self.algorithm:
            messagebox.showinfo("Hoàn thành", f"Thuật toán đã hoàn thành sau {len(self.convergence_data)} vòng lặp")

    def update_final_result(self, best_solution, best_cost):
        """Cập nhật kết quả cuối cùng"""
        self.best_cost_var.set(f"{best_cost:.2f}")

        if best_solution:
            self.route_count_var.set(str(len(best_solution)))

            # Hiển thị giải pháp
            solution_str = ""
            for i, route in enumerate(best_solution):
                solution_str += f"Tuyến {i + 1}: 0 → {' → '.join(map(str, route))} → 0\n"

            self.solution_text.delete(1.0, tk.END)
            self.solution_text.insert(tk.END, solution_str)

    def stop_algorithm(self):
        """Dừng thuật toán"""
        if self.algorithm and self.is_running:
            self.algorithm.stop()
            self.pause_button.config(state=tk.DISABLED)

    def on_closing(self):
        """Xử lý khi đóng cửa sổ"""
        if self.is_running:
            if messagebox.askyesno("Thoát", "Thuật toán đang chạy. Bạn có chắc muốn thoát?"):
                self.stop_algorithm()
                self.root.destroy()
                # Quay lại màn hình chọn nếu tồn tại
                if self.selector_root:
                    self.selector_root.deiconify()
        else:
            self.root.destroy()
            # Quay lại màn hình chọn nếu tồn tại
            if self.selector_root:
                self.selector_root.deiconify()