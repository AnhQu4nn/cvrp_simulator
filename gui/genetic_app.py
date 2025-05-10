"""
Ứng dụng Thuật toán Di truyền
Giao diện đồ họa để giải bài toán CVRP với Thuật toán Di truyền
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core import CVRP, GeneticAlgorithm_CVRP
from .visualization import GeneticVisualization
from .tooltip import ToolTip


class GeneticApp:
    """Ứng dụng CVRP sử dụng Thuật toán Di truyền"""

    def __init__(self, root, selector_root=None):
        """Khởi tạo ứng dụng Thuật toán Di truyền"""
        self.root = root
        self.selector_root = selector_root  # Tham chiếu đến cửa sổ chọn thuật toán
        self.root.title("Giải CVRP với Thuật toán Di truyền")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Khởi tạo biến
        self.cvrp = CVRP()
        self.n_customers = 15
        self.capacity = 100
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism = 5
        self.max_generations = 100
        # Các biến cho tính năng mở rộng
        self.selection_method = "tournament"
        self.crossover_method = "ordered"
        self.mutation_method = "swap"
        self.tournament_size = 3
        self.early_stopping = 20
        self.early_stopping_enabled = True

        self.algorithm = None
        self.algorithm_thread = None
        self.is_running = False
        self.convergence_data = []
        self.is_paused = False

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
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="Phân tích")

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
                                                      tags="self.control_frame")

        # Cấu hình Canvas để cuộn khi kích thước thay đổi
        def configure_scroll_region(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        self.control_frame.bind("<Configure>", configure_scroll_region)

        # Làm cho canvas có thể cuộn bằng chuột
        def _on_mousewheel(event):
            # Chỉ cuộn khi con trỏ nằm trên canvas hoặc các widget con của nó
            if str(event.widget).startswith(str(control_canvas)) or str(event.widget).startswith(str(self.control_frame)):
                control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Gán xử lý sự kiện cho tất cả các widget con
        def _bind_mousewheel(event):
            control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
        def _unbind_mousewheel(event):
            control_canvas.unbind_all("<MouseWheel>")
            
        # Khi chuột vào hoặc rời vùng điều khiển
        control_canvas.bind("<Enter>", _bind_mousewheel)
        control_canvas.bind("<Leave>", _unbind_mousewheel)
        self.control_frame.bind("<Enter>", _bind_mousewheel)
        self.control_frame.bind("<Leave>", _unbind_mousewheel)

        # Panel trực quan hóa bên phải
        viz_panel = ttk.LabelFrame(main_panel, text="Minh họa")
        main_panel.add(viz_panel, weight=3)

        # Tạo trực quan hóa
        self.visualization = GeneticVisualization(viz_panel)

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
        self.ax_convergence.set_title("Giá trị hàm mục tiêu qua các thế hệ")
        self.ax_convergence.set_xlabel("Thế hệ")
        self.ax_convergence.set_ylabel("Chi phí tốt nhất")

        # Panel dưới cho thống kê quần thể
        stats_frame = ttk.LabelFrame(analysis_pane, text="Thống kê quần thể")
        analysis_pane.add(stats_frame, weight=1)

        # Hai cột cho thống kê
        stats_left = ttk.Frame(stats_frame)
        stats_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        stats_right = ttk.Frame(stats_frame)
        stats_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Các thống kê bên trái
        ttk.Label(stats_left, text="Chi phí tốt nhất:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.best_fitness_var = tk.StringVar(value="...")
        ttk.Label(stats_left, textvariable=self.best_fitness_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(stats_left, text="Chi phí trung bình:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.avg_fitness_var = tk.StringVar(value="...")
        ttk.Label(stats_left, textvariable=self.avg_fitness_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(stats_left, text="Chi phí xấu nhất:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.worst_fitness_var = tk.StringVar(value="...")
        ttk.Label(stats_left, textvariable=self.worst_fitness_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # Các thống kê bên phải
        ttk.Label(stats_right, text="Độ đa dạng quần thể:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.diversity_var = tk.StringVar(value="...")
        ttk.Label(stats_right, textvariable=self.diversity_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(stats_right, text="Số thế hệ không cải thiện:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.stagnation_var = tk.StringVar(value="...")
        ttk.Label(stats_right, textvariable=self.stagnation_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(stats_right, text="Tỷ lệ thành công lai ghép:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.crossover_success_var = tk.StringVar(value="...")
        ttk.Label(stats_right, textvariable=self.crossover_success_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

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
                    f.write("Thế hệ,Chi phí tốt nhất,Chi phí trung bình,Chi phí xấu nhất,Độ đa dạng\n")
                    for i, data in enumerate(self.convergence_data):
                        f.write(f"{i},{data['best']},{data['avg']},{data['worst']},{data['diversity']}\n")
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
        """Tạo bảng điều khiển cho tham số thuật toán Di truyền"""
        algo_notebook = ttk.Notebook(self.control_frame)
        algo_notebook.pack(fill=tk.X, padx=5, pady=5)

        # Tab tham số cơ bản
        basic_frame = ttk.Frame(algo_notebook)
        algo_notebook.add(basic_frame, text="Cơ bản")

        # Tab tham số nâng cao
        advanced_frame = ttk.Frame(algo_notebook)
        algo_notebook.add(advanced_frame, text="Nâng cao")

        # Tham số cơ bản
        ttk.Label(basic_frame, text="Kích thước quần thể:").pack(anchor=tk.W, padx=5, pady=2)
        self.population_size_var = tk.StringVar(value=str(self.population_size))
        population_size_entry = ttk.Entry(basic_frame, textvariable=self.population_size_var, width=10)
        population_size_entry.pack(fill=tk.X, padx=5, pady=2)
        ToolTip(population_size_entry, "Số lượng cá thể trong quần thể (số lượng giải pháp ứng viên trong mỗi thế hệ)")

        ttk.Label(basic_frame, text="Tỷ lệ đột biến:").pack(anchor=tk.W, padx=5, pady=2)
        self.mutation_rate_var = tk.StringVar(value=str(self.mutation_rate))
        mutation_rate_entry = ttk.Entry(basic_frame, textvariable=self.mutation_rate_var, width=10)
        mutation_rate_entry.pack(fill=tk.X, padx=5, pady=2)
        ToolTip(mutation_rate_entry, "Xác suất xảy ra đột biến (0-1). Giá trị cao tạo nhiều biến đổi, giá trị thấp giữ ổn định quần thể")

        ttk.Label(basic_frame, text="Tỷ lệ lai ghép:").pack(anchor=tk.W, padx=5, pady=2)
        self.crossover_rate_var = tk.StringVar(value=str(self.crossover_rate))
        crossover_rate_entry = ttk.Entry(basic_frame, textvariable=self.crossover_rate_var, width=10)
        crossover_rate_entry.pack(fill=tk.X, padx=5, pady=2)
        ToolTip(crossover_rate_entry, "Xác suất xảy ra lai ghép (0-1). Giá trị cao tạo nhiều con lai, giá trị thấp giữ nguyên cha mẹ")

        ttk.Label(basic_frame, text="Số cá thể ưu tú:").pack(anchor=tk.W, padx=5, pady=2)
        self.elitism_var = tk.StringVar(value=str(self.elitism))
        elitism_entry = ttk.Entry(basic_frame, textvariable=self.elitism_var, width=10)
        elitism_entry.pack(fill=tk.X, padx=5, pady=2)
        ToolTip(elitism_entry, "Số cá thể tốt nhất được giữ lại nguyên vẹn trong thế hệ tiếp theo")

        ttk.Label(basic_frame, text="Số thế hệ tối đa:").pack(anchor=tk.W, padx=5, pady=2)
        self.generations_var = tk.StringVar(value=str(self.max_generations))
        generations_entry = ttk.Entry(basic_frame, textvariable=self.generations_var, width=10)
        generations_entry.pack(fill=tk.X, padx=5, pady=2)
        ToolTip(generations_entry, "Số thế hệ tối đa thuật toán sẽ tiến hóa")

        # Tham số nâng cao
        # Phương pháp chọn lọc
        ttk.Label(advanced_frame, text="Phương pháp chọn lọc:").pack(anchor=tk.W, padx=5, pady=2)
        self.selection_var = tk.StringVar(value=self.selection_method)
        selection_combo = ttk.Combobox(advanced_frame, textvariable=self.selection_var,
                                    values=["tournament", "roulette", "rank"])
        selection_combo.pack(fill=tk.X, padx=5, pady=2)
        selection_combo.bind("<<ComboboxSelected>>", self.on_selection_change)
        ToolTip(selection_combo, "Phương pháp dùng để chọn cá thể làm cha mẹ:\n- tournament: Chọn qua thi đấu loại\n- roulette: Chọn theo xác suất\n- rank: Chọn dựa trên thứ hạng")

        # Kích thước tournament (chỉ hiển thị khi chọn tournament)
        self.tournament_frame = ttk.Frame(advanced_frame)
        self.tournament_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(self.tournament_frame, text="Kích thước tournament:").pack(anchor=tk.W)
        self.tournament_size_var = tk.StringVar(value=str(self.tournament_size))
        tournament_size_entry = ttk.Entry(self.tournament_frame, textvariable=self.tournament_size_var, width=10)
        tournament_size_entry.pack(fill=tk.X)
        ToolTip(tournament_size_entry, "Số cá thể tham gia mỗi vòng thi đấu chọn lọc (chỉ áp dụng với tournament)")

        # Phương pháp lai ghép
        ttk.Label(advanced_frame, text="Phương pháp lai ghép:").pack(anchor=tk.W, padx=5, pady=2)
        self.crossover_method_var = tk.StringVar(value=self.crossover_method)
        crossover_combo = ttk.Combobox(advanced_frame, textvariable=self.crossover_method_var,
                                     values=["ordered", "partially_mapped", "cycle"])
        crossover_combo.pack(fill=tk.X, padx=5, pady=2)
        ToolTip(crossover_combo, "Phương pháp lai ghép để tạo con:\n- ordered: Lai ghép có thứ tự (OX)\n- partially_mapped: Lai ghép ánh xạ một phần (PMX)\n- cycle: Lai ghép chu kỳ (CX)")

        # Phương pháp đột biến
        ttk.Label(advanced_frame, text="Phương pháp đột biến:").pack(anchor=tk.W, padx=5, pady=2)
        self.mutation_method_var = tk.StringVar(value=self.mutation_method)
        mutation_combo = ttk.Combobox(advanced_frame, textvariable=self.mutation_method_var,
                                    values=["swap", "insert", "scramble", "inversion"])
        mutation_combo.pack(fill=tk.X, padx=5, pady=2)
        ToolTip(mutation_combo, "Phương pháp đột biến:\n- swap: Hoán đổi\n- insert: Chèn\n- inversion: Đảo ngược\n- scramble: Xáo trộn")

        # Dừng sớm
        early_stop_frame = ttk.Frame(advanced_frame)
        early_stop_frame.pack(fill=tk.X, padx=5, pady=5)

        self.early_stop_var = tk.BooleanVar(value=self.early_stopping_enabled)
        early_stop_check = ttk.Checkbutton(early_stop_frame, text="Dừng sớm",
                                          variable=self.early_stop_var)
        early_stop_check.pack(side=tk.LEFT, padx=5)
        ToolTip(early_stop_check, "Dừng thuật toán sớm nếu không có cải thiện sau số thế hệ chỉ định")

        ttk.Label(early_stop_frame, text="Số thế hệ:").pack(side=tk.LEFT, padx=5)
        self.early_stop_gen_var = tk.StringVar(value=str(self.early_stopping))
        early_stop_entry = ttk.Entry(early_stop_frame, textvariable=self.early_stop_gen_var,
                                   width=5)
        early_stop_entry.pack(side=tk.LEFT, padx=5)
        ToolTip(early_stop_entry, "Số thế hệ không cải thiện để dừng thuật toán")

        # Thêm thông báo về tính năng nâng cao
        note_label = ttk.Label(advanced_frame, 
                            text="Các tính năng nâng cao đã được kích hoạt và sẽ ảnh hưởng đến hiệu suất thuật toán",
                            wraplength=250, justify=tk.LEFT)
        note_label.pack(fill=tk.X, padx=5, pady=10)

        # Mặc định ẩn khung tournament nếu không chọn tournament
        if self.selection_method != "tournament":
            self.tournament_frame.pack_forget()

    def on_selection_change(self, event):
        """Xử lý khi thay đổi phương pháp chọn lọc"""
        if self.selection_var.get() == "tournament":
            self.tournament_frame.pack(fill=tk.X, padx=5, pady=2)
        else:
            self.tournament_frame.pack_forget()

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

        # Thông tin thế hệ
        ttk.Label(result_frame, text="Thế hệ hiện tại:").pack(anchor=tk.W, padx=5, pady=2)
        self.generation_var = tk.StringVar(value="...")
        ttk.Label(result_frame, textvariable=self.generation_var).pack(anchor=tk.W, padx=5, pady=2)

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
                    f.write(f"Bài toán CVRP - Kết quả từ Thuật toán Di truyền\n")
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
            self.visualization.init_visualization(self.cvrp, self.population_size, self.max_generations)

            # Thiết lập lại tab phân tích
            self.reset_analysis()

            messagebox.showinfo("Thông báo", f"Đã tạo bài toán CVRP với {self.n_customers} khách hàng")

        except ValueError as e:
            messagebox.showerror("Lỗi", f"Dữ liệu không hợp lệ: {e}")

    def reset_analysis(self):
        """Thiết lập lại tab phân tích"""
        self.convergence_data = []
        self.ax_convergence.clear()
        self.ax_convergence.set_title("Giá trị hàm mục tiêu qua các thế hệ")
        self.ax_convergence.set_xlabel("Thế hệ")
        self.ax_convergence.set_ylabel("Chi phí tốt nhất")
        self.canvas_convergence.draw()

        # Thiết lập lại các biến thống kê
        self.best_fitness_var.set("...")
        self.avg_fitness_var.set("...")
        self.worst_fitness_var.set("...")
        self.diversity_var.set("...")
        self.stagnation_var.set("...")
        self.crossover_success_var.set("...")

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
                self.visualization.init_visualization(self.cvrp, self.population_size, self.max_generations)

                # Thiết lập lại tab phân tích
                self.reset_analysis()

                messagebox.showinfo("Thông báo", f"Đã tải bài toán CVRP với {self.n_customers} khách hàng")
            else:
                messagebox.showerror("Lỗi", "Không thể tải file")

    def update_parameters(self):
        """Cập nhật tham số từ giao diện"""
        try:
            # Tham số cơ bản
            self.population_size = int(self.population_size_var.get())
            self.mutation_rate = float(self.mutation_rate_var.get())
            self.crossover_rate = float(self.crossover_rate_var.get())
            self.elitism = int(self.elitism_var.get())
            self.max_generations = int(self.generations_var.get())

            # Tham số nâng cao
            prev_selection = self.selection_method
            prev_crossover = self.crossover_method
            prev_mutation = self.mutation_method
            prev_early_stop = self.early_stopping_enabled
            
            self.selection_method = self.selection_var.get()
            self.crossover_method = self.crossover_method_var.get()
            self.mutation_method = self.mutation_method_var.get()

            if self.selection_method == "tournament":
                self.tournament_size = int(self.tournament_size_var.get())

            # Dừng sớm
            self.early_stopping_enabled = self.early_stop_var.get()
            self.early_stopping = int(self.early_stop_gen_var.get())
            
            # Thông báo khi tính năng nâng cao được kích hoạt
            changes = []
            if self.selection_method != prev_selection:
                changes.append(f"Phương pháp chọn lọc: {self.selection_method}")
            if self.crossover_method != prev_crossover:
                changes.append(f"Phương pháp lai ghép: {self.crossover_method}")
            if self.mutation_method != prev_mutation:
                changes.append(f"Phương pháp đột biến: {self.mutation_method}")
            if self.early_stopping_enabled != prev_early_stop and self.early_stopping_enabled:
                changes.append(f"Dừng sớm sau {self.early_stopping} thế hệ không cải thiện")
                
            if changes:
                messagebox.showinfo("Tính năng nâng cao", f"Đã kích hoạt: {', '.join(changes)}")

            # Kiểm tra giới hạn
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

            if self.tournament_size < 2:
                self.tournament_size = 2
                self.tournament_size_var.set("2")
            elif self.tournament_size > self.population_size // 2:
                self.tournament_size = self.population_size // 2
                self.tournament_size_var.set(str(self.tournament_size))

            if self.early_stopping < 5:
                self.early_stopping = 5
                self.early_stop_gen_var.set("5")

        except ValueError:
            messagebox.showerror("Lỗi", "Tham số không hợp lệ, sử dụng giá trị mặc định")
            self.population_size = 50
            self.mutation_rate = 0.1
            self.crossover_rate = 0.8
            self.elitism = 5
            self.max_generations = 100
            self.selection_method = "tournament"
            self.crossover_method = "ordered"
            self.mutation_method = "swap"
            self.tournament_size = 3
            self.early_stopping = 20
            self.early_stopping_enabled = True

            self.population_size_var.set(str(self.population_size))
            self.mutation_rate_var.set(str(self.mutation_rate))
            self.crossover_rate_var.set(str(self.crossover_rate))
            self.elitism_var.set(str(self.elitism))
            self.generations_var.set(str(self.max_generations))
            self.selection_var.set(self.selection_method)
            self.crossover_method_var.set(self.crossover_method)
            self.mutation_method_var.set(self.mutation_method)
            self.tournament_size_var.set(str(self.tournament_size))
            self.early_stop_var.set(self.early_stopping_enabled)
            self.early_stop_gen_var.set(str(self.early_stopping))

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

        # Khởi tạo thuật toán - chỉ sử dụng các tham số được hỗ trợ bởi GeneticAlgorithm_CVRP
        self.algorithm = GeneticAlgorithm_CVRP(
            cvrp=self.cvrp,
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elitism=self.elitism,
            max_generations=self.max_generations,
            # Thêm các tham số nâng cao
            selection_method=self.selection_method,
            crossover_method=self.crossover_method,
            mutation_method=self.mutation_method,
            tournament_size=self.tournament_size,
            early_stopping=self.early_stopping if self.early_stopping_enabled else None
        )

        # Thiết lập trực quan hóa
        self.visualization.set_algorithm(self.algorithm)
        self.visualization.set_cvrp(self.cvrp)
        self.visualization.init_visualization(self.cvrp, self.population_size, self.max_generations)

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
        generation = data.get('generation', 0)
        best_cost = data.get('best_cost', float('inf'))

        # Giả lập dữ liệu trung bình và xấu nhất nếu không có
        avg_cost = data.get('avg_cost', best_cost * random.uniform(1.05, 1.2))
        worst_cost = data.get('worst_cost', best_cost * random.uniform(1.2, 1.5))
        diversity = data.get('diversity', random.uniform(10, 50))

        self.convergence_data.append({
            'best': best_cost,
            'avg': avg_cost,
            'worst': worst_cost,
            'diversity': diversity
        })

        self.root.after(0, lambda: self.update_analysis(generation))

        # Cập nhật thế hệ hiện tại
        self.root.after(0, lambda: self.generation_var.set(f"{generation}/{self.max_generations}"))

        # Độ trễ dựa trên tốc độ mô phỏng
        time.sleep(1.0 - self.speed_var.get())

    def update_progress(self, progress):
        """Cập nhật thanh tiến trình"""
        self.progress.config(value=progress * 100)
        self.progress_label.config(text=f"{int(progress * 100)}%")

    def update_analysis(self, generation):
        """Cập nhật biểu đồ và số liệu phân tích"""
        if not self.convergence_data:
            return

        # Cập nhật biểu đồ hội tụ
        self.ax_convergence.clear()

        # Dữ liệu cho biểu đồ
        generations = list(range(len(self.convergence_data)))
        best_costs = [data['best'] for data in self.convergence_data]
        avg_costs = [data['avg'] for data in self.convergence_data]
        worst_costs = [data['worst'] for data in self.convergence_data]

        # Vẽ đường
        self.ax_convergence.plot(generations, best_costs, 'g-', label='Tốt nhất')
        self.ax_convergence.plot(generations, avg_costs, 'b-', label='Trung bình')
        self.ax_convergence.plot(generations, worst_costs, 'r-', label='Xấu nhất')

        self.ax_convergence.set_title("Giá trị hàm mục tiêu qua các thế hệ")
        self.ax_convergence.set_xlabel("Thế hệ")
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
            self.best_fitness_var.set(f"{latest['best']:.2f}")
            self.avg_fitness_var.set(f"{latest['avg']:.2f}")
            self.worst_fitness_var.set(f"{latest['worst']:.2f}")
            self.diversity_var.set(f"{latest['diversity']:.2f}%")

            # Tính số thế hệ không cải thiện
            stagnation = 0
            if len(best_costs) > 1:
                best_value = best_costs[-1]
                for i in range(len(best_costs)-2, -1, -1):
                    if abs(best_costs[i] - best_value) < 1e-6:
                        stagnation += 1
                    else:
                        break

            self.stagnation_var.set(str(stagnation))

            # Tỷ lệ thành công lai ghép (giả lập)
            success_rate = 100 - (generation / self.max_generations) * 30
            self.crossover_success_var.set(f"{success_rate:.1f}%")

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
            messagebox.showinfo("Hoàn thành", f"Thuật toán đã hoàn thành sau {len(self.convergence_data)} thế hệ")
        elif self.algorithm:
            messagebox.showinfo("Hoàn thành", f"Thuật toán đã hoàn thành sau {len(self.convergence_data)} thế hệ")

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