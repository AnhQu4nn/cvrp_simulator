"""
Thuật toán Di truyền cải tiến cho bài toán CVRP (Capacitated Vehicle Routing Problem)
"""

import numpy as np
import random
import time
import threading


class GeneticAlgorithm_CVRP:
    """Thuật toán Di truyền cho bài toán Định tuyến Phương tiện có Giới hạn Tải trọng (CVRP)"""

    def __init__(self, cvrp, population_size=50, mutation_rate=0.1, crossover_rate=0.8, elitism=5, max_generations=100,
                 selection_method="tournament", crossover_method="ordered", mutation_method="swap",
                 tournament_size=3, early_stopping=None):
        """
        Khởi tạo Thuật toán Di truyền cho CVRP

        Tham số:
        cvrp -- Đối tượng CVRP
        population_size -- Kích thước quần thể
        mutation_rate -- Xác suất đột biến
        crossover_rate -- Xác suất lai ghép
        elitism -- Số cá thể ưu tú giữ lại
        max_generations -- Số thế hệ tối đa
        selection_method -- Phương pháp chọn lọc ('tournament', 'roulette', 'rank')
        crossover_method -- Phương pháp lai ghép ('ordered', 'partially_mapped', 'cycle')
        mutation_method -- Phương pháp đột biến ('swap', 'insert', 'inversion', 'scramble')
        tournament_size -- Kích thước tournament (chỉ dùng khi selection_method là 'tournament')
        early_stopping -- Số thế hệ không cải thiện để dừng sớm (None nếu không dùng)
        """
        self.cvrp = cvrp
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.max_generations = max_generations
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.tournament_size = tournament_size
        self.early_stopping = early_stopping

        # Số lượng khách hàng
        self.n = len(cvrp.customers)

        # Lưu kết quả
        self.best_solution = None
        self.best_cost = float('inf')

        # Trạng thái cho trực quan hóa
        self.current_solution = None
        self.current_cost = float('inf')

        # Lịch sử chi phí và thời gian
        self.cost_history = []
        self.time_history = []
        self.avg_cost_history = []
        self.worst_cost_history = []
        self.diversity_history = []

        # Cờ dừng, tạm dừng và biến stagnation
        self.stop_flag = False
        self.paused = False
        self.pause_condition = threading.Condition()
        self.stagnation_count = 0
        self.was_stopped = False

    def run(self, callback=None, step_callback=None):
        """
        Chạy Thuật toán Di truyền

        Tham số:
        callback -- Hàm gọi lại khi hoàn thành
        step_callback -- Hàm gọi lại sau mỗi thế hệ
        """
        self.stop_flag = False
        self.was_stopped = False
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.time_history = []
        self.avg_cost_history = []
        self.worst_cost_history = []
        self.diversity_history = []
        self.stagnation_count = 0

        # Khởi tạo quần thể
        population = self.initialize_population()

        # Vòng lặp chính
        for generation in range(self.max_generations):
            # Kiểm tra dừng
            if self.stop_flag:
                self.was_stopped = True
                break

            # Kiểm tra tạm dừng
            with self.pause_condition:
                while self.paused and not self.stop_flag:
                    self.pause_condition.wait()

            # Bắt đầu tính thời gian
            start_time = time.time()

            # Đánh giá quần thể
            fitness_values = [self.evaluate_fitness(individual) for individual in population]
            best_idx = np.argmin(fitness_values)
            current_best_solution = self.decode_chromosome(population[best_idx])
            current_best_cost = fitness_values[best_idx]

            # Cập nhật giải pháp tốt nhất
            if current_best_cost < self.best_cost:
                self.best_solution = current_best_solution
                self.best_cost = current_best_cost
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Tính toán các thống kê
            avg_cost = sum(fitness_values) / len(fitness_values)
            worst_cost = max(fitness_values)
            diversity = self.calculate_diversity(population)

            # Lưu lịch sử
            self.cost_history.append(self.best_cost)
            self.avg_cost_history.append(avg_cost)
            self.worst_cost_history.append(worst_cost)
            self.diversity_history.append(diversity)

            # Lưu thời gian
            end_time = time.time()
            self.time_history.append(end_time - start_time)

            # Gọi hàm callback cho mỗi bước
            if step_callback:
                self.current_solution = current_best_solution
                self.current_cost = current_best_cost

                step_data = {
                    'generation': generation,
                    'progress': (generation + 1) / self.max_generations,
                    'solution': self.current_solution,
                    'cost': self.current_cost,
                    'best_solution': self.best_solution,
                    'best_cost': self.best_cost,
                    'avg_cost': avg_cost,
                    'worst_cost': worst_cost,
                    'diversity': diversity,
                    'population': population.copy(),
                    'fitness_values': fitness_values.copy(),
                    'cost_history': self.cost_history.copy(),
                    'avg_cost_history': self.avg_cost_history.copy(),
                    'worst_cost_history': self.worst_cost_history.copy(),
                    'diversity_history': self.diversity_history.copy(),
                    'time_history': self.time_history.copy(),
                    'stagnation': self.stagnation_count
                }
                step_callback(step_data)

            # Kiểm tra dừng sớm
            if self.early_stopping and self.stagnation_count >= self.early_stopping:
                break

            # Tạo quần thể mới
            new_population = []

            # Elitism - giữ lại các cá thể tốt nhất
            sorted_indices = np.argsort(fitness_values)
            for i in range(self.elitism):
                new_population.append(population[sorted_indices[i]])

            # Tạo cá thể mới cho quần thể tiếp theo
            while len(new_population) < self.population_size:
                # Chọn lọc
                if self.selection_method == "tournament":
                    parent1 = self.tournament_selection(population, fitness_values)
                    parent2 = self.tournament_selection(population, fitness_values)
                elif self.selection_method == "roulette":
                    parent1 = self.roulette_wheel_selection(population, fitness_values)
                    parent2 = self.roulette_wheel_selection(population, fitness_values)
                elif self.selection_method == "rank":
                    parent1 = self.rank_selection(population, fitness_values)
                    parent2 = self.rank_selection(population, fitness_values)
                else:  # Mặc định tournament
                    parent1 = self.tournament_selection(population, fitness_values)
                    parent2 = self.tournament_selection(population, fitness_values)

                # Lai ghép
                if random.random() < self.crossover_rate:
                    if self.crossover_method == "ordered":
                        child1, child2 = self.ordered_crossover(parent1, parent2)
                    elif self.crossover_method == "partially_mapped":
                        child1, child2 = self.partially_mapped_crossover(parent1, parent2)
                    elif self.crossover_method == "cycle":
                        child1, child2 = self.cycle_crossover(parent1, parent2)
                    else:  # Mặc định ordered
                        child1, child2 = self.ordered_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Đột biến
                if random.random() < self.mutation_rate:
                    if self.mutation_method == "swap":
                        self.swap_mutation(child1)
                    elif self.mutation_method == "insert":
                        self.insert_mutation(child1)
                    elif self.mutation_method == "inversion":
                        self.inversion_mutation(child1)
                    elif self.mutation_method == "scramble":
                        self.scramble_mutation(child1)
                    else:  # Mặc định swap
                        self.swap_mutation(child1)

                if random.random() < self.mutation_rate:
                    if self.mutation_method == "swap":
                        self.swap_mutation(child2)
                    elif self.mutation_method == "insert":
                        self.insert_mutation(child2)
                    elif self.mutation_method == "inversion":
                        self.inversion_mutation(child2)
                    elif self.mutation_method == "scramble":
                        self.scramble_mutation(child2)
                    else:  # Mặc định swap
                        self.swap_mutation(child2)

                # Thêm vào quần thể mới
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            # Cập nhật quần thể
            population = new_population

        # Kết thúc - gọi callback nếu có
        if callback:
            callback((self.best_solution, self.best_cost))

        return self.best_solution, self.best_cost

    def initialize_population(self):
        """
        Khởi tạo quần thể ban đầu

        Trả về:
        Danh sách các nhiễm sắc thể
        """
        population = []

        for _ in range(self.population_size):
            # Tạo hoán vị ngẫu nhiên của khách hàng
            chromosome = list(range(1, self.n))
            random.shuffle(chromosome)
            population.append(chromosome)

        return population

    def decode_chromosome(self, chromosome):
        """
        Giải mã nhiễm sắc thể thành giải pháp CVRP

        Tham số:
        chromosome -- Nhiễm sắc thể (hoán vị của khách hàng)

        Trả về:
        Danh sách các tuyến
        """
        solution = []
        route = []
        capacity_left = self.cvrp.capacity

        for customer in chromosome:
            demand = self.cvrp.customers[customer].demand

            # Nếu không đủ dung lượng, bắt đầu tuyến mới
            if demand > capacity_left:
                if route:  # Chỉ thêm tuyến nếu không rỗng
                    solution.append(route)
                route = [customer]
                capacity_left = self.cvrp.capacity - demand
            else:
                route.append(customer)
                capacity_left -= demand

        # Thêm tuyến cuối cùng nếu có
        if route:
            solution.append(route)

        return solution

    def evaluate_fitness(self, chromosome):
        """
        Đánh giá độ thích nghi của một nhiễm sắc thể

        Tham số:
        chromosome -- Nhiễm sắc thể để đánh giá

        Trả về:
        Giá trị thích nghi (chi phí, thấp hơn là tốt hơn)
        """
        solution = self.decode_chromosome(chromosome)
        cost = self.cvrp.calculate_solution_cost(solution)
        return cost

    def calculate_diversity(self, population):
        """
        Tính đa dạng của quần thể dựa trên khoảng cách Hamming trung bình

        Tham số:
        population -- Quần thể

        Trả về:
        Phần trăm đa dạng
        """
        if not population or len(population) < 2:
            return 0

        total_diff = 0
        count = 0

        for i in range(len(population)):
            for j in range(i+1, len(population)):
                # Đếm số vị trí khác nhau
                diff = sum(1 for a, b in zip(population[i], population[j]) if a != b)
                total_diff += diff
                count += 1

        # Trả về phần trăm đa dạng
        if count > 0 and len(population[0]) > 0:
            return (total_diff / count / len(population[0])) * 100
        else:
            return 0

    # Các phương pháp chọn lọc
    def tournament_selection(self, population, fitness_values):
        """
        Chọn lọc tournament

        Tham số:
        population -- Quần thể
        fitness_values -- Giá trị thích nghi tương ứng

        Trả về:
        Cá thể được chọn
        """
        # Kích thước tournament
        k = self.tournament_size

        # Chọn ngẫu nhiên k cá thể
        selected_indices = random.sample(range(len(population)), k)
        tournament_fitness = [fitness_values[i] for i in selected_indices]

        # Chọn cá thể tốt nhất từ tournament
        best_idx = selected_indices[np.argmin(tournament_fitness)]
        return population[best_idx].copy()

    def roulette_wheel_selection(self, population, fitness_values):
        """
        Chọn lọc bánh xe roulette

        Tham số:
        population -- Quần thể
        fitness_values -- Giá trị thích nghi tương ứng

        Trả về:
        Cá thể được chọn
        """
        # Chuyển fitness từ minimize thành maximize bằng cách lấy nghịch đảo
        max_fitness = max(fitness_values)
        inv_fitness = [max_fitness - fitness + 0.01 for fitness in fitness_values]  # +0.01 để tránh 0
        total_fitness = sum(inv_fitness)

        if total_fitness == 0:
            # Nếu tổng bằng 0, chọn ngẫu nhiên
            return random.choice(population).copy()

        # Tạo xác suất chọn
        selection_probs = [fit / total_fitness for fit in inv_fitness]

        # Chọn dựa trên xác suất
        selected_idx = np.random.choice(len(population), p=selection_probs)
        return population[selected_idx].copy()

    def rank_selection(self, population, fitness_values):
        """
        Chọn lọc theo thứ hạng

        Tham số:
        population -- Quần thể
        fitness_values -- Giá trị thích nghi tương ứng

        Trả về:
        Cá thể được chọn
        """
        # Sắp xếp các chỉ số theo thứ tự tăng dần của fitness (tốt nhất đầu tiên)
        sorted_indices = np.argsort(fitness_values)

        # Gán thứ hạng (thứ hạng cao hơn cho cá thể tốt hơn)
        ranks = np.zeros(len(population))
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = len(population) - rank

        # Sử dụng thứ hạng làm xác suất chọn
        total_rank = sum(ranks)
        selection_probs = [rank / total_rank for rank in ranks]

        # Chọn dựa trên xác suất
        selected_idx = np.random.choice(len(population), p=selection_probs)
        return population[selected_idx].copy()

    # Các phương pháp lai ghép
    def ordered_crossover(self, parent1, parent2):
        """
        Lai ghép thứ tự (OX)

        Tham số:
        parent1, parent2 -- Hai nhiễm sắc thể cha mẹ

        Trả về:
        Hai nhiễm sắc thể con
        """
        # Lai ghép thứ tự (OX)
        size = len(parent1)

        # Chọn hai điểm lai ghép
        point1 = random.randint(0, size - 2)
        point2 = random.randint(point1 + 1, size - 1)

        # Tạo mask cho đoạn giữa hai điểm
        mask = [False] * size
        for i in range(point1, point2 + 1):
            mask[i] = True

        # Tạo hai nhiễm sắc thể con
        child1 = [-1] * size
        child2 = [-1] * size

        # Sao chép đoạn giữa hai điểm
        for i in range(point1, point2 + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]

        # Điền phần còn lại
        self.fill_ox(parent2, child1, mask)
        self.fill_ox(parent1, child2, mask)

        return child1, child2

    def fill_ox(self, parent, child, mask):
        """
        Điền phần còn lại của nhiễm sắc thể con với giá trị từ cha mẹ

        Tham số:
        parent -- Nhiễm sắc thể cha mẹ
        child -- Nhiễm sắc thể con (đã điền một phần)
        mask -- Mask chỉ ra những vị trí đã điền
        """
        size = len(parent)
        j = 0  # Chỉ số trong cha mẹ

        for i in range(size):
            if not mask[i]:  # Nếu vị trí chưa điền
                # Tìm giá trị tiếp theo từ cha mẹ không có trong con
                while parent[j] in child:
                    j = (j + 1) % size

                child[i] = parent[j]
                j = (j + 1) % size

    def partially_mapped_crossover(self, parent1, parent2):
        """
        Lai ghép ánh xạ một phần (PMX)

        Tham số:
        parent1, parent2 -- Hai nhiễm sắc thể cha mẹ

        Trả về:
        Hai nhiễm sắc thể con
        """
        size = len(parent1)

        # Chọn hai điểm lai ghép
        point1 = random.randint(0, size - 2)
        point2 = random.randint(point1 + 1, size - 1)

        # Tạo bản sao cho con
        child1 = [-1] * size
        child2 = [-1] * size

        # Sao chép đoạn giữa hai điểm
        for i in range(point1, point2 + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]

        # Tạo ánh xạ
        mapping1 = {}  # parent1[i] -> parent2[i]
        mapping2 = {}  # parent2[i] -> parent1[i]

        for i in range(point1, point2 + 1):
            mapping1[parent1[i]] = parent2[i]
            mapping2[parent2[i]] = parent1[i]

        # Điền những vị trí còn lại cho child1
        for i in range(size):
            if i < point1 or i > point2:
                value = parent2[i]
                while value in child1:
                    value = mapping2[value]
                child1[i] = value

        # Điền những vị trí còn lại cho child2
        for i in range(size):
            if i < point1 or i > point2:
                value = parent1[i]
                while value in child2:
                    value = mapping1[value]
                child2[i] = value

        return child1, child2

    def cycle_crossover(self, parent1, parent2):
        """
        Lai ghép chu trình (CX)

        Tham số:
        parent1, parent2 -- Hai nhiễm sắc thể cha mẹ

        Trả về:
        Hai nhiễm sắc thể con
        """
        size = len(parent1)

        # Tạo bản sao cho con
        child1 = [-1] * size
        child2 = [-1] * size

        # Tạo ánh xạ vị trí
        positions = {}
        for i in range(size):
            positions[parent2[i]] = i

        # Tạo mask để theo dõi các chu trình
        visited = [False] * size

        # Xử lý từng chu trình
        for i in range(size):
            if not visited[i]:
                # Bắt đầu một chu trình mới
                j = i
                cycle_mod = 0  # 0 = parent1 cho child1, 1 = parent2 cho child1

                while not visited[j]:
                    visited[j] = True

                    # Đặt giá trị cho con dựa trên chu trình
                    if cycle_mod == 0:
                        child1[j] = parent1[j]
                        child2[j] = parent2[j]
                    else:
                        child1[j] = parent2[j]
                        child2[j] = parent1[j]

                    # Tìm vị trí tiếp theo trong chu trình
                    j = positions[parent1[j]]

                # Chuyển đổi cycle_mod cho chu trình tiếp theo
                cycle_mod = 1 - cycle_mod

        return child1, child2

    # Các phương pháp đột biến
    def swap_mutation(self, chromosome):
        """
        Đột biến hoán đổi

        Tham số:
        chromosome -- Nhiễm sắc thể để đột biến
        """
        size = len(chromosome)

        # Chọn hai vị trí ngẫu nhiên để hoán đổi
        idx1 = random.randint(0, size - 1)
        idx2 = random.randint(0, size - 1)

        # Hoán đổi
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    def insert_mutation(self, chromosome):
        """
        Đột biến chèn

        Tham số:
        chromosome -- Nhiễm sắc thể để đột biến
        """
        size = len(chromosome)

        # Chọn hai vị trí ngẫu nhiên
        idx1 = random.randint(0, size - 1)
        idx2 = random.randint(0, size - 1)

        # Đảm bảo idx1 < idx2
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        if idx1 != idx2:
            # Lấy giá trị tại idx2
            value = chromosome[idx2]

            # Di chuyển tất cả các giá trị trong khoảng (idx1, idx2) lên một vị trí
            for i in range(idx2, idx1, -1):
                chromosome[i] = chromosome[i-1]

            # Chèn giá trị vào idx1
            chromosome[idx1] = value

    def inversion_mutation(self, chromosome):
        """
        Đột biến đảo ngược

        Tham số:
        chromosome -- Nhiễm sắc thể để đột biến
        """
        size = len(chromosome)

        # Chọn hai vị trí ngẫu nhiên
        idx1 = random.randint(0, size - 2)
        idx2 = random.randint(idx1 + 1, size - 1)

        # Đảo ngược đoạn từ idx1 đến idx2
        chromosome[idx1:idx2+1] = reversed(chromosome[idx1:idx2+1])

    def scramble_mutation(self, chromosome):
        """
        Đột biến xáo trộn

        Tham số:
        chromosome -- Nhiễm sắc thể để đột biến
        """
        size = len(chromosome)

        # Chọn hai vị trí ngẫu nhiên
        idx1 = random.randint(0, size - 2)
        idx2 = random.randint(idx1 + 1, size - 1)

        # Lấy đoạn cần xáo trộn
        segment = chromosome[idx1:idx2+1]

        # Xáo trộn đoạn
        random.shuffle(segment)

        # Đặt lại đoạn đã xáo trộn
        chromosome[idx1:idx2+1] = segment

    def stop(self):
        """Dừng thuật toán"""
        self.stop_flag = True

    def pause(self):
        """Tạm dừng thuật toán"""
        self.paused = True

    def resume(self):
        """Tiếp tục thuật toán"""
        with self.pause_condition:
            self.paused = False
            self.pause_condition.notify_all()