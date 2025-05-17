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
                 tournament_size=3, early_stopping=None, local_search=False):
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
        local_search -- Sử dụng tìm kiếm cục bộ
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
        self.local_search = local_search

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

            # Bắt đầu tính thời gian tính toán thuần túy
            start_time = time.time()

            # Đánh giá quần thể
            fitness_values = [self.evaluate_fitness(individual) for individual in population]
            best_idx = np.argmin(fitness_values)
            current_best_solution = self.decode_chromosome_with_feasibility_check(population[best_idx])
            current_best_cost = fitness_values[best_idx]

            # Kiểm tra và sửa chữa giải pháp tốt nhất nếu cần
            if not self.cvrp.is_solution_valid(current_best_solution) or not self.check_solution_feasibility(current_best_solution):
                current_best_solution = self.repair_solution(current_best_solution)
                # Đảm bảo tính khả thi của giải pháp đã sửa chữa
                if not self.check_solution_feasibility(current_best_solution):
                    # Nếu vẫn không khả thi, chia thành các tuyến riêng biệt
                    new_solution = []
                    for route in current_best_solution:
                        for customer in route:
                            new_solution.append([customer])
                    current_best_solution = new_solution
                # Cập nhật lại chi phí
                current_best_cost = self.cvrp.calculate_solution_cost(current_best_solution)
            
            # Áp dụng tìm kiếm cục bộ nếu được kích hoạt
            if self.local_search and current_best_solution:
                improved_solution = self.local_search_2opt(current_best_solution)
                # Cập nhật lại chi phí sau khi áp dụng tìm kiếm cục bộ
                improved_cost = self.cvrp.calculate_solution_cost(improved_solution)
                if improved_cost < current_best_cost:
                    current_best_solution = improved_solution
                    current_best_cost = improved_cost

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

            # Lưu thời gian và kết thúc đo thời gian tính toán thuần túy
            end_time = time.time()
            computation_time = end_time - start_time
            self.time_history.append(computation_time)

            # Gọi hàm callback cho mỗi bước và truyền thời gian tính toán thuần túy
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
                    'stagnation': self.stagnation_count,
                    'computation_time': computation_time,  # Thời gian tính toán thuần túy
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
                
                # Kiểm tra và sửa chữa nhiễm sắc thể sau khi lai ghép
                child1 = self.check_and_repair_chromosomes(child1)
                child2 = self.check_and_repair_chromosomes(child2)
                
                # Kiểm tra tính khả thi của giải pháp sau khi lai ghép
                solution1 = self.decode_chromosome(child1)
                solution2 = self.decode_chromosome(child2)
                
                if not self.check_solution_feasibility(solution1):
                    # Giải pháp không khả thi, cần sửa chữa
                    # Sử dụng lại phương pháp decode có kiểm tra khả thi
                    solution1 = self.decode_chromosome_with_feasibility_check(child1)
                
                if not self.check_solution_feasibility(solution2):
                    # Giải pháp không khả thi, cần sửa chữa
                    solution2 = self.decode_chromosome_with_feasibility_check(child2)

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
                    
                    # Kiểm tra và sửa chữa sau khi đột biến
                    child1 = self.check_and_repair_chromosomes(child1)
                    
                    # Kiểm tra tính khả thi sau đột biến
                    solution1 = self.decode_chromosome(child1)
                    if not self.check_solution_feasibility(solution1):
                        # Giải pháp không khả thi, cần sửa chữa
                        solution1 = self.decode_chromosome_with_feasibility_check(child1)

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
                    
                    # Kiểm tra và sửa chữa sau khi đột biến
                    child2 = self.check_and_repair_chromosomes(child2)
                    
                    # Kiểm tra tính khả thi sau đột biến
                    solution2 = self.decode_chromosome(child2)
                    if not self.check_solution_feasibility(solution2):
                        # Giải pháp không khả thi, cần sửa chữa
                        solution2 = self.decode_chromosome_with_feasibility_check(child2)

                # Thêm vào quần thể mới
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            # Cập nhật quần thể
            population = new_population

        # Đảm bảo giải pháp tốt nhất cuối cùng là hợp lệ và khả thi
        if self.best_solution:
            if not self.cvrp.is_solution_valid(self.best_solution) or not self.check_solution_feasibility(self.best_solution):
                self.best_solution = self.repair_solution(self.best_solution)
                # Kiểm tra lại tính khả thi
                if not self.check_solution_feasibility(self.best_solution):
                    # Nếu vẫn không khả thi, chia thành các tuyến riêng biệt
                    new_solution = []
                    for route in self.best_solution:
                        for customer in route:
                            new_solution.append([customer])
                    self.best_solution = new_solution
                self.best_cost = self.cvrp.calculate_solution_cost(self.best_solution)

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
        Giải mã nhiễm sắc thể thành giải pháp CVRP đảm bảo ràng buộc về sức chứa
        
        Tham số:
        chromosome -- Nhiễm sắc thể (hoán vị của khách hàng)
        
        Trả về:
        Danh sách các tuyến hợp lệ
        """
        solution = []
        
        for customer in chromosome:
            customer_demand = self.cvrp.customers[customer].demand
            
            if customer_demand > self.cvrp.capacity:
                solution.append([customer])
                # print(f"Warning: Customer {customer} has demand {customer_demand} exceeding vehicle capacity {self.cvrp.capacity}")
                continue
                
            best_route_idx = -1
            # best_remaining = -1 # Không còn cần thiết nếu chỉ dựa vào best_insertion_cost
            min_overall_insertion_cost = float('inf')
            best_pos_for_min_overall_cost = -1
            
            for i, route in enumerate(solution):
                current_demand = sum(self.cvrp.customers[c].demand for c in route)
                remaining_capacity_in_route = self.cvrp.capacity - current_demand
                
                if remaining_capacity_in_route < customer_demand:
                    continue
                    
                original_route_distance = self.cvrp.calculate_route_distance(route)
                current_route_best_insertion_cost = float('inf')
                current_route_best_pos = -1
                
                for pos in range(len(route) + 1):
                    new_route_candidate = route[:pos] + [customer] + route[pos:]
                    new_candidate_distance = self.cvrp.calculate_route_distance(new_route_candidate)
                    insertion_cost_for_this_pos = new_candidate_distance - original_route_distance
                    
                    if insertion_cost_for_this_pos < current_route_best_insertion_cost:
                        current_route_best_insertion_cost = insertion_cost_for_this_pos
                        current_route_best_pos = pos
                
                # Sau khi duyệt hết các vị trí trong tuyến `i`, current_route_best_insertion_cost là chi phí tốt nhất để chèn vào tuyến `i`
                # và current_route_best_pos là vị trí tương ứng.
                # Bây giờ so sánh nó với chi phí chèn tốt nhất tổng thể đã tìm thấy.
                if current_route_best_insertion_cost < min_overall_insertion_cost:
                    min_overall_insertion_cost = current_route_best_insertion_cost
                    best_route_idx = i
                    best_pos_for_min_overall_cost = current_route_best_pos
            
            if best_route_idx != -1:
                # Chèn khách hàng vào tuyến đã chọn tại vị trí tốt nhất đã tìm thấy cho tuyến đó
                route_to_insert_into = solution[best_route_idx]
                solution[best_route_idx] = route_to_insert_into[:best_pos_for_min_overall_cost] + [customer] + route_to_insert_into[best_pos_for_min_overall_cost:]
            else:
                solution.append([customer])
        
        return solution

    def check_and_repair_capacity(self, solution):
        """
        Kiểm tra và sửa chữa giải pháp không hợp lệ về ràng buộc sức chứa
        bằng cách chuyển khách hàng giữa các tuyến để giảm thiểu vi phạm
        
        Phương pháp này sẽ được sử dụng trước khi tính toán độ thích nghi cuối cùng
        
        Tham số:
        solution -- Giải pháp cần kiểm tra và sửa chữa (danh sách các tuyến)
        
        Trả về:
        Giải pháp đã sửa chữa
        """
        # Kiểm tra xem có tuyến nào vượt quá capacity
        invalid_routes = []
        valid_routes = []
        
        for i, route in enumerate(solution):
            demand = sum(self.cvrp.customers[customer].demand for customer in route)
            if demand > self.cvrp.capacity:
                invalid_routes.append((i, route, demand))
            else:
                valid_routes.append((i, route, demand))
        
        # Nếu không có tuyến không hợp lệ, trả về giải pháp ban đầu
        if not invalid_routes:
            return solution
        
        # Thực hiện hoán đổi khách hàng giữa các tuyến để tối ưu hóa
        # Ưu tiên di chuyển khách hàng từ các tuyến vượt quá capacity
        
        # Sắp xếp tuyến không hợp lệ theo mức độ vượt quá (giảm dần)
        invalid_routes.sort(key=lambda x: x[2] - self.cvrp.capacity, reverse=True)
        
        # Tạo danh sách tuyến mới
        new_solution = [route for _, route, _ in valid_routes + invalid_routes]
        
        # Thực hiện cải tiến lặp đi lặp lại
        max_iterations = 10  # Giới hạn số lần lặp để tránh chạy quá lâu
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i, route_i in enumerate(new_solution):
                # Tính demand của tuyến hiện tại
                demand_i = sum(self.cvrp.customers[customer].demand for customer in route_i)
                
                # Nếu tuyến không vượt quá capacity, bỏ qua
                if demand_i <= self.cvrp.capacity:
                    continue
                
                # Tìm khách hàng để chuyển
                for pos_i, customer in enumerate(route_i):
                    customer_demand = self.cvrp.customers[customer].demand
                    
                    # Tìm tuyến khác để chuyển khách hàng này vào
                    for j, route_j in enumerate(new_solution):
                        if i == j:  # Không chuyển trong cùng một tuyến
                            continue
                        
                        # Tính demand của tuyến đích
                        demand_j = sum(self.cvrp.customers[c].demand for c in route_j)
                        
                        # Kiểm tra xem có thể thêm khách hàng vào tuyến j không
                        if demand_j + customer_demand <= self.cvrp.capacity:
                            # Tìm vị trí tốt nhất để chèn vào route_j
                            best_pos = 0
                            best_cost = float('inf')
                            
                            for pos_j in range(len(route_j) + 1):
                                # Tạo tuyến mới
                                new_route_j = route_j[:pos_j] + [customer] + route_j[pos_j:]
                                cost = self.cvrp.calculate_route_distance(new_route_j)
                                
                                if cost < best_cost:
                                    best_cost = cost
                                    best_pos = pos_j
                            
                            # Thực hiện chuyển khách hàng
                            new_solution[j] = route_j[:best_pos] + [customer] + route_j[best_pos:]
                            new_solution[i] = route_i[:pos_i] + route_i[pos_i+1:]
                            improved = True
                            
                            # Break để tránh chuyển quá nhiều khách hàng cùng lúc
                            break
                    
                    if improved:
                        # Break để tính toán lại demand cho tất cả các tuyến
                        break
                
                if improved:
                    # Break để xem xét các tuyến vẫn còn vượt quá
                    break
        
        # Loại bỏ các tuyến rỗng
        new_solution = [route for route in new_solution if route]
        
        return new_solution

    def evaluate_fitness(self, chromosome):
        """
        Đánh giá độ thích nghi của một nhiễm sắc thể với phạt cho giải pháp không hợp lệ
        
        Tham số:
        chromosome -- Nhiễm sắc thể để đánh giá
        
        Trả về:
        Giá trị thích nghi (chi phí, thấp hơn là tốt hơn)
        """
        # Giải mã nhiễm sắc thể thành giải pháp và kiểm tra tính khả thi
        solution = self.decode_chromosome_with_feasibility_check(chromosome)
        
        # Tính tổng chi phí của giải pháp
        cost = self.cvrp.calculate_solution_cost(solution)
        
        # Kiểm tra tính hợp lệ và áp dụng phạt nếu cần
        total_penalty = 0
        
        # Phạt cho vi phạm ràng buộc sức chứa
        for route in solution:
            route_demand = sum(self.cvrp.customers[customer].demand for customer in route)
            if route_demand > self.cvrp.capacity:
                # Phạt tỷ lệ với mức độ vi phạm
                excess = route_demand - self.cvrp.capacity
                penalty = excess * 100  # Hệ số phạt lớn
                total_penalty += penalty
        
        # Phạt cho số lượng tuyến quá nhiều (ưu tiên ít tuyến hơn)
        if len(solution) > len(self.cvrp.customers) / 3:  # Một ngưỡng hợp lý
            routes_penalty = (len(solution) - len(self.cvrp.customers) / 3) * 50
            total_penalty += routes_penalty
        
        # Trả về chi phí với phạt (nếu có)
        return cost + total_penalty

    def local_search_2opt(self, solution):
        """
        Áp dụng tìm kiếm cục bộ 2-opt cho mỗi tuyến

        Tham số:
        solution -- Giải pháp (danh sách các tuyến)

        Trả về:
        Giải pháp cải tiến
        """
        improved_solution = []

        for route in solution:
            # Bỏ qua các tuyến quá ngắn
            if len(route) <= 2:
                improved_solution.append(route)
                continue

            # Áp dụng 2-opt cho tuyến
            improved_route = self.apply_2opt(route)
            improved_solution.append(improved_route)

        return improved_solution

    def apply_2opt(self, route):
        """
        Áp dụng tìm kiếm 2-opt cho một tuyến đơn

        Tham số:
        route -- Tuyến ban đầu

        Trả về:
        Tuyến cải tiến
        """
        if not route or len(route) < 2: # Thêm kiểm tra cho tuyến rỗng hoặc quá ngắn
            return route

        best_route = route.copy()
        improved = True

        while improved:
            improved = False
            best_distance = self.cvrp.calculate_route_distance(best_route) # MODIFIED

            for i in range(len(best_route) - 1): # Sửa: lặp trên best_route
                for j in range(i + 1, len(best_route)): # Sửa: lặp trên best_route
                    # Đảo ngược đoạn từ i+1 đến j (bao gồm cả hai đầu)
                    # new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    # Logic đảo ngược chính xác hơn cho 2-opt:
                    # Tuyến mới được hình thành bằng cách lấy: route[0...i-1] + route[i...j reversed] + route[j+1...end]
                    # Ví dụ: 0-1-2-3-4-5. i=1, j=3. Segment là 1-2-3. Reversed là 3-2-1.
                    # new_route = 0-3-2-1-4-5
                    segment_to_reverse = best_route[i+1:j+1]
                    if not segment_to_reverse: # Nếu đoạn rỗng, bỏ qua
                        continue
                    
                    new_route = best_route[:i+1] + segment_to_reverse[::-1] + best_route[j+1:]
                    current_distance = self.cvrp.calculate_route_distance(new_route) # MODIFIED

                    if current_distance < best_distance:
                        best_route = new_route
                        best_distance = current_distance
                        improved = True
            
            # Nếu không có cải thiện trong vòng lặp này, thoát
            # Điều này làm cho 2-opt dừng sau lượt lặp đầu tiên tìm thấy cải thiện.
            # Để 2-opt triệt để hơn, loại bỏ break này và để vòng while kiểm soát.
            # if improved:
            # break
            # Tuy nhiên, để nhất quán với đánh giá trước, tôi sẽ giữ lại logic dừng sớm
            # cho đến khi có yêu cầu thay đổi cụ thể hơn.
            # Theo yêu cầu mới là làm cho 2-opt triệt để hơn, nên loại bỏ break này
            if not improved: # Nếu không có cải thiện nào trong cả một lượt quét i, j thì mới dừng
                 break

        return best_route

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
        child1 = parent2.copy()
        child2 = parent1.copy()

        # Tạo danh sách các gene đã được ánh xạ
        mapped_genes1 = set()
        mapped_genes2 = set()

        # Sao chép đoạn giữa hai điểm và xác định các gene đã được ánh xạ
        for i in range(point1, point2 + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
            mapped_genes1.add(parent1[i])
            mapped_genes2.add(parent2[i])

        # Xử lý xung đột cho child1
        for i in range(size):
            if i < point1 or i > point2:
                if child1[i] in mapped_genes1:
                    # Tìm gene thay thế không có trong danh sách đã ánh xạ
                    conflicts = set()
                    current = child1[i]
                    
                    # Tìm tất cả các xung đột có thể có
                    while current in mapped_genes1:
                        # Tìm vị trí của gene trong parent1
                        for j in range(point1, point2 + 1):
                            if parent1[j] == current:
                                # Thay thế bằng gene tương ứng trong parent2
                                current = parent2[j]
                                conflicts.add(current)
                                break
                    
                    # Áp dụng gene thay thế
                    child1[i] = current

        # Xử lý xung đột cho child2
        for i in range(size):
            if i < point1 or i > point2:
                if child2[i] in mapped_genes2:
                    # Tìm gene thay thế không có trong danh sách đã ánh xạ
                    conflicts = set()
                    current = child2[i]
                    
                    # Tìm tất cả các xung đột có thể có
                    while current in mapped_genes2:
                        # Tìm vị trí của gene trong parent2
                        for j in range(point1, point2 + 1):
                            if parent2[j] == current:
                                # Thay thế bằng gene tương ứng trong parent1
                                current = parent1[j]
                                conflicts.add(current)
                                break
                    
                    # Áp dụng gene thay thế
                    child2[i] = current

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

    def repair_solution(self, solution):
        """
        Sửa chữa giải pháp không hợp lệ bằng cách phân phối lại khách hàng giữa các tuyến
        
        Tham số:
        solution -- Giải pháp cần sửa chữa (danh sách các tuyến)
        
        Trả về:
        Giải pháp đã sửa chữa
        """
        # Kiểm tra xem có tuyến nào vượt quá capacity
        invalid_routes = []
        valid_routes = []
        
        for i, route in enumerate(solution):
            demand = sum(self.cvrp.customers[customer].demand for customer in route)
            if demand > self.cvrp.capacity:
                invalid_routes.append((i, route, demand))
            else:
                valid_routes.append((i, route, demand))
        
        # Nếu không có tuyến không hợp lệ, trả về giải pháp ban đầu
        if not invalid_routes:
            return solution
            
        # Sắp xếp tuyến không hợp lệ theo mức độ vượt quá (giảm dần)
        invalid_routes.sort(key=lambda x: x[2], reverse=True)
        
        # Tạo giải pháp mới
        new_solution = [route for _, route, _ in valid_routes]
        
        # Xử lý từng tuyến không hợp lệ
        for _, route, _ in invalid_routes:
            # Sắp xếp khách hàng trong tuyến theo demand (giảm dần)
            sorted_customers = sorted(route, key=lambda c: self.cvrp.customers[c].demand, reverse=True)
            
            # Thử chèn vào các tuyến hợp lệ hoặc tạo tuyến mới
            for customer in sorted_customers:
                customer_demand = self.cvrp.customers[customer].demand
                inserted = False
                
                # Tìm tuyến tốt nhất để chèn (best-fit)
                best_route_idx = -1
                best_remaining = -1
                
                for i, current_route in enumerate(new_solution):
                    current_demand = sum(self.cvrp.customers[c].demand for c in current_route)
                    remaining = self.cvrp.capacity - current_demand
                    
                    # Nếu có thể chèn và còn nhiều dung lượng hơn
                    if customer_demand <= remaining and remaining > best_remaining:
                        best_route_idx = i
                        best_remaining = remaining
                
                # Nếu tìm thấy tuyến phù hợp
                if best_route_idx >= 0:
                    # Tìm vị trí tốt nhất để chèn (cheapest insertion)
                    best_position = 0
                    min_increase = float('inf')
                    route = new_solution[best_route_idx]
                    
                    # Thử các vị trí có thể
                    for pos in range(len(route) + 1):
                        # Tạo tuyến mới với khách hàng được chèn
                        new_route = route[:pos] + [customer] + route[pos:]
                        new_distance = self.cvrp.calculate_route_distance(new_route)
                        old_distance = self.cvrp.calculate_route_distance(route)
                        increase = new_distance - old_distance
                        
                        if increase < min_increase:
                            min_increase = increase
                            best_position = pos
                    
                    # Chèn vào vị trí tốt nhất
                    new_solution[best_route_idx] = route[:best_position] + [customer] + route[best_position:]
                    inserted = True
                
                # Nếu không tìm thấy tuyến phù hợp, tạo tuyến mới
                if not inserted:
                    new_solution.append([customer])
        
        return new_solution

    def check_and_repair_chromosomes(self, chromosome):
        """
        Kiểm tra và sửa chữa nhiễm sắc thể sau khi lai ghép hoặc đột biến
        
        Đảm bảo mỗi khách hàng xuất hiện đúng một lần (không trùng lặp, không thiếu)
        
        Tham số:
        chromosome -- Nhiễm sắc thể cần kiểm tra và sửa chữa
        
        Trả về:
        Nhiễm sắc thể đã sửa chữa
        """
        # Tất cả các khách hàng cần có
        all_customers = set(range(1, self.n))
        
        # Khách hàng hiện có trong nhiễm sắc thể
        existing_customers = set(chromosome)
        
        # Kiểm tra trùng lặp
        if len(chromosome) != len(existing_customers):
            # Có khách hàng bị trùng lặp
            duplicates = []
            seen = set()
            
            for customer in chromosome:
                if customer in seen:
                    duplicates.append(customer)
                else:
                    seen.add(customer)
            
            # Thay thế các khách hàng trùng lặp bằng các khách hàng bị thiếu
            missing_customers = all_customers - existing_customers
            
            if missing_customers:
                # Có khách hàng bị thiếu
                for i, customer in enumerate(chromosome):
                    if customer in duplicates:
                        # Thay thế bằng khách hàng bị thiếu
                        if missing_customers:
                            missing = missing_customers.pop()
                            chromosome[i] = missing
                            duplicates.remove(customer)
        
        # Kiểm tra thiếu
        missing_customers = all_customers - set(chromosome)
        
        if missing_customers:
            # Có khách hàng bị thiếu, nhưng không có trùng lặp
            # (Trường hợp này có thể xảy ra nếu có khách hàng ngoài phạm vi)
            extras = set(chromosome) - all_customers
            
            if extras:
                # Thay thế các khách hàng không hợp lệ bằng các khách hàng bị thiếu
                for i, customer in enumerate(chromosome):
                    if customer in extras:
                        # Thay thế bằng khách hàng bị thiếu
                        if missing_customers:
                            missing = missing_customers.pop()
                            chromosome[i] = missing
                            extras.remove(customer)
        
        return chromosome

    def check_route_feasibility(self, route):
        """
        Kiểm tra tính khả thi của một tuyến đường
        Đảm bảo rằng có thể đi từ một khách hàng đến khách hàng kế tiếp
        
        Tham số:
        route -- Tuyến đường cần kiểm tra
        
        Trả về:
        True nếu tuyến đường khả thi, False nếu không
        """
        if not route:
            return True
            
        # Kiểm tra khoảng cách giữa các khách hàng liên tiếp
        for i in range(len(route) - 1):
            current = route[i]
            next_customer = route[i + 1]
            
            # Kiểm tra nếu có đường nối giữa hai khách hàng (khoảng cách != 0)
            if self.cvrp.distances[current][next_customer] == 0:
                return False
                
            # Kiểm tra khoảng cách có quá lớn không (có thể điều chỉnh ngưỡng tùy theo bài toán)
            if self.cvrp.distances[current][next_customer] > 1000:  # Ngưỡng khoảng cách tối đa
                return False
                
        # Kiểm tra khoảng cách từ depot đến khách hàng đầu tiên
        if self.cvrp.distances[0][route[0]] == 0:
            return False
            
        # Kiểm tra khoảng cách từ khách hàng cuối cùng về depot
        if self.cvrp.distances[route[-1]][0] == 0:
            return False
            
        return True
        
    def check_and_repair_capacity_for_route(self, route):
        """
        Kiểm tra và sửa chữa một tuyến đường đảm bảo không vượt quá sức chứa
        
        Tham số:
        route -- Tuyến đường cần kiểm tra và sửa chữa
        
        Trả về:
        Danh sách các tuyến đường sau khi sửa chữa
        """
        if not route:
            return []
            
        # Tính tổng demand của tuyến
        total_demand = sum(self.cvrp.customers[customer].demand for customer in route)
        
        # Nếu không vượt quá capacity, giữ nguyên tuyến
        if total_demand <= self.cvrp.capacity:
            return [route]
            
        # Nếu vượt quá, chia thành nhiều tuyến
        result_routes = []
        current_route = []
        current_demand = 0
        
        for customer in route:
            customer_demand = self.cvrp.customers[customer].demand
            
            # Nếu thêm khách hàng này vào sẽ vượt quá capacity
            if current_demand + customer_demand > self.cvrp.capacity:
                # Thêm tuyến hiện tại vào kết quả nếu không rỗng
                if current_route:
                    result_routes.append(current_route)
                    
                # Bắt đầu tuyến mới
                current_route = [customer]
                current_demand = customer_demand
            else:
                # Thêm khách hàng vào tuyến hiện tại
                current_route.append(customer)
                current_demand += customer_demand
                
        # Thêm tuyến cuối cùng vào kết quả nếu không rỗng
        if current_route:
            result_routes.append(current_route)
            
        return result_routes
        
    def check_solution_feasibility(self, solution):
        """
        Kiểm tra tính khả thi của giải pháp
        
        Tham số:
        solution -- Giải pháp cần kiểm tra
        
        Trả về:
        True nếu giải pháp khả thi, False nếu không
        """
        # Kiểm tra từng tuyến đường
        for route in solution:
            if not self.check_route_feasibility(route):
                return False
                
        return True
        
    def decode_chromosome_with_feasibility_check(self, chromosome):
        """
        Giải mã nhiễm sắc thể thành giải pháp CVRP và kiểm tra tính khả thi
        
        Tham số:
        chromosome -- Nhiễm sắc thể (hoán vị của khách hàng)
        
        Trả về:
        Giải pháp hợp lệ và khả thi
        """
        # Giải mã nhiễm sắc thể thành giải pháp
        solution = self.decode_chromosome(chromosome)
        
        # Kiểm tra và sửa chữa tính khả thi của từng tuyến đường
        new_solution = []
        
        for route in solution:
            # Kiểm tra tính khả thi của tuyến
            if self.check_route_feasibility(route):
                # Kiểm tra và sửa chữa ràng buộc về sức chứa
                fixed_routes = self.check_and_repair_capacity_for_route(route)
                new_solution.extend(fixed_routes)
            else:
                # Nếu tuyến không khả thi, chia thành các tuyến nhỏ hơn
                # Mỗi khách hàng đi riêng một tuyến
                for customer in route:
                    new_solution.append([customer])
        
        return new_solution