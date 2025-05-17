"""
Thuật toán Ant Colony Optimization cải tiến cho bài toán CVRP
(Capacitated Vehicle Routing Problem)
"""

import numpy as np
import random
import time
import threading
import copy


class ACO_CVRP:
    """Thuật toán Ant Colony Optimization cho bài toán Định tuyến Phương tiện có Giới hạn Tải trọng (CVRP)"""

    def __init__(self, cvrp, num_ants=20, alpha=1.0, beta=2.0, rho=0.5, q=100, max_iterations=100,
                 min_max_aco=False, local_search=False, elitist_ants=0, initial_pheromone=1.0):
        """
        Khởi tạo thuật toán ACO cho CVRP

        Tham số:
        cvrp -- Đối tượng CVRP
        num_ants -- Số lượng kiến
        alpha -- Tầm quan trọng của pheromone
        beta -- Tầm quan trọng của heuristic
        rho -- Tỷ lệ bay hơi pheromone
        q -- Hệ số lượng pheromone thả
        max_iterations -- Số vòng lặp tối đa
        min_max_aco -- Sử dụng biến thể MIN-MAX Ant System
        local_search -- Sử dụng tìm kiếm cục bộ
        elitist_ants -- Số lượng kiến ưu tú
        initial_pheromone -- Giá trị pheromone khởi tạo ban đầu
        """
        self.cvrp = cvrp
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_iterations = max_iterations
        self.min_max_aco = min_max_aco
        self.local_search = local_search
        self.elitist_ants = elitist_ants
        self.initial_pheromone = initial_pheromone

        # Số lượng khách hàng
        self.n = len(cvrp.customers)

        # Khởi tạo ma trận pheromone
        self.pheromone = np.ones((self.n, self.n)) * self.initial_pheromone

        # Khởi tạo giá trị pheromone tối thiểu và tối đa (cho MIN-MAX ACO)
        self.max_pheromone = 1.0
        self.min_pheromone = 0.1

        # Khởi tạo ma trận heuristic (nghịch đảo của khoảng cách)
        self.heuristic = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j and cvrp.distances[i, j] > 0:
                    self.heuristic[i, j] = 1.0 / cvrp.distances[i, j]

        # Lưu kết quả
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_iteration = 0

        # Trạng thái cho trực quan hóa
        self.current_solution = None
        self.current_cost = float('inf')
        self.current_iteration = 0

        # Lịch sử chi phí
        self.cost_history = []
        self.avg_cost_history = []
        self.worst_cost_history = []
        self.pheromone_stats_history = []

        # Cờ dừng và tạm dừng
        self.stop_flag = False
        self.paused = False
        self.pause_condition = threading.Condition()
        self.was_stopped = False

    def run(self, callback=None, step_callback=None):
        """
        Chạy thuật toán ACO

        Tham số:
        callback -- Hàm gọi lại khi hoàn thành
        step_callback -- Hàm gọi lại sau mỗi vòng lặp
        """
        self.stop_flag = False
        self.was_stopped = False
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_iteration = 0
        self.cost_history = []
        self.avg_cost_history = []
        self.worst_cost_history = []
        self.pheromone_stats_history = []
        self.time_history = []

        # Nếu sử dụng MIN-MAX ACO, khởi tạo giá trị pheromone tối đa
        if self.min_max_aco:
            initial_solution = self.construct_initial_solution()
            initial_cost = self.cvrp.calculate_solution_cost(initial_solution)
            self.max_pheromone = 1.0 / (self.rho * initial_cost)
            self.min_pheromone = self.max_pheromone * 0.001

        for iteration in range(self.max_iterations):
            # Kiểm tra dừng
            if self.stop_flag:
                self.was_stopped = True
                break

            # Kiểm tra tạm dừng
            with self.pause_condition:
                while self.paused and not self.stop_flag:
                    self.pause_condition.wait()

            # Đo thời gian tính toán thuần túy bắt đầu (không bao gồm thời gian UI)
            start_time = time.time()
            self.current_iteration = iteration

            ant_solutions = []
            ant_costs = []

            # Mỗi kiến xây dựng một giải pháp
            for ant in range(self.num_ants):
                solution = self.construct_solution()

                # Áp dụng tìm kiếm cục bộ nếu được kích hoạt
                if self.local_search:
                    solution = self.local_search_2opt(solution)

                cost = self.cvrp.calculate_solution_cost(solution)

                ant_solutions.append(solution)
                ant_costs.append(cost)

                # Cập nhật giải pháp tốt nhất
                if cost < self.best_cost:
                    self.best_solution = copy.deepcopy(solution)
                    self.best_cost = cost
                    self.best_iteration = iteration

            # Cập nhật pheromone
            self.update_pheromone(ant_solutions, ant_costs)

            # Tính toán các thống kê
            avg_cost = np.mean(ant_costs)
            worst_cost = np.max(ant_costs)

            # Thống kê pheromone
            pheromone_stats = {
                'avg': np.mean(self.pheromone),
                'max': np.max(self.pheromone),
                'min': np.min(self.pheromone),
            }

            # Lưu lịch sử và kết thúc đo thời gian tính toán thuần túy
            end_time = time.time()
            computation_time = end_time - start_time
            self.time_history.append(computation_time)
            self.cost_history.append(self.best_cost)
            self.avg_cost_history.append(avg_cost)
            self.worst_cost_history.append(worst_cost)
            self.pheromone_stats_history.append(pheromone_stats)

            # Lấy giải pháp hiện tại tốt nhất trong quần thể kiến
            current_best_idx = np.argmin(ant_costs)
            current_solution = ant_solutions[current_best_idx]
            current_cost = ant_costs[current_best_idx]
            self.current_solution = current_solution
            self.current_cost = current_cost

            # Gọi callback từng bước và truyền thời gian tính toán thuần túy
            if step_callback:
                progress = (iteration + 1) / self.max_iterations
                data = {
                    'iteration': iteration + 1,
                    'progress': progress,
                    'solution': current_solution,
                    'cost': current_cost,
                    'best_solution': self.best_solution,
                    'best_cost': self.best_cost,
                    'avg_cost': avg_cost,
                    'worst_cost': worst_cost,
                    'pheromone': self.pheromone,
                    'cost_history': self.cost_history,
                    'computation_time': computation_time,  # Thời gian tính toán thuần túy
                }
                should_stop = step_callback(data)
                if should_stop:
                    self.was_stopped = True
                    break

        # Gọi callback khi hoàn thành
        if callback and not self.was_stopped:
            callback((self.best_solution, self.best_cost))

        return self.best_solution, self.best_cost

    def construct_initial_solution(self):
        """
        Xây dựng giải pháp ban đầu bằng phương pháp savings

        Trả về:
        Danh sách các tuyến đường
        """
        # Tạo tuyến đường riêng cho từng khách hàng
        routes = [[i] for i in range(1, self.n)]

        # Tính toán các giá trị savings
        savings = []
        for i in range(1, self.n):
            for j in range(i+1, self.n):
                # Savings = dist(0,i) + dist(0,j) - dist(i,j)
                saving = (self.cvrp.distances[0, i] + self.cvrp.distances[0, j] -
                          self.cvrp.distances[i, j])
                savings.append((saving, i, j))

        # Sắp xếp theo savings giảm dần
        savings.sort(reverse=True)

        # Ánh xạ giữa khách hàng và tuyến đường
        customer_to_route = {i: idx for idx, route in enumerate(routes) for i in route}

        # Hợp nhất các tuyến đường dựa trên savings
        for saving, i, j in savings:
            # Nếu i và j đã ở cùng tuyến đường, bỏ qua
            if customer_to_route[i] == customer_to_route[j]:
                continue

            # Lấy tuyến đường chứa i và j
            route_i = routes[customer_to_route[i]]
            route_j = routes[customer_to_route[j]]

            # Kiểm tra nếu i và j là đầu/cuối của các tuyến đường
            is_i_end = (i == route_i[0] or i == route_i[-1])
            is_j_end = (j == route_j[0] or j == route_j[-1])

            # Nếu cả hai không phải đầu/cuối, bỏ qua
            if not (is_i_end and is_j_end):
                continue

            # Kiểm tra ràng buộc dung lượng
            total_demand = sum(self.cvrp.customers[k].demand for k in route_i + route_j)
            if total_demand > self.cvrp.capacity:
                continue

            # Hợp nhất hai tuyến đường
            # Xác định vị trí của i và j trong tuyến đường để hợp nhất hợp lý
            if i == route_i[0]:
                route_i.reverse()
            if j == route_j[-1]:
                route_j.reverse()

            # Hợp nhất và cập nhật ánh xạ
            new_route = route_i + route_j
            routes[customer_to_route[i]] = new_route
            routes.pop(customer_to_route[j])

            # Cập nhật ánh xạ
            for k in route_j:
                customer_to_route[k] = customer_to_route[i]

            # Cập nhật các chỉ số tuyến đường
            for k in range(1, self.n):
                for idx, route in enumerate(routes):
                    if k in route:
                        customer_to_route[k] = idx

        return routes

    def construct_solution(self):
        """
        Xây dựng một giải pháp bằng một kiến

        Trả về:
        Danh sách các tuyến đường (mỗi tuyến là một danh sách khách hàng)
        """
        solution = []
        remaining = list(range(1, self.n))  # Danh sách khách hàng chưa thăm (bỏ qua depot 0)

        while remaining:
            # Bắt đầu một tuyến mới từ depot
            route = []
            current_capacity = 0
            current_node = 0  # Depot

            while True:
                # Tìm các khách hàng tiếp theo có thể thăm
                candidates = []
                for node in remaining:
                    if current_capacity + self.cvrp.customers[node].demand <= self.cvrp.capacity:
                        candidates.append(node)

                if not candidates:
                    break  # Không thể thêm khách hàng vào tuyến hiện tại

                # Chọn khách hàng tiếp theo dựa trên quy tắc chọn của ACO
                next_node = self.select_next_node(current_node, candidates)
                route.append(next_node)
                remaining.remove(next_node)

                # Cập nhật dung lượng hiện tại
                current_capacity += self.cvrp.customers[next_node].demand
                current_node = next_node

            if route:  # Nếu tuyến không rỗng, thêm vào giải pháp
                solution.append(route)

        return solution

    def select_next_node(self, current, candidates):
        """
        Chọn khách hàng tiếp theo dựa trên pheromone và heuristic

        Tham số:
        current -- Khách hàng hiện tại
        candidates -- Danh sách các khách hàng tiếp theo có thể chọn

        Trả về:
        Khách hàng tiếp theo được chọn
        """
        if not candidates:
            return None

        # Tính xác suất cho mỗi ứng viên
        probabilities = np.zeros(len(candidates))

        for i, candidate in enumerate(candidates):
            pheromone = self.pheromone[current, candidate] ** self.alpha
            heuristic_value = self.heuristic[current, candidate] ** self.beta
            probabilities[i] = pheromone * heuristic_value

        # Chuẩn hóa xác suất
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
        else:
            probabilities = np.ones(len(candidates)) / len(candidates)

        # Chọn khách hàng tiếp theo dựa trên xác suất
        selected = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected]

    def update_pheromone(self, solutions, costs):
        """
        Cập nhật ma trận pheromone

        Tham số:
        solutions -- Danh sách các giải pháp
        costs -- Danh sách các chi phí tương ứng
        """
        # Bay hơi pheromone
        self.pheromone = (1 - self.rho) * self.pheromone

        if self.min_max_aco:
            # Trong MIN-MAX ACO, chỉ kiến tốt nhất vòng lặp hoặc tốt nhất toàn cục thả pheromone
            best_idx = np.argmin(costs)
            best_solution = solutions[best_idx]
            best_cost = costs[best_idx]

            # Thả pheromone cho giải pháp tốt nhất vòng lặp
            self.deposit_pheromone(best_solution, best_cost)

            # Thả pheromone cho giải pháp tốt nhất toàn cục nếu có
            if self.best_solution:
                self.deposit_pheromone(self.best_solution, self.best_cost, weight=2.0)

            # Giới hạn pheromone trong khoảng [min_pheromone, max_pheromone]
            self.pheromone = np.clip(self.pheromone, self.min_pheromone, self.max_pheromone)
        else:
            # Trong ACO tiêu chuẩn, tất cả kiến thả pheromone
            for solution, cost in zip(solutions, costs):
                self.deposit_pheromone(solution, cost)

            # Thả pheromone bổ sung cho giải pháp tốt nhất nếu sử dụng kiến ưu tú
            if self.elitist_ants > 0 and self.best_solution:
                self.deposit_pheromone(self.best_solution, self.best_cost, weight=self.elitist_ants)

    def deposit_pheromone(self, solution, cost, weight=1.0):
        """
        Thả pheromone trên các cạnh của một giải pháp

        Tham số:
        solution -- Giải pháp (danh sách các tuyến)
        cost -- Chi phí của giải pháp
        weight -- Hệ số nhân cho lượng pheromone (mặc định = 1.0)
        """
        delta = (self.q / cost) * weight if cost > 0 else 0

        for route in solution:
            prev_node = 0  # Bắt đầu từ depot

            for node in route:
                self.pheromone[prev_node, node] += delta
                self.pheromone[node, prev_node] += delta  # Đồ thị vô hướng
                prev_node = node

            # Quay lại depot
            self.pheromone[prev_node, 0] += delta
            self.pheromone[0, prev_node] += delta

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
        if not route or len(route) < 2:
            return route

        best_route = route.copy()
        improved = True

        while improved:
            improved = False
            best_distance = self.cvrp.calculate_route_distance(best_route)

            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    if j - i < 1:
                        continue

                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    current_distance = self.cvrp.calculate_route_distance(new_route)

                    if current_distance < best_distance:
                        best_route = new_route
                        best_distance = current_distance
                        improved = True
            
            if not improved:
                break
        
        return best_route

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