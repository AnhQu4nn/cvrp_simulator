# Giải Thích Chi Tiết `core/genetic.py` - Thuật Toán Di Truyền (Genetic Algorithm)

Tệp `core/genetic.py` triển khai Thuật toán Di truyền (GA) để giải quyết Bài toán Định tuyến Phương tiện Có Giới hạn Tải trọng (Capacitated Vehicle Routing Problem - CVRP). Dưới đây là giải thích chi tiết về các thành phần chính của lớp `GeneticAlgorithm_CVRP`.

## 1. Khởi Tạo (`__init__`)

Hàm khởi tạo `__init__` thiết lập các tham số và trạng thái ban đầu cho thuật toán.

```python
class GeneticAlgorithm_CVRP:
    def __init__(self, cvrp, population_size=50, mutation_rate=0.1, crossover_rate=0.8, elitism=5, max_generations=100,
                 selection_method="tournament", crossover_method="ordered", mutation_method="swap",
                 tournament_size=3, early_stopping=None, local_search=False):
        self.cvrp = cvrp  # Đối tượng CVRP chứa thông tin về bài toán (khách hàng, kho, sức chứa xe)
        self.population_size = population_size  # Kích thước quần thể (số lượng nhiễm sắc thể trong mỗi thế hệ)
        self.mutation_rate = mutation_rate  # Tỷ lệ đột biến (xác suất một nhiễm sắc thể bị đột biến)
        self.crossover_rate = crossover_rate  # Tỷ lệ lai ghép (xác suất hai nhiễm sắc thể cha mẹ tạo con)
        self.elitism = elitism  # Số lượng cá thể ưu tú (tốt nhất) được giữ lại cho thế hệ sau
        self.max_generations = max_generations  # Số thế hệ tối đa mà thuật toán sẽ chạy
        self.selection_method = selection_method  # Phương pháp chọn lọc cha mẹ ('tournament', 'roulette', 'rank')
        self.crossover_method = crossover_method  # Phương pháp lai ghép ('ordered', 'partially_mapped', 'cycle')
        self.mutation_method = mutation_method  # Phương pháp đột biến ('swap', 'insert', 'inversion', 'scramble')
        self.tournament_size = tournament_size  # Kích thước của giải đấu (nếu dùng 'tournament selection')
        self.early_stopping = early_stopping  # Số thế hệ không cải thiện để dừng sớm (nếu có)
        self.local_search = local_search # Cờ bật/tắt tìm kiếm cục bộ 2-opt

        self.n = len(cvrp.customers)  # Số lượng khách hàng (bao gồm cả kho)

        # Lưu trữ giải pháp tốt nhất tìm được
        self.best_solution = None
        self.best_cost = float('inf')

        # Trạng thái hiện tại cho mục đích trực quan hóa hoặc callback
        self.current_solution = None
        self.current_cost = float('inf')

        # Lịch sử các chỉ số qua các thế hệ
        self.cost_history = []  # Lịch sử chi phí của giải pháp tốt nhất
        self.time_history = []  # Lịch sử thời gian tính toán mỗi thế hệ
        self.avg_cost_history = []  # Lịch sử chi phí trung bình của quần thể
        self.worst_cost_history = []  # Lịch sử chi phí của giải pháp tệ nhất
        self.diversity_history = [] # Lịch sử độ đa dạng của quần thể

        # Cờ điều khiển và trạng thái dừng/tạm dừng
        self.stop_flag = False  # Cờ để dừng thuật toán
        self.paused = False  # Cờ để tạm dừng thuật toán
        self.pause_condition = threading.Condition()  # Điều kiện đồng bộ hóa cho tạm dừng
        self.stagnation_count = 0  # Đếm số thế hệ không có cải thiện (cho dừng sớm)
        self.was_stopped = False # Cờ cho biết thuật toán có bị dừng bởi người dùng không
```

**Giải thích các tham số:**

*   `cvrp`: Một đối tượng chứa thông tin về bài toán CVRP, bao gồm danh sách khách hàng (vị trí, nhu cầu) và thông tin về kho (depot), sức chứa của xe.
*   `population_size`: Số lượng giải pháp (nhiễm sắc thể) được duy trì trong mỗi thế hệ. Quần thể lớn hơn có thể khám phá không gian tìm kiếm rộng hơn nhưng tốn nhiều tài nguyên hơn.
*   `mutation_rate`: Xác suất một gen (khách hàng) trong một nhiễm sắc thể bị thay đổi. Đột biến giúp duy trì sự đa dạng và tránh bị mắc kẹt ở các điểm cực trị cục bộ.
*   `crossover_rate`: Xác suất hai nhiễm sắc thể cha mẹ được chọn sẽ thực hiện lai ghép để tạo ra con cái. Lai ghép kết hợp các đặc điểm tốt từ cha mẹ.
*   `elitism`: Số lượng nhiễm sắc thể có độ thích nghi cao nhất từ thế hệ hiện tại được chuyển trực tiếp sang thế hệ tiếp theo mà không qua lai ghép hay đột biến. Điều này đảm bảo rằng giải pháp tốt nhất không bị mất.
*   `max_generations`: Số lượng vòng lặp (thế hệ) tối đa mà thuật toán sẽ chạy.
*   `selection_method`: Cách thức chọn lọc cha mẹ cho thế hệ tiếp theo.
    *   `tournament`: Chọn ngẫu nhiên `tournament_size` cá thể và chọn cá thể tốt nhất trong số đó.
    *   `roulette`: Chọn cá thể dựa trên xác suất tỷ lệ thuận với độ thích nghi của chúng (giống như vòng quay roulette).
    *   `rank`: Chọn cá thể dựa trên thứ hạng độ thích nghi của chúng, thay vì giá trị độ thích nghi tuyệt đối.
*   `crossover_method`: Cách thức kết hợp thông tin di truyền từ hai cha mẹ để tạo ra con cái.
    *   `ordered` (OX1): Chọn một đoạn gen từ cha mẹ 1, các gen còn lại được lấy từ cha mẹ 2 theo thứ tự xuất hiện.
    *   `partially_mapped` (PMX): Trao đổi một đoạn gen giữa hai cha mẹ và giải quyết xung đột bằng cách sử dụng ánh xạ.
    *   `cycle` (CX): Đảm bảo mỗi gen của con cái đến từ một trong hai cha mẹ tại cùng một vị trí.
*   `mutation_method`: Cách thức thay đổi ngẫu nhiên một hoặc nhiều gen trong một nhiễm sắc thể.
    *   `swap`: Hoán đổi vị trí của hai gen ngẫu nhiên.
    *   `insert`: Di chuyển một gen đến một vị trí ngẫu nhiên khác.
    *   `inversion`: Đảo ngược một đoạn gen ngẫu nhiên.
    *   `scramble`: Xáo trộn một đoạn gen ngẫu nhiên.
*   `tournament_size`: Nếu `selection_method` là 'tournament', tham số này xác định số lượng cá thể tham gia vào mỗi "giải đấu".
*   `early_stopping`: Nếu được đặt, thuật toán sẽ dừng nếu không có cải thiện nào trong `early_stopping` thế hệ liên tiếp.
*   `local_search`: Cờ boolean cho biết có nên áp dụng thuật toán tìm kiếm cục bộ (ví dụ: 2-opt) để cải thiện các giải pháp sau mỗi thế hệ hay không.

## 2. Chạy Thuật Toán (`run`)

Phương thức `run` là vòng lặp chính của thuật toán di truyền.

```python
    def run(self, callback=None, step_callback=None):
        # ... (khởi tạo lại các biến trạng thái và lịch sử) ...

        population = self.initialize_population() # Khởi tạo quần thể ban đầu

        for generation in range(self.max_generations):
            if self.stop_flag: # Kiểm tra cờ dừng từ bên ngoài
                self.was_stopped = True
                break

            with self.pause_condition: # Xử lý tạm dừng
                while self.paused and not self.stop_flag:
                    self.pause_condition.wait()

            start_time = time.time() # Bắt đầu đo thời gian tính toán

            fitness_values = [self.evaluate_fitness(individual) for individual in population] # Đánh giá độ thích nghi của từng cá thể
            best_idx = np.argmin(fitness_values) # Tìm cá thể tốt nhất hiện tại
            current_best_solution = self.decode_chromosome_with_feasibility_check(population[best_idx])
            current_best_cost = fitness_values[best_idx]
            
            # Kiểm tra và sửa chữa giải pháp tốt nhất nếu cần
            if not self.cvrp.is_solution_valid(current_best_solution) or not self.check_solution_feasibility(current_best_solution):
                current_best_solution = self.repair_solution(current_best_solution)
                if not self.check_solution_feasibility(current_best_solution):
                    new_solution = []
                    for route in current_best_solution:
                        for customer in route:
                            new_solution.append([customer])
                    current_best_solution = new_solution
                current_best_cost = self.cvrp.calculate_solution_cost(current_best_solution)


            if self.local_search and current_best_solution: # Áp dụng tìm kiếm cục bộ 2-opt nếu được kích hoạt
                improved_solution = self.local_search_2opt(current_best_solution)
                improved_cost = self.cvrp.calculate_solution_cost(improved_solution)
                if improved_cost < current_best_cost:
                    current_best_solution = improved_solution
                    current_best_cost = improved_cost

            # Cập nhật giải pháp toàn cục tốt nhất
            if current_best_cost < self.best_cost:
                self.best_solution = current_best_solution
                self.best_cost = current_best_cost
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # ... (tính toán avg_cost, worst_cost, diversity) ...
            # ... (lưu lịch sử cost_history, avg_cost_history, worst_cost_history, diversity_history) ...

            end_time = time.time()
            computation_time = end_time - start_time # Thời gian tính toán thuần túy của thế hệ
            self.time_history.append(computation_time)

            if step_callback: # Gọi hàm callback sau mỗi thế hệ (nếu có)
                # ... (chuẩn bị dữ liệu cho step_callback) ...
                step_callback(step_data)

            if self.early_stopping and self.stagnation_count >= self.early_stopping: # Kiểm tra điều kiện dừng sớm
                break

            new_population = [] # Quần thể mới

            # Elitism: Giữ lại các cá thể tốt nhất
            sorted_indices = np.argsort(fitness_values)
            for i in range(self.elitism):
                new_population.append(population[sorted_indices[i]])

            # Tạo các cá thể mới cho đến khi đủ kích thước quần thể
            while len(new_population) < self.population_size:
                # Chọn lọc cha mẹ
                parent1 = self.select_parent(population, fitness_values) # Sử dụng phương thức chọn lọc đã cấu hình
                parent2 = self.select_parent(population, fitness_values)

                # Lai ghép
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2) # Sử dụng phương thức lai ghép đã cấu hình
                else:
                    child1, child2 = parent1.copy(), parent2.copy() # Không lai ghép, giữ nguyên cha mẹ

                # Kiểm tra và sửa chữa nhiễm sắc thể sau khi lai ghép (đảm bảo không có gen trùng lặp hoặc thiếu)
                child1 = self.check_and_repair_chromosomes(child1)
                child2 = self.check_and_repair_chromosomes(child2)

                # Kiểm tra tính khả thi của giải pháp sau khi lai ghép (ví dụ: sức chứa xe)
                # và sửa chữa nếu cần
                solution1 = self.decode_chromosome_with_feasibility_check(child1)
                solution2 = self.decode_chromosome_with_feasibility_check(child2)


                # Đột biến
                if random.random() < self.mutation_rate:
                    self.mutate(child1) # Sử dụng phương thức đột biến đã cấu hình
                if random.random() < self.mutation_rate:
                    self.mutate(child2)
                
                # Kiểm tra và sửa chữa nhiễm sắc thể sau khi đột biến
                child1 = self.check_and_repair_chromosomes(child1)
                child2 = self.check_and_repair_chromosomes(child2)

                # Thêm con cái vào quần thể mới (nếu còn chỗ)
                if len(new_population) < self.population_size:
                    new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population # Cập nhật quần thể cho thế hệ tiếp theo

        if callback: # Gọi hàm callback khi thuật toán hoàn thành (nếu có)
            # ... (chuẩn bị dữ liệu cho callback) ...
            callback(result_data)
        
        return self.best_solution, self.best_cost
```

**Quy trình chính trong mỗi thế hệ:**

1.  **Đánh giá (Evaluation):** Tính toán độ thích nghi (thường là tổng chi phí quãng đường) cho mỗi nhiễm sắc thể trong quần thể hiện tại.
2.  **Cập nhật tốt nhất:** So sánh giải pháp tốt nhất của thế hệ hiện tại với giải pháp tốt nhất toàn cục đã tìm được và cập nhật nếu tốt hơn.
3.  **Tìm kiếm cục bộ (Tùy chọn):** Nếu `local_search` được bật, áp dụng thuật toán tìm kiếm cục bộ (ví dụ: 2-opt) để cố gắng cải thiện giải pháp tốt nhất hiện tại.
4.  **Ghi lại lịch sử:** Lưu lại các thông số như chi phí tốt nhất, trung bình, tệ nhất, độ đa dạng và thời gian tính toán.
5.  **Callback (Tùy chọn):** Gọi hàm `step_callback` để cung cấp thông tin về tiến trình của thuật toán, thường dùng để cập nhật giao diện người dùng.
6.  **Kiểm tra dừng sớm:** Nếu không có cải thiện sau một số thế hệ nhất định (`early_stopping`), thuật toán có thể dừng lại.
7.  **Tạo quần thể mới:**
    *   **Elitism:** Giữ lại một số cá thể tốt nhất từ quần thể hiện tại.
    *   **Chọn lọc (Selection):** Chọn các cặp cha mẹ từ quần thể hiện tại dựa trên phương pháp chọn lọc đã định (tournament, roulette, rank).
    *   **Lai ghép (Crossover):** Với một xác suất (`crossover_rate`), các cặp cha mẹ sẽ tạo ra con cái bằng cách sử dụng phương pháp lai ghép đã định. Nếu không lai ghép, con cái sẽ là bản sao của cha mẹ.
    *   **Sửa chữa nhiễm sắc thể sau lai ghép:** Đảm bảo nhiễm sắc thể con hợp lệ (ví dụ: không có khách hàng bị lặp lại hoặc bỏ sót).
    *   **Kiểm tra và sửa chữa tính khả thi sau lai ghép:** Đảm bảo giải pháp được giải mã từ nhiễm sắc thể con là khả thi (ví dụ: không vi phạm sức chứa xe).
    *   **Đột biến (Mutation):** Với một xác suất (`mutation_rate`), các nhiễm sắc thể con sẽ bị đột biến bằng cách sử dụng phương pháp đột biến đã định.
    *   **Sửa chữa nhiễm sắc thể sau đột biến:** Tương tự như sau lai ghép.
    *   Thêm các con cái đã qua lai ghép và đột biến vào quần thể mới.
8.  **Thay thế quần thể:** Quần thể mới thay thế quần thể cũ cho thế hệ tiếp theo.

Vòng lặp này tiếp tục cho đến khi đạt `max_generations` hoặc điều kiện dừng sớm được thỏa mãn.

## 3. Khởi Tạo Quần Thể (`initialize_population`)

```python
    def initialize_population(self):
        population = []
        customers = list(range(1, self.n)) # Danh sách ID khách hàng (bỏ qua kho ID 0)
        for _ in range(self.population_size):
            chromosome = random.sample(customers, len(customers)) # Tạo một hoán vị ngẫu nhiên của khách hàng
            population.append(chromosome)
        return population
```

*   Tạo ra `population_size` nhiễm sắc thể.
*   Mỗi nhiễm sắc thể là một **hoán vị** của các ID khách hàng (không bao gồm kho). Thứ tự các khách hàng trong nhiễm sắc thể đại diện cho một trình tự tiềm năng mà xe sẽ đi qua. Việc giải mã nhiễm sắc thể này thành các tuyến đường cụ thể (có tính đến sức chứa xe) được thực hiện trong các hàm khác.

## 4. Giải Mã Nhiễm Sắc Thể

Có hai phương thức chính để giải mã một nhiễm sắc thể (chuỗi các ID khách hàng) thành một tập hợp các tuyến đường khả thi:

### a. `decode_chromosome` (Giải mã cơ bản)

```python
    def decode_chromosome(self, chromosome):
        solution = []
        current_route = []
        current_capacity = self.cvrp.capacity

        for customer_id in chromosome:
            customer = self.cvrp.customers[customer_id]
            if current_capacity >= customer.demand: # Nếu xe còn đủ sức chứa
                current_route.append(customer_id)
                current_capacity -= customer.demand
            else: # Nếu xe không đủ sức chứa
                if current_route: # Hoàn thành tuyến đường hiện tại
                    solution.append(current_route)
                current_route = [customer_id] # Bắt đầu tuyến đường mới với khách hàng này
                current_capacity = self.cvrp.capacity - customer.demand
        
        if current_route: # Thêm tuyến đường cuối cùng (nếu có)
            solution.append(current_route)
        
        return solution
```

*   Phương thức này duyệt qua các khách hàng trong nhiễm sắc thể theo thứ tự.
*   Nó cố gắng thêm từng khách hàng vào tuyến đường hiện tại của xe.
*   Nếu việc thêm một khách hàng không làm vượt quá sức chứa của xe, khách hàng đó được thêm vào tuyến đường hiện tại.
*   Nếu vượt quá sức chứa, tuyến đường hiện tại được hoàn thành (thêm vào `solution`), và một tuyến đường mới được bắt đầu với khách hàng hiện tại.
*   Đây là một chiến lược tham lam đơn giản để tạo tuyến đường.

### b. `decode_chromosome_with_feasibility_check` (Giải mã có kiểm tra và sửa chữa)

```python
    def decode_chromosome_with_feasibility_check(self, chromosome):
        # ... (Tương tự như decode_chromosome ban đầu) ...
        # Sau khi tạo solution ban đầu:
        solution = self.check_and_repair_capacity(solution) # Kiểm tra và sửa chữa sức chứa cho toàn bộ giải pháp
        
        final_solution = []
        for route in solution:
            if self.check_route_feasibility(route): # Kiểm tra tính khả thi của từng tuyến (ví dụ: không rỗng)
                final_solution.append(route)
            else: 
                # Xử lý tuyến không khả thi (ví dụ: chia nhỏ nếu có nhiều khách hàng, hoặc bỏ qua nếu rỗng)
                # Mã hiện tại có thể cần cải thiện logic này
                repaired_route = self.check_and_repair_capacity_for_route(route)
                if repaired_route and self.check_route_feasibility(repaired_route):
                     final_solution.append(repaired_route)
                # Nếu vẫn không khả thi, có thể cần phải phân tách các khách hàng thành các tuyến đơn lẻ
                # hoặc một cơ chế sửa chữa mạnh mẽ hơn.
                # Ví dụ đơn giản: nếu tuyến không rỗng, vẫn thêm vào
                elif route: # Thêm tạm để tránh mất khách hàng, cần cải thiện
                    final_solution.append(route)


        # Đảm bảo tất cả khách hàng trong nhiễm sắc thể đều có mặt trong giải pháp
        all_customers_in_chromosome = set(chromosome)
        all_customers_in_solution = set()
        for route in final_solution:
            for cust_id in route:
                all_customers_in_solution.add(cust_id)

        missing_customers = list(all_customers_in_chromosome - all_customers_in_solution)
        
        # Thêm các khách hàng bị thiếu vào các tuyến mới (mỗi khách hàng một tuyến)
        # Đây là một cách xử lý đơn giản, có thể tối ưu hơn
        for cust_id in missing_customers:
            final_solution.append([cust_id])
            
        # Sửa chữa lại toàn bộ giải pháp sau khi thêm khách hàng bị thiếu
        final_solution = self.repair_solution(final_solution)

        return final_solution
```

*   Phiên bản này phức tạp hơn, nó không chỉ giải mã mà còn cố gắng **kiểm tra và sửa chữa** các vấn đề về tính khả thi, đặc biệt là về sức chứa.
*   Gọi `check_and_repair_capacity` để điều chỉnh các tuyến đường nhằm đảm bảo không vi phạm sức chứa.
*   Gọi `check_route_feasibility` để kiểm tra từng tuyến đơn lẻ.
*   Có logic để xử lý các khách hàng bị "thiếu" (không được gán vào tuyến nào sau khi giải mã ban đầu) bằng cách tạo các tuyến mới cho chúng.
*   Cuối cùng, gọi `repair_solution` một lần nữa để đảm bảo giải pháp tổng thể là tốt nhất có thể sau các bước sửa chữa.

## 5. Kiểm Tra và Sửa Chữa (Feasibility Checks and Repairs)

Có nhiều phương thức liên quan đến việc kiểm tra và sửa chữa các giải pháp và nhiễm sắc thể:

### a. `check_and_repair_capacity(solution)`

```python
    def check_and_repair_capacity(self, solution):
        repaired_solution = []
        current_vehicle_customers = [] # Các khách hàng chưa được phục vụ, sẽ được thử gán lại

        for route_idx, route in enumerate(solution):
            current_route_repaired = []
            current_capacity = self.cvrp.capacity
            
            # Sắp xếp khách hàng trong tuyến theo một tiêu chí nào đó (ví dụ: nhu cầu, khoảng cách - hiện tại chưa có)
            # Hoặc đơn giản là duyệt theo thứ tự hiện có
            
            for customer_id in route:
                customer = self.cvrp.customers[customer_id]
                if current_capacity >= customer.demand:
                    current_route_repaired.append(customer_id)
                    current_capacity -= customer.demand
                else:
                    # Khách hàng này không vừa với tuyến hiện tại (sau khi đã cố gắng lấp đầy)
                    # Đưa vào danh sách chờ để thử gán vào các tuyến sau hoặc tuyến mới
                    current_vehicle_customers.append(customer_id) 
            
            if current_route_repaired: # Nếu tuyến này có khách hàng sau khi sửa
                 repaired_solution.append(current_route_repaired)

        # Xử lý các khách hàng chưa được phục vụ (current_vehicle_customers)
        # Cố gắng thêm chúng vào các tuyến hiện có hoặc tạo tuyến mới
        temp_route = []
        temp_capacity = self.cvrp.capacity
        remaining_customers_to_assign = list(current_vehicle_customers) # Tạo bản sao để duyệt
        
        processed_customers = set() # Để tránh xử lý lại khách hàng đã gán

        while remaining_customers_to_assign:
            customer_assigned_in_this_pass = False
            customers_for_next_pass = []

            for cust_id in remaining_customers_to_assign:
                if cust_id in processed_customers:
                    continue

                customer = self.cvrp.customers[cust_id]
                
                # Cố gắng thêm vào một tuyến hiện có trong repaired_solution mà còn chỗ
                added_to_existing = False
                for r_idx, r_route in enumerate(repaired_solution):
                    route_cap = self.cvrp.capacity
                    for c_in_r in r_route:
                        route_cap -= self.cvrp.customers[c_in_r].demand
                    
                    if route_cap >= customer.demand:
                        repaired_solution[r_idx].append(cust_id)
                        processed_customers.add(cust_id)
                        added_to_existing = True
                        customer_assigned_in_this_pass = True
                        break 
                
                if added_to_existing:
                    continue

                # Nếu không thêm được vào tuyến hiện có, thử thêm vào tuyến tạm thời đang xây dựng
                if temp_capacity >= customer.demand:
                    temp_route.append(cust_id)
                    temp_capacity -= customer.demand
                    processed_customers.add(cust_id)
                    customer_assigned_in_this_pass = True
                else:
                    # Không vừa với tuyến tạm, lưu lại cho lượt sau hoặc tuyến mới hơn
                    customers_for_next_pass.append(cust_id)
            
            if temp_route: # Nếu tuyến tạm có khách hàng
                repaired_solution.append(list(temp_route)) # Thêm bản sao
                temp_route = [] # Bắt đầu tuyến tạm mới
                temp_capacity = self.cvrp.capacity

            remaining_customers_to_assign = customers_for_next_pass # Cập nhật danh sách cho lượt sau

            if not customer_assigned_in_this_pass and remaining_customers_to_assign:
                # Nếu không gán được khách hàng nào trong lượt này mà vẫn còn khách hàng
                # thì tạo một tuyến mới cho khách hàng đầu tiên còn lại để tránh vòng lặp vô hạn
                if remaining_customers_to_assign:
                    first_unassigned = remaining_customers_to_assign.pop(0)
                    repaired_solution.append([first_unassigned])
                    processed_customers.add(first_unassigned)


        # Loại bỏ các tuyến rỗng có thể đã được tạo ra
        final_repaired_solution = [route for route in repaired_solution if route]
        return final_repaired_solution
```

*   Mục tiêu: Duyệt qua một `solution` (danh sách các tuyến đường) và cố gắng sửa chữa các vấn đề về sức chứa.
*   Nó duyệt qua từng tuyến, cố gắng lấp đầy xe. Nếu một khách hàng không vừa, khách hàng đó được đưa vào danh sách `current_vehicle_customers` để được xử lý sau.
*   Sau đó, nó cố gắng gán các khách hàng trong `current_vehicle_customers` vào các tuyến hiện có (nếu còn chỗ) hoặc tạo các tuyến mới cho chúng.
*   Logic này khá phức tạp và có thể có nhiều cách tiếp cận khác nhau để "đóng gói" khách hàng vào các xe một cách tối ưu.

### b. `check_and_repair_chromosomes(chromosome)`

```python
    def check_and_repair_chromosomes(self, chromosome):
        # Đảm bảo tất cả khách hàng (1 đến n-1) đều xuất hiện đúng một lần
        all_customers = set(range(1, self.n))
        chromosome_set = set(chromosome)
        
        # Khách hàng bị thiếu
        missing_customers = list(all_customers - chromosome_set)
        
        # Khách hàng bị trùng lặp và vị trí của chúng
        counts = {}
        duplicates = []
        for i, cust_id in enumerate(chromosome):
            counts[cust_id] = counts.get(cust_id, 0) + 1
            if counts[cust_id] > 1:
                duplicates.append((cust_id, i)) # Lưu (id, vị trí)

        repaired_chromosome = list(chromosome)

        # Thay thế các bản sao dư thừa bằng các khách hàng bị thiếu
        # Ưu tiên giữ lại lần xuất hiện đầu tiên của khách hàng trùng lặp
        
        # Xác định vị trí cần thay thế (các lần xuất hiện sau của gen trùng lặp)
        indices_to_replace = []
        seen_once = set()
        for i, cust_id in enumerate(repaired_chromosome):
            if cust_id in seen_once: # Đây là bản sao
                indices_to_replace.append(i)
            else:
                seen_once.add(cust_id)
        
        # Thực hiện thay thế
        for i in range(min(len(indices_to_replace), len(missing_customers))):
            repaired_chromosome[indices_to_replace[i]] = missing_customers[i]
            
        # Nếu sau khi thay thế vẫn còn thiếu khách hàng (do số lượng trùng lặp ít hơn số thiếu)
        # hoặc vẫn còn gen trùng lặp (do số thiếu ít hơn số trùng lặp), cần xử lý thêm.
        # Trường hợp đơn giản: nếu số lượng gen trong chromosome không đúng, tạo lại từ đầu (cần cải thiện)
        final_check_set = set(repaired_chromosome)
        if len(repaired_chromosome) != (self.n - 1) or len(final_check_set) != (self.n - 1) :
            # Nếu có vấn đề nghiêm trọng, tạo một hoán vị ngẫu nhiên mới
            # Đây là một giải pháp cuối cùng, có thể không tối ưu.
            # Lý tưởng hơn là sửa chữa một cách thông minh hơn.
            temp_customers = list(range(1, self.n))
            random.shuffle(temp_customers)
            return temp_customers


        # Đảm bảo không có khách hàng nào ngoài phạm vi (ví dụ: số 0 hoặc lớn hơn n-1)
        # Điều này thường không xảy ra nếu khởi tạo và các toán tử được viết đúng
        # nhưng kiểm tra thêm để đảm bảo
        valid_customer_range = set(range(1, self.n))
        for cust_id in repaired_chromosome:
            if cust_id not in valid_customer_range:
                # Xử lý lỗi: thay thế bằng một khách hàng hợp lệ bị thiếu, hoặc báo lỗi
                # Hiện tại, nếu gặp phải tình huống này, có thể dẫn đến lỗi sau đó.
                # Một cách xử lý đơn giản là thay bằng một khách hàng bị thiếu nếu có.
                current_chromosome_set = set(repaired_chromosome)
                still_missing = list(valid_customer_range - current_chromosome_set)
                if still_missing:
                    # Tìm vị trí của gen không hợp lệ và thay thế
                    for idx_cr, val_cr in enumerate(repaired_chromosome):
                        if val_cr == cust_id: # Gen không hợp lệ
                             repaired_chromosome[idx_cr] = still_missing.pop(0)
                             if not still_missing: break # Hết khách hàng thiếu để thay
                # Nếu không còn khách hàng thiếu để thay thế, nhiễm sắc thể này có vấn đề.

        return repaired_chromosome
```

*   Mục tiêu: Đảm bảo một `chromosome` (danh sách ID khách hàng) là một hoán vị hợp lệ của tất cả các khách hàng (mỗi khách hàng xuất hiện đúng một lần).
*   Tìm các khách hàng bị thiếu và các khách hàng bị trùng lặp.
*   Thay thế các bản sao dư thừa của khách hàng trùng lặp bằng các khách hàng bị thiếu.
*   Nếu sau các bước sửa chữa mà nhiễm sắc thể vẫn không hợp lệ (ví dụ: sai độ dài, vẫn còn trùng lặp hoặc thiếu), nó có thể tạo một hoán vị ngẫu nhiên mới như một giải pháp cuối cùng (điều này có thể cần cải thiện).

### c. `repair_solution(solution)`

```python
    def repair_solution(self, solution):
        # Loại bỏ các tuyến rỗng
        non_empty_routes = [route for route in solution if route]
        
        # Thu thập tất cả các khách hàng đã được phục vụ
        served_customers = set()
        for route in non_empty_routes:
            for customer_id in route:
                served_customers.add(customer_id)
        
        # Xác định các khách hàng chưa được phục vụ
        all_customers_ids = set(range(1, self.n))
        unserved_customers = list(all_customers_ids - served_customers)
        
        repaired_solution = [list(route) for route in non_empty_routes] # Tạo bản sao để sửa đổi

        # Cố gắng thêm các khách hàng chưa được phục vụ vào các tuyến hiện có
        # nếu còn sức chứa
        customers_still_unserved = []
        for cust_id_unserved in unserved_customers:
            customer_obj = self.cvrp.customers[cust_id_unserved]
            added = False
            for route_idx, route in enumerate(repaired_solution):
                current_route_demand = sum(self.cvrp.customers[c].demand for c in route)
                if self.cvrp.capacity - current_route_demand >= customer_obj.demand:
                    repaired_solution[route_idx].append(cust_id_unserved)
                    added = True
                    break
            if not added:
                customers_still_unserved.append(cust_id_unserved)
        
        # Với những khách hàng vẫn chưa được phục vụ, tạo các tuyến mới cho chúng
        # (mỗi khách hàng một tuyến hoặc cố gắng nhóm chúng lại)
        
        # Cách đơn giản: mỗi khách hàng còn lại một tuyến mới
        # for cust_id_remaining in customers_still_unserved:
        #     repaired_solution.append([cust_id_remaining])

        # Cách phức tạp hơn: cố gắng nhóm các khách hàng còn lại vào các tuyến mới
        current_new_route = []
        current_new_route_capacity = self.cvrp.capacity
        for cust_id_remaining in customers_still_unserved:
            customer_obj = self.cvrp.customers[cust_id_remaining]
            if current_new_route_capacity >= customer_obj.demand:
                current_new_route.append(cust_id_remaining)
                current_new_route_capacity -= customer_obj.demand
            else:
                if current_new_route: # Hoàn thành tuyến mới hiện tại
                    repaired_solution.append(list(current_new_route))
                current_new_route = [cust_id_remaining] # Bắt đầu tuyến mới khác
                current_new_route_capacity = self.cvrp.capacity - customer_obj.demand
        
        if current_new_route: # Thêm tuyến mới cuối cùng (nếu có)
            repaired_solution.append(list(current_new_route))

        # Kiểm tra lại sức chứa của tất cả các tuyến sau khi sửa chữa
        final_solution_after_repair = []
        customers_to_reassign_again = []

        for route in repaired_solution:
            if not route: continue # Bỏ qua tuyến rỗng

            route_demand = sum(self.cvrp.customers[c].demand for c in route)
            if route_demand <= self.cvrp.capacity:
                final_solution_after_repair.append(list(route))
            else:
                # Tuyến này vẫn quá tải, cần phải chia nhỏ hoặc xử lý lại
                # Tạm thời, các khách hàng trong tuyến này sẽ được đưa vào danh sách chờ gán lại
                # Đây là một điểm cần cải thiện logic, ví dụ chia tuyến một cách thông minh hơn
                # thay vì chỉ đưa tất cả vào danh sách chờ.
                #
                # Logic chia nhỏ:
                temp_split_route = []
                temp_split_capacity = self.cvrp.capacity
                for cust_in_overloaded_route in route:
                    cust_obj_ol = self.cvrp.customers[cust_in_overloaded_route]
                    if temp_split_capacity >= cust_obj_ol.demand:
                        temp_split_route.append(cust_in_overloaded_route)
                        temp_split_capacity -= cust_obj_ol.demand
                    else:
                        if temp_split_route:
                            final_solution_after_repair.append(list(temp_split_route))
                        temp_split_route = [cust_in_overloaded_route]
                        temp_split_capacity = self.cvrp.capacity - cust_obj_ol.demand
                if temp_split_route:
                     final_solution_after_repair.append(list(temp_split_route))
        
        # Xử lý lại những khách hàng có thể đã bị tách ra (nếu có) - vòng lặp này có thể không cần thiết
        # nếu logic chia nhỏ ở trên đã đủ tốt.
        # Tuy nhiên, để đảm bảo, có thể thêm một lượt kiểm tra và gán lại.
        # (Phần này có thể được đơn giản hóa hoặc tích hợp vào vòng lặp `customers_still_unserved` ở trên)

        # Đảm bảo không có khách hàng bị trùng lặp giữa các tuyến
        # và tất cả khách hàng đều được phục vụ
        # (Logic này cần được làm cẩn thận để tránh vòng lặp vô hạn hoặc mất khách hàng)
        
        # Bước cuối: đảm bảo tất cả các khách hàng đều có mặt
        final_served_cust = set()
        for r in final_solution_after_repair:
            for c_in_r in r:
                final_served_cust.add(c_in_r)
        
        truly_missing = list(all_customers_ids - final_served_cust)
        for missing_c in truly_missing:
            # Tìm tuyến có thể thêm vào hoặc tạo tuyến mới
            added_finally = False
            for r_idx_final, r_final in enumerate(final_solution_after_repair):
                cap_final = self.cvrp.capacity - sum(self.cvrp.customers[c_f].demand for c_f in r_final)
                if cap_final >= self.cvrp.customers[missing_c].demand:
                    final_solution_after_repair[r_idx_final].append(missing_c)
                    added_finally = True
                    break
            if not added_finally:
                final_solution_after_repair.append([missing_c])


        return [route for route in final_solution_after_repair if route] # Loại bỏ tuyến rỗng cuối cùng
```

*   Mục tiêu: Nhận một `solution` và cố gắng làm cho nó khả thi nhất có thể, chủ yếu bằng cách đảm bảo tất cả khách hàng được phục vụ và không có tuyến nào vi phạm sức chứa.
*   Loại bỏ các tuyến rỗng.
*   Tìm các khách hàng chưa được phục vụ và cố gắng thêm họ vào các tuyến hiện có hoặc tạo tuyến mới.
*   Kiểm tra lại sức chứa của các tuyến và có thể chia nhỏ các tuyến quá tải.
*   Đây là một trong những phần phức tạp nhất vì có nhiều cách để "sửa chữa" một giải pháp không khả thi, và việc tìm ra cách sửa chữa tốt nhất (không làm tăng chi phí quá nhiều) là một thách thức.

### d. `check_solution_feasibility(solution)` và `check_route_feasibility(route)`

```python
    def check_solution_feasibility(self, solution):
        if not solution: return False # Giải pháp rỗng không khả thi (trừ khi không có khách hàng)
        if self.n > 1 and not solution: return False # Có khách hàng nhưng giải pháp rỗng

        all_served_customers = set()
        for route in solution:
            if not self.check_route_feasibility(route):
                return False # Nếu một tuyến không khả thi, toàn bộ giải pháp không khả thi
            for customer_id in route:
                if customer_id in all_served_customers:
                    return False # Khách hàng bị lặp lại giữa các tuyến
                all_served_customers.add(customer_id)
        
        # Kiểm tra xem tất cả khách hàng (trừ kho) có được phục vụ không
        required_customers = set(range(1, self.n))
        if all_served_customers != required_customers:
            return False 
            
        return True

    def check_route_feasibility(self, route):
        if not route: # Tuyến rỗng được coi là khả thi (không vi phạm gì) nhưng thường không mong muốn
            return True # Hoặc False nếu yêu cầu tuyến phải có khách hàng
        
        current_capacity = self.cvrp.capacity
        for customer_id in route:
            if not (0 < customer_id < self.n): # ID khách hàng không hợp lệ
                return False
            customer = self.cvrp.customers[customer_id]
            current_capacity -= customer.demand
            if current_capacity < 0: # Vượt quá sức chứa
                return False
        return True
```

*   `check_solution_feasibility`: Kiểm tra xem toàn bộ giải pháp có khả thi không.
    *   Mỗi tuyến trong giải pháp phải khả thi.
    *   Mỗi khách hàng phải được phục vụ đúng một lần.
    *   Tất cả khách hàng phải được phục vụ.
*   `check_route_feasibility`: Kiểm tra xem một tuyến đường đơn lẻ có khả thi không.
    *   ID khách hàng phải hợp lệ.
    *   Tổng nhu cầu của khách hàng trên tuyến không được vượt quá sức chứa của xe.

## 6. Đánh Giá Độ Thích Nghi (`evaluate_fitness`)

```python
    def evaluate_fitness(self, chromosome):
        # Giải mã nhiễm sắc thể thành một giải pháp (tập hợp các tuyến đường)
        # Sử dụng phiên bản có kiểm tra và sửa chữa để đảm bảo giải pháp gần với khả thi nhất
        solution = self.decode_chromosome_with_feasibility_check(chromosome)

        # Nếu giải pháp sau khi giải mã và sửa chữa không khả thi hoàn toàn,
        # có thể áp dụng một hình phạt (penalty) vào chi phí.
        # Hiện tại, chúng ta dựa vào việc repair_solution sẽ cố gắng làm cho nó khả thi.
        
        cost = self.cvrp.calculate_solution_cost(solution) # Tính tổng chi phí của giải pháp

        # Xử lý trường hợp giải pháp không khả thi (ví dụ: nếu repair không thành công tuyệt đối)
        # Bằng cách thêm một hình phạt lớn.
        # Điều này khuyến khích thuật toán hướng tới các giải pháp khả thi.
        if not self.check_solution_feasibility(solution):
             # Hình phạt có thể dựa trên mức độ không khả thi (ví dụ: số khách hàng chưa được phục vụ,
             # lượng vượt quá sức chứa, v.v.)
             # Hình phạt đơn giản:
             penalty_value = float('inf') / 2 # Một giá trị rất lớn nhưng không phải vô cùng tuyệt đối
                                            # để vẫn có thể so sánh giữa các giải pháp không khả thi.
                                            # Hoặc có thể là một hằng số lớn: 1_000_000_000
             cost += penalty_value


        # Nếu muốn, có thể thêm các yếu tố khác vào hàm fitness, ví dụ: số lượng xe sử dụng.
        # cost += self.alpha * number_of_vehicles_used (với alpha là một trọng số)

        return cost
```

*   Hàm này xác định "độ tốt" của một nhiễm sắc thể.
*   Đầu tiên, nó giải mã nhiễm sắc thể thành một giải pháp (tập hợp các tuyến đường) bằng cách sử dụng `decode_chromosome_with_feasibility_check`.
*   Sau đó, nó tính tổng chi phí (thường là tổng quãng đường di chuyển) của giải pháp đó bằng cách gọi `self.cvrp.calculate_solution_cost(solution)`.
*   **Quan trọng:** Nếu giải pháp kết quả không khả thi (sau khi đã cố gắng sửa chữa), một **hình phạt (penalty)** lớn sẽ được cộng vào chi phí. Điều này làm cho các giải pháp không khả thi trở nên kém hấp dẫn hơn đối với thuật toán, hướng nó tìm kiếm các giải pháp khả thi.

## 7. Các Toán Tử Di Truyền

### a. Chọn Lọc (Selection)

Mục đích của chọn lọc là chọn ra các cá thể từ quần thể hiện tại để làm cha mẹ cho thế hệ tiếp theo. Các cá thể có độ thích nghi cao hơn (chi phí thấp hơn) có nhiều khả năng được chọn hơn.

*   **`tournament_selection(population, fitness_values)`**:
    *   Chọn ngẫu nhiên `self.tournament_size` cá thể từ quần thể.
    *   Trả về cá thể có độ thích nghi tốt nhất (chi phí thấp nhất) trong số đó.
    *   Lặp lại hai lần để chọn hai cha mẹ.

*   **`roulette_wheel_selection(population, fitness_values)`**:
    *   Tính toán xác suất chọn cho mỗi cá thể, tỷ lệ nghịch với chi phí của nó (chi phí càng thấp, xác suất càng cao).
    *   Thực hiện lựa chọn dựa trên các xác suất này (giống như quay một vòng quay roulette).
    *   Cần xử lý trường hợp chi phí bằng 0 hoặc các giá trị fitness âm (nếu fitness không phải là chi phí). Mã hiện tại giả định fitness là chi phí (giá trị dương).

*   **`rank_selection(population, fitness_values)`**:
    *   Sắp xếp các cá thể theo độ thích nghi của chúng.
    *   Gán xác suất chọn dựa trên thứ hạng (rank) của chúng, thay vì giá trị độ thích nghi tuyệt đối. Điều này có thể giúp tránh việc một vài cá thể quá vượt trội chiếm ưu thế quá sớm.

Một phương thức `select_parent` chung có thể được sử dụng để gọi phương thức chọn lọc cụ thể dựa trên `self.selection_method`.

```python
    def select_parent(self, population, fitness_values):
        if self.selection_method == "tournament":
            return self.tournament_selection(population, fitness_values)
        elif self.selection_method == "roulette":
            return self.roulette_wheel_selection(population, fitness_values)
        elif self.selection_method == "rank":
            return self.rank_selection(population, fitness_values)
        else: # Mặc định là tournament
            return self.tournament_selection(population, fitness_values)
```

### b. Lai Ghép (Crossover)

Lai ghép kết hợp vật liệu di truyền từ hai cha mẹ để tạo ra một hoặc hai con cái.

*   **`ordered_crossover (OX1)`**:
    *   Chọn một đoạn (substring) ngẫu nhiên từ cha mẹ 1.
    *   Sao chép đoạn này vào con cái tại cùng vị trí.
    *   Các vị trí còn lại trong con cái được điền bằng các gen từ cha mẹ 2, theo thứ tự xuất hiện của chúng trong cha mẹ 2, bỏ qua các gen đã có từ cha mẹ 1.

*   **`partially_mapped_crossover (PMX)`**:
    *   Chọn hai điểm cắt ngẫu nhiên, xác định một đoạn ánh xạ.
    *   Các gen trong đoạn này được trao đổi trực tiếp giữa hai cha mẹ để tạo con cái.
    *   Đối với các gen bên ngoài đoạn ánh xạ, nếu một gen từ cha mẹ 1 dẫn đến xung đột (đã tồn tại trong con cái do đoạn ánh xạ từ cha mẹ 2), thì gen đó được thay thế bằng gen tương ứng trong ánh xạ.

*   **`cycle_crossover (CX)`**:
    *   Phức tạp hơn, đảm bảo rằng mỗi gen trong con cái đến từ một trong hai cha mẹ tại cùng một vị trí.
    *   Nó xác định các "chu trình" (cycles) giữa hai cha mẹ. Gen tại vị trí bắt đầu của một chu trình từ cha mẹ 1 được sao chép vào con cái. Sau đó, gen tại cùng vị trí đó từ cha mẹ 2 được xem xét. Vị trí của gen này trong cha mẹ 1 được tìm, và gen tại vị trí đó từ cha mẹ 1 lại được sao chép vào con cái, cứ tiếp tục như vậy cho đến khi quay lại vị trí bắt đầu.
    *   Đối với các chu trình khác nhau, các gen có thể được lấy xen kẽ từ cha mẹ 1 hoặc cha mẹ 2.

Một phương thức `crossover` chung có thể gọi phương thức lai ghép cụ thể:
```python
    def crossover(self, parent1, parent2):
        if self.crossover_method == "ordered":
            return self.ordered_crossover(parent1, parent2)
        elif self.crossover_method == "partially_mapped":
            return self.partially_mapped_crossover(parent1, parent2)
        elif self.crossover_method == "cycle":
            return self.cycle_crossover(parent1, parent2)
        else: # Mặc định là ordered
            return self.ordered_crossover(parent1, parent2)
```

### c. Đột Biến (Mutation)

Đột biến thay đổi ngẫu nhiên một hoặc nhiều gen trong một nhiễm sắc thể để tạo ra sự đa dạng và giúp thuật toán thoát khỏi các điểm tối ưu cục bộ.

*   **`swap_mutation`**: Chọn hai vị trí ngẫu nhiên trong nhiễm sắc thể và hoán đổi các gen tại hai vị trí đó.
*   **`insert_mutation`**: Chọn một gen ngẫu nhiên và một vị trí chèn ngẫu nhiên. Di chuyển gen đó đến vị trí chèn, dịch chuyển các gen khác nếu cần.
*   **`inversion_mutation`**: Chọn một đoạn ngẫu nhiên trong nhiễm sắc thể và đảo ngược thứ tự các gen trong đoạn đó.
*   **`scramble_mutation`**: Chọn một đoạn ngẫu nhiên trong nhiễm sắc thể và xáo trộn (permute) các gen trong đoạn đó một cách ngẫu nhiên.

Một phương thức `mutate` chung:
```python
    def mutate(self, chromosome):
        # Lưu ý: các hàm đột biến này sửa đổi chromosome trực tiếp (in-place)
        if self.mutation_method == "swap":
            self.swap_mutation(chromosome)
        elif self.mutation_method == "insert":
            self.insert_mutation(chromosome)
        elif self.mutation_method == "inversion":
            self.inversion_mutation(chromosome)
        elif self.mutation_method == "scramble":
            self.scramble_mutation(chromosome)
        # Không cần trả về vì chromosome được sửa đổi in-place
```

## 8. Tìm Kiếm Cục Bộ 2-Opt (`local_search_2opt` và `apply_2opt`)

```python
    def local_search_2opt(self, solution):
        if not solution: return [] # Trả về rỗng nếu không có giải pháp
        improved_solution = [list(route) for route in solution] # Tạo bản sao để làm việc
        made_improvement = True

        while made_improvement:
            made_improvement = False
            for r_idx, route in enumerate(improved_solution):
                if len(route) < 2: continue # Cần ít nhất 2 khách hàng để thực hiện 2-opt trên một cạnh

                # Tạo tuyến đường hoàn chỉnh bao gồm kho (0) ở đầu và cuối
                # để tính toán chi phí của các cạnh liên quan đến kho
                full_route_for_cost = [0] + route + [0]
                
                best_route_in_iteration = list(route) # Lưu tuyến tốt nhất trong lần lặp 2-opt này
                current_best_route_cost = self.cvrp.calculate_route_cost(route)


                # 2-opt cho một tuyến đơn lẻ
                # (i, j) là các chỉ số của các cạnh cần được xem xét để hoán đổi
                # Cạnh 1: (route[i], route[i+1])
                # Cạnh 2: (route[j], route[j+1])
                # Sau khi hoán đổi: (route[i], route[j]) và (route[i+1], route[j+1])
                # và đảo ngược đoạn giữa route[i+1] và route[j]
                for i in range(len(route) - 1): # Từ khách hàng đầu tiên đến kế cuối
                    for j in range(i + 1, len(route)): # Từ i+1 đến khách hàng cuối cùng
                        if i == j : continue # Không thể hoán đổi cùng một điểm
                        
                        # Tạo tuyến mới bằng cách áp dụng 2-opt swap
                        new_route = list(route) # Bắt đầu từ tuyến gốc của lần lặp 2-opt này
                        
                        # Đoạn cần đảo ngược là từ new_route[i+1] đến new_route[j]
                        segment_to_reverse = new_route[i+1 : j+1]
                        segment_to_reverse.reverse()
                        
                        # Tạo tuyến mới sau khi đảo ngược
                        temp_new_route = new_route[:i+1] + segment_to_reverse + new_route[j+1:]
                        
                        # Kiểm tra tính khả thi của tuyến mới (quan trọng!)
                        # Ví dụ, nếu có ràng buộc về thời gian hoặc thứ tự ưu tiên, cần kiểm tra ở đây.
                        # Đối với CVRP cơ bản, chỉ cần kiểm tra sức chứa.
                        # Tuy nhiên, 2-opt trên một tuyến đã khả thi về sức chứa thường vẫn khả thi.
                        # Nhưng nếu có các ràng buộc phức tạp hơn, cần kiểm tra cẩn thận.
                        
                        new_route_cost = self.cvrp.calculate_route_cost(temp_new_route)

                        if new_route_cost < current_best_route_cost:
                            best_route_in_iteration = list(temp_new_route)
                            current_best_route_cost = new_route_cost
                            made_improvement = True # Đánh dấu có sự cải thiện trong toàn bộ giải pháp
                
                improved_solution[r_idx] = best_route_in_iteration # Cập nhật tuyến trong giải pháp

        return improved_solution

    def apply_2opt(self, route): # Phương thức này dường như không được sử dụng trực tiếp trong local_search_2opt hiện tại
                                # mà logic 2-opt được triển khai bên trong local_search_2opt
        # ... (Logic tương tự như vòng lặp bên trong của local_search_2opt cho một tuyến) ...
        # Có vẻ như đây là một phiên bản cũ hơn hoặc một helper chưa được tích hợp hoàn toàn.
        # Logic chính của 2-opt cho một tuyến đã nằm trong local_search_2opt.
        pass
```
*   **2-Opt** là một thuật toán tìm kiếm cục bộ phổ biến để cải thiện các giải pháp cho các bài toán định tuyến.
*   Nguyên tắc cơ bản: Chọn hai cạnh không kề nhau trong một tuyến đường, loại bỏ chúng và nối lại các điểm cuối theo một cách khác để tạo thành một tuyến đường mới. Nếu tuyến đường mới có chi phí thấp hơn, nó sẽ được giữ lại.
*   `local_search_2opt(solution)`: Áp dụng 2-opt cho mỗi tuyến đường trong `solution`. Nó lặp lại quá trình này cho đến khi không tìm thấy cải thiện nào nữa.
*   Bên trong, nó duyệt qua tất cả các cặp cạnh có thể có trong một tuyến, thử thực hiện hoán đổi 2-opt (bằng cách đảo ngược đoạn giữa hai điểm cuối của các cạnh được chọn), tính toán chi phí mới và cập nhật nếu có cải thiện.
*   **Lưu ý quan trọng:** Việc áp dụng 2-opt phải đảm bảo rằng tuyến đường mới vẫn khả thi (ví dụ: không vi phạm các ràng buộc về thời gian nếu có). Trong CVRP cơ bản, nếu tuyến ban đầu khả thi về sức chứa, thì 2-opt thường không làm thay đổi tổng nhu cầu, do đó vẫn khả thi về sức chứa.

## 9. Tính Toán Độ Đa Dạng (`calculate_diversity`)

```python
    def calculate_diversity(self, population):
        # Tính toán độ đa dạng của quần thể
        # Một cách đơn giản là tính khoảng cách Hamming trung bình giữa các cặp nhiễm sắc thể
        # Hoặc sự khác biệt về cấu trúc giải pháp
        if not population or len(population) < 2:
            return 0.0

        total_distance = 0
        num_pairs = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Sử dụng khoảng cách Hamming cho nhiễm sắc thể (số vị trí khác nhau)
                # chromosome1 = population[i]
                # chromosome2 = population[j]
                # dist = sum(1 for k in range(len(chromosome1)) if chromosome1[k] != chromosome2[k])
                # total_distance += dist
                
                # Hoặc dựa trên sự khác biệt về chi phí của giải pháp được giải mã
                sol1 = self.decode_chromosome_with_feasibility_check(population[i])
                cost1 = self.cvrp.calculate_solution_cost(sol1)
                
                sol2 = self.decode_chromosome_with_feasibility_check(population[j])
                cost2 = self.cvrp.calculate_solution_cost(sol2)
                
                # Sử dụng sự khác biệt tuyệt đối của chi phí (đã chuẩn hóa nếu cần)
                # Đây là một thước đo đơn giản, có thể không phản ánh hết độ đa dạng cấu trúc.
                total_distance += abs(cost1 - cost2)

                num_pairs += 1
        
        if num_pairs == 0:
            return 0.0
        
        average_diversity = total_distance / num_pairs
        
        # Chuẩn hóa nếu cần (ví dụ, chia cho chi phí trung bình hoặc một giá trị tham chiếu)
        # avg_cost_of_population = sum(self.evaluate_fitness(ind) for ind in population) / len(population)
        # if avg_cost_of_population > 0:
        #     return average_diversity / avg_cost_of_population
        
        return average_diversity # Giá trị này cần được diễn giải trong ngữ cảnh của bài toán
```

*   Độ đa dạng của quần thể là một thước đo quan trọng. Quần thể đa dạng giúp thuật toán khám phá các vùng khác nhau của không gian tìm kiếm và tránh hội tụ sớm vào một giải pháp dưới tối ưu.
*   Phương thức này tính toán độ đa dạng bằng cách so sánh các cặp nhiễm sắc thể trong quần thể.
*   Có nhiều cách để đo độ đa dạng:
    *   **Khoảng cách Hamming:** Đếm số lượng gen khác nhau giữa hai nhiễm sắc thể. (Đã được comment lại trong code)
    *   **Dựa trên chi phí:** Tính sự khác biệt về chi phí của các giải pháp được giải mã từ các nhiễm sắc thể. (Cách đang được sử dụng)
    *   Các thước đo phức tạp hơn dựa trên cấu trúc của giải pháp.
*   Giá trị độ đa dạng có thể được sử dụng để điều chỉnh các tham số của thuật toán một cách tự động (ví dụ: tăng tỷ lệ đột biến nếu độ đa dạng thấp).

## 10. Các Hàm Điều Khiển (`stop`, `pause`, `resume`)

*   **`stop()`**: Đặt cờ `self.stop_flag = True` để báo hiệu cho vòng lặp chính của thuật toán dừng lại.
*   **`pause()`**: Đặt `self.paused = True`. Vòng lặp chính sẽ chờ trên `self.pause_condition` khi cờ này được đặt.
*   **`resume()`**: Đặt `self.paused = False` và thông báo (`notify_all()`) cho `self.pause_condition` để vòng lặp chính tiếp tục chạy.

Đây là những thành phần chính của thuật toán di truyền được triển khai trong `core/genetic.py`. Sự kết hợp của các toán tử chọn lọc, lai ghép, đột biến, cùng với các cơ chế sửa chữa và đánh giá, cho phép thuật toán tìm kiếm các giải pháp ngày càng tốt hơn cho bài toán CVRP. 