# Tính năng nâng cao trong CVRP Simulator

## Thuật toán ACO (Ant Colony Optimization)

### 1. MIN-MAX ACO
MIN-MAX ACO (Max-Min Ant System) là phiên bản cải tiến của ACO, giới hạn lượng pheromone trong một khoảng [min, max] để tránh hội tụ sớm vào các lời giải cục bộ. Cơ chế này giúp tăng khả năng khám phá không gian tìm kiếm.

**Khi nào sử dụng:** Khi thuật toán dễ bị mắc kẹt ở các cực tiểu cục bộ hoặc hội tụ quá nhanh vào một lời giải chưa tối ưu.

### 2. Tìm kiếm cục bộ (Local Search)
Áp dụng thuật toán cải tiến 2-opt trên mỗi tuyến đường sau khi kiến đã tạo ra lời giải. Kỹ thuật này sẽ thử hoán đổi các cạnh nếu việc hoán đổi giúp giảm chi phí tuyến đường.

**Khi nào sử dụng:** Khi cần tinh chỉnh lời giải để có kết quả tốt hơn, đặc biệt là trong các bài toán có nhiều khách hàng.

### 3. Kiến ưu tú (Elitist Ants)
Cho phép một số kiến ưu tú (có lời giải tốt nhất) được phép đặt thêm pheromone trên tuyến đường của chúng. Cơ chế này giúp tăng cường khai thác các lời giải tốt đã tìm được.

**Khi nào sử dụng:** Khi muốn tăng tốc quá trình hội tụ và khai thác vùng xung quanh lời giải tốt.

## Thuật toán GA (Genetic Algorithm)

### 1. Phương pháp chọn lọc (Selection Methods)

- **Tournament Selection**: Chọn ngẫu nhiên một số cá thể và lấy cá thể tốt nhất trong nhóm.
  - *Tournament Size*: Số cá thể tham gia vào mỗi vòng giải đấu (càng lớn thì áp lực chọn lọc càng cao).

- **Roulette Wheel Selection**: Xác suất chọn cá thể tỷ lệ thuận với độ thích nghi (fitness).

- **Rank Selection**: Xác suất chọn cá thể dựa trên thứ hạng thay vì giá trị fitness trực tiếp.

**Khi nào sử dụng:**
- Tournament: Khi có sự chênh lệch lớn giữa các giá trị fitness.
- Roulette: Khi sự chênh lệch fitness không quá lớn.
- Rank: Khi muốn giảm áp lực chọn lọc và duy trì đa dạng di truyền.

### 2. Phương pháp lai ghép (Crossover Methods)

- **Ordered Crossover (OX)**: Giữ nguyên một đoạn gen từ cha/mẹ đầu tiên và điền các gen còn thiếu theo thứ tự từ cha/mẹ thứ hai.

- **Partially Mapped Crossover (PMX)**: Xây dựng bản đồ ánh xạ gen từ hai cha mẹ để tạo ra con lai.

- **Cycle Crossover (CX)**: Giữ nguyên vị trí các gen theo chu kỳ từ cha mẹ.

**Khi nào sử dụng:**
- OX: Hiệu quả cho bài toán TSP và CVRP nói chung.
- PMX: Tốt khi cần duy trì mối quan hệ tương đối giữa các gen.
- CX: Tốt khi vị trí tuyệt đối của các gen là quan trọng.

### 3. Phương pháp đột biến (Mutation Methods)

- **Swap Mutation**: Hoán đổi vị trí của hai gen ngẫu nhiên.
- **Insert Mutation**: Di chuyển một gen từ vị trí này đến vị trí khác.
- **Inversion Mutation**: Đảo ngược thứ tự các gen trong một đoạn ngẫu nhiên.
- **Scramble Mutation**: Xáo trộn ngẫu nhiên các gen trong một đoạn.

**Khi nào sử dụng:**
- Swap: Đơn giản và hiệu quả cho hầu hết các trường hợp.
- Insert: Khi cần thay đổi nhỏ trên chuỗi.
- Inversion: Khi cần thay đổi lớn nhưng vẫn giữ cấu trúc cục bộ.
- Scramble: Khi cần thay đổi mạnh mẽ để thoát khỏi cực tiểu cục bộ.

### 4. Dừng sớm (Early Stopping)
Dừng thuật toán khi không có cải thiện sau một số thế hệ nhất định. Giúp tiết kiệm thời gian tính toán khi thuật toán đã hội tụ.

**Khi nào sử dụng:** Khi không muốn chạy hết số thế hệ tối đa và tiết kiệm thời gian khi thuật toán đã hội tụ.

## Hướng dẫn sử dụng

1. Mở ứng dụng CVRP Simulator
2. Chọn thuật toán ACO hoặc GA
3. Tạo bài toán CVRP mới hoặc tải bài toán có sẵn
4. Cấu hình các tham số thuật toán trong tab "Tính năng nâng cao"
5. Nhấn "Chạy" để bắt đầu giải quyết bài toán

## Lưu ý

- Việc chọn tham số phù hợp ảnh hưởng lớn đến hiệu suất thuật toán
- Nên thử nghiệm với nhiều cấu hình khác nhau để tìm ra cấu hình tốt nhất cho từng bài toán
- Các đồ thị phân tích trên tab "Phân tích" sẽ giúp đánh giá hiệu quả của các tham số 