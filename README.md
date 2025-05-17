# CVRP Simulator

## Giới thiệu
CVRP Simulator là một ứng dụng mô phỏng trực quan giúp giải và so sánh các thuật toán cho Bài toán Định tuyến Phương tiện Có Giới hạn Tải trọng (Capacitated Vehicle Routing Problem - CVRP). Ứng dụng này được phát triển trong khuôn khổ môn học Thiết kế và Phân tích Thuật toán (DAA) năm 2025. Nó cung cấp giao diện trực quan để tạo bài toán, chạy thuật toán và phân tích kết quả, nhằm mục đích nghiên cứu và học tập.

## Tính năng chính

- **Hai thuật toán tiên tiến**:
  - **Ant Colony Optimization (ACO)**: Thuật toán tối ưu đàn kiến với nhiều tùy chọn cấu hình nâng cao
  - **Genetic Algorithm (GA)**: Thuật toán di truyền với đa dạng toán tử chọn lọc, lai ghép và đột biến

- **Môi trường thử nghiệm toàn diện**:
  - Tạo, lưu và tải bài toán CVRP
  - Tùy chỉnh đa dạng thông số thuật toán
  - Theo dõi tiến trình thuật toán theo thời gian thực
  - Xuất kết quả và phân tích

- **Chức năng so sánh đa thuật toán**:
  - So sánh ACO và GA trên cùng một bài toán
  - Đánh giá thời gian thực thi và chất lượng giải pháp
  - Tối ưu hóa CPU cho hiệu suất tốt nhất

- **Công cụ kiểm thử tham số**:
  - Chạy nhiều cấu hình thuật toán khác nhau
  - Phân tích ảnh hưởng của các tham số đến chất lượng giải pháp
  - Tìm cấu hình tối ưu cho từng bài toán cụ thể

- **Trực quan hóa đa chiều**:
  - Hiển thị giải pháp trên bản đồ
  - Biểu đồ hội tụ theo thời gian
  - Thống kê chi tiết về tiến trình thuật toán

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Các thư viện Python:
  - NumPy
  - Matplotlib
  - TkInter
  - Pillow
  - Pandas

## Cài đặt

```bash
# Clone mã nguồn
git clone https://github.com/AnhQu4nn/cvrp_simulator.git
cd cvrp_simulator

# Tạo môi trường ảo (tùy chọn)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

## Sử dụng

### Khởi động ứng dụng

```bash
python main.py
```

### Hướng dẫn sử dụng cơ bản

1. **Màn hình chính**:
   - Chọn một trong các tùy chọn: ACO, GA, So sánh thuật toán, hoặc Kiểm thử tham số

2. **Tạo bài toán CVRP**:
   - Thiết lập số lượng khách hàng và sức chứa phương tiện
   - Nhấn "Tạo bài toán" để sinh ngẫu nhiên bài toán mới
   - Hoặc tải bài toán có sẵn qua "Tải bài toán"

3. **Thiết lập tham số thuật toán**:
   - Tùy chỉnh các tham số cơ bản và nâng cao của thuật toán
   - Ví dụ: Cho ACO - số kiến, alpha, beta, rho, và các tham số nâng cao khác
   - Ví dụ: Cho GA - kích thước quần thể, tỷ lệ đột biến, phương pháp lai ghép, phương pháp chọn lọc

4. **Chạy mô phỏng**:
   - Nhấn "Bắt đầu" để chạy thuật toán
   - Theo dõi quá trình tìm kiếm giải pháp và sự hội tụ của thuật toán
   - Tạm dừng hoặc dừng thuật toán khi cần thiết

5. **Phân tích kết quả**:
   - Xem biểu đồ hội tụ và thống kê chi tiết
   - Xuất kết quả để phân tích sâu hơn
   - Lưu giải pháp tốt nhất

### Chế độ so sánh thuật toán

1. Chọn "So sánh thuật toán" từ màn hình chính
2. Cấu hình tham số cho cả ACO và GA
3. Nhấn "Bắt đầu" để chạy đồng thời cả hai thuật toán
4. Theo dõi hiệu suất và kết quả của từng thuật toán trong thời gian thực
5. Xem kết quả so sánh chi tiết sau khi hoàn thành

### Chế độ kiểm thử tham số

1. Chọn "Kiểm thử tham số" từ màn hình chính
2. Chọn thuật toán (ACO hoặc GA) để thử nghiệm
3. Thiết lập các khoảng giá trị tham số cần kiểm tra
4. Chọn "Thêm nhiều cấu hình" để tạo tổ hợp tham số
5. Nhấn "Bắt đầu" để chạy tất cả các cấu hình
6. Phân tích kết quả để xác định cấu hình tối ưu

## Cấu trúc mã nguồn

```
cvrp_simulator/
├── main.py                # Điểm khởi chạy ứng dụng
├── core/                  # Các thuật toán cốt lõi
│   ├── aco.py             # Thuật toán Ant Colony Optimization
│   ├── genetic.py         # Thuật toán di truyền
│   └── cvrp.py            # Định nghĩa bài toán CVRP
├── gui/                   # Giao diện người dùng
│   ├── aco_app.py         # Giao diện cho thuật toán ACO
│   ├── genetic_app.py     # Giao diện cho thuật toán GA
│   ├── comparison_app.py  # Giao diện so sánh thuật toán
│   ├── selector.py        # Màn hình chọn chức năng
│   └── visualization/     # Các công cụ trực quan hóa
│       ├── base.py        # Lớp trực quan hóa cơ sở
│       ├── aco_viz.py     # Trực quan hóa cho ACO
│       └── genetic_viz.py # Trực quan hóa cho GA
└── parameter_tester.py    # Công cụ kiểm thử tham số
```

## Tùy chỉnh nâng cao

### Thuật toán ACO

- **Min-Max ACO**: Cải thiện chất lượng giải pháp bằng cách giới hạn lượng pheromone
- **Kiến ưu tú**: Tăng tốc độ hội tụ bằng cách cho kiến tốt nhất có ảnh hưởng lớn hơn
- **Pheromone ban đầu**: Điều chỉnh giá trị pheromone khởi tạo
- **Tìm kiếm cục bộ**: Áp dụng 2-opt để cải thiện chất lượng giải pháp

### Thuật toán GA

- **Phương pháp lai ghép**: Ordered (OX1), Partially Mapped (PMX), hoặc Cycle (CX)
- **Phương pháp chọn lọc**: Tournament, Roulette Wheel, hoặc Rank
- **Phương pháp đột biến**: Swap, Insert, Inversion, hoặc Scramble
- **Elitism**: Giữ lại các cá thể tốt nhất qua các thế hệ
- **Tìm kiếm cục bộ**: Áp dụng 2-opt để tối ưu hóa giải pháp

## Đóng góp và phát triển

Nếu bạn muốn đóng góp cho dự án, hãy làm theo các bước sau:

1. Fork repository
2. Tạo nhánh tính năng mới (`git checkout -b feature/amazing-feature`)
3. Commit thay đổi của bạn (`git commit -m 'Add some amazing feature'`)
4. Push lên nhánh của bạn (`git push origin feature/amazing-feature`)
5. Mở Pull Request

## Nhóm phát triển

Dự án được thực hiện bởi:
- Nguyễn Anh Quân
- Vũ Quốc Long

Đây là một sản phẩm trong khuôn khổ môn học Thiết kế và Phân tích Thuật toán (DAA) - 2025.

