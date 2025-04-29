# Phần mềm mô phỏng thuật toán CVRP

Ứng dụng Python để giải quyết và trực quan hóa bài toán Định tuyến phương tiện có giới hạn sức chứa (CVRP) sử dụng các thuật toán tối ưu.

## Tác giả
- **Nguyễn Anh Quân**
- **Vũ Quốc Long**

## Tổng quan

Bài toán Định tuyến phương tiện có giới hạn sức chứa (CVRP) là một biến thể của bài toán định tuyến phương tiện, trong đó nhiều phương tiện có sức chứa giới hạn cần phục vụ các khách hàng với nhu cầu đã biết trước, xuất phát và kết thúc tại một kho hàng. Mục tiêu là giảm thiểu tổng quãng đường di chuyển trong khi đáp ứng tất cả nhu cầu của khách hàng mà không vượt quá sức chứa của phương tiện.

Ứng dụng này cung cấp giao diện người dùng đồ họa để:
- Tạo và tải các bài toán CVRP
- Giải quyết CVRP bằng các thuật toán khác nhau
- Trực quan hóa quá trình giải quyết theo thời gian thực
- So sánh hiệu suất của các thuật toán

## Các thuật toán được triển khai

- **Thuật toán Tối ưu hóa đàn kiến (ACO)**: Thuật toán meta-heuristic lấy cảm hứng từ hành vi tìm kiếm thức ăn của kiến, sử dụng các vết pheromone để định hướng tìm kiếm.
- **Thuật toán Di truyền (GA)**: Thuật toán tiến hóa sử dụng các cơ chế lấy cảm hứng từ sự tiến hóa sinh học như đột biến, lai ghép và chọn lọc.

## Cấu trúc ứng dụng

```
cvrp_app/
├── main.py                  # Điểm khởi chạy chính
├── core/                    # Các thuật toán cốt lõi và định nghĩa bài toán
│   ├── cvrp.py              # Định nghĩa bài toán CVRP
│   ├── aco.py               # Thuật toán Tối ưu hóa đàn kiến
│   └── genetic.py           # Thuật toán Di truyền
├── gui/                     # Các module giao diện
│   ├── chon_thuat_toan.py   # Menu chọn thuật toán
│   ├── ung_dung_aco.py      # Ứng dụng Tối ưu hóa đàn kiến
│   ├── ung_dung_di_truyen.py # Ứng dụng Thuật toán Di truyền
│   ├── ung_dung_so_sanh.py  # Ứng dụng so sánh thuật toán
│   └── truc_quan_hoa/       # Các module trực quan hóa
│       ├── co_so.py         # Lớp trực quan hóa cơ sở
│       ├── aco_viz.py       # Trực quan hóa ACO
│       └── genetic_viz.py   # Trực quan hóa Thuật toán Di truyền
```

## Yêu cầu

- Python 3.6 trở lên
- NumPy
- Matplotlib
- Tkinter

## Cài đặt

1. Tải mã nguồn:
```bash
git clone https://github.com/AnhQu4nn/cvrp_simulator.git
cd cvrp_simulator
```

2. Cài đặt các gói cần thiết:
```bash
pip install numpy matplotlib
```

3. Chạy ứng dụng:
```bash
python main.py
```

## Cách sử dụng

1. Từ menu chính, chọn thuật toán bạn muốn sử dụng hoặc công cụ so sánh.
2. Tạo một bài toán CVRP mới bằng cách:
   - Tạo bài toán ngẫu nhiên với số lượng khách hàng và sức chứa phương tiện đã chỉ định
   - Tải bài toán đã lưu trước đó từ file JSON
3. Cấu hình tham số thuật toán theo nhu cầu của bạn
4. Bắt đầu thuật toán và quan sát quá trình tìm giải pháp được trực quan hóa
5. Xem kết quả cuối cùng và nếu muốn, lưu bài toán để sử dụng trong tương lai

## Tính năng

- **Trực quan hóa thời gian thực** về tiến trình của thuật toán
- **Tinh chỉnh tham số** để thử nghiệm với các cài đặt thuật toán khác nhau
- **Tạo và lưu bài toán** để tái tạo và chia sẻ
- **So sánh thuật toán** để đánh giá sự khác biệt về hiệu suất
- **Giao diện tương tác** với theo dõi tiến trình và điều khiển thuật toán

## Ví dụ

### Tạo bài toán ngẫu nhiên

Bạn có thể tạo một bài toán CVRP ngẫu nhiên bằng cách chỉ định số lượng khách hàng và sức chứa phương tiện. Tùy chọn, bạn có thể cung cấp seed để tái tạo lại bài toán.

### Lưu và tải bài toán

Các bài toán có thể được lưu vào và tải từ các file JSON, tạo điều kiện cho việc chia sẻ và tái tạo kết quả.

### So sánh thuật toán

Công cụ so sánh cho phép bạn chạy nhiều thuật toán trên cùng một bài toán và so sánh hiệu suất của chúng về:
- Thời gian thực thi
- Chất lượng giải pháp (tổng quãng đường)
- Số lượng tuyến đường được tạo ra

###

Dự án này được phát triển bởi Nguyễn Anh Quân và Vũ Quốc Long.
