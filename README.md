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

## Ứng dụng Đánh giá Tham số (Parameter Tester)

Ứng dụng Parameter Tester cho phép thử nghiệm các cấu hình tham số khác nhau cho thuật toán ACO và GA để tìm ra bộ tham số tối ưu cho các bài toán CVRP.

### Các tính năng chính:
- Tạo và tải các bài toán CVRP
- Cấu hình nhiều bộ tham số khác nhau cho ACO và GA
- Thực hiện nhiều lần chạy và so sánh kết quả
- Trực quan hóa kết quả qua các biểu đồ

## Ứng dụng So sánh Thuật toán (Benchmark Application)

**Mới!** Ứng dụng Benchmark cho phép so sánh hiệu suất của thuật toán GA và ACO với tham số tối ưu trên nhiều bộ dữ liệu CVRP tiêu chuẩn.

### Các bộ dữ liệu được hỗ trợ:
- Augerat (bộ dữ liệu nhỏ)
- Christofides-Mingozzi-Toth (CMT)
- Golden 
- X-n Series (Uchoa et al., 2017)

### Các tính năng chính:
- Tự động tải và phân tích bộ dữ liệu CVRP tiêu chuẩn
- Phân tích phân bố khách hàng (ngẫu nhiên, cụm, hỗn hợp)
- So sánh hiệu suất GA và ACO với tham số tối ưu
- Tạo báo cáo và biểu đồ so sánh chi tiết
- Phân tích thống kê về chất lượng giải pháp và thời gian chạy

# Tính năng mới:

- **Tùy chọn dùng dữ liệu mẫu**: Ứng dụng bây giờ có thể chạy với các tập dữ liệu mẫu được tạo sẵn, không cần tải dữ liệu từ internet
- **Tự động phục hồi**: Khi không thể tải hoặc đọc dữ liệu từ bộ dữ liệu, ứng dụng tự động chuyển sang sử dụng dữ liệu mẫu
- **Hỗ trợ nhiều phân bố**: Dữ liệu mẫu bao gồm các mẫu với phân bố khách hàng ngẫu nhiên, theo cụm và hỗn hợp

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
├── parameter_tester.py      # Ứng dụng thử nghiệm tham số
├── benchmark_app/           # Ứng dụng so sánh hiệu suất thuật toán
│   ├── main.py              # Điểm khởi chạy ứng dụng benchmark
│   ├── gui.py               # Giao diện người dùng cho ứng dụng benchmark
│   ├── data_loader.py       # Module tải và phân tích bộ dữ liệu
│   ├── benchmark_runner.py  # Module chạy thử nghiệm benchmark
│   ├── datasets/            # Thư mục chứa bộ dữ liệu đã tải về
│   └── results/             # Thư mục lưu trữ kết quả benchmark
```

## Yêu cầu

- Python 3.6 trở lên
- NumPy
- Matplotlib
- Pandas
- Tkinter

## Cài đặt

1. Tải mã nguồn:
```bash
git clone https://github.com/AnhQu4nn/cvrp_simulator.git
cd cvrp_simulator
```

2. Cài đặt các gói cần thiết:
```bash
pip install -r requirements.txt
```

3. Chạy ứng dụng chính:
```bash
python main.py
```

4. Chạy ứng dụng thử nghiệm tham số:
```bash
python parameter_tester.py
```

5. Chạy ứng dụng benchmark:
```bash
cd benchmark_app
python main.py
```

## Cách sử dụng ứng dụng Benchmark

1. Chọn một bộ dữ liệu từ menu (Augerat, CMT, Golden, hoặc X-n Series)
2. Nhập kích thước bài toán tối đa (tùy chọn)
3. Nhấn "Load Instances" để tải các bài toán từ bộ dữ liệu
4. Chọn các bài toán để so sánh
5. Đặt số lần chạy và giới hạn thời gian 
6. Nhấn "Run Benchmark" để bắt đầu so sánh
7. Xem kết quả và biểu đồ trong các tab tương ứng
8. Tạo báo cáo chi tiết bằng cách nhấn "Generate Report"

## Lưu ý

Các bộ dữ liệu sẽ được tự động tải về từ thư viện CVRPLIB khi cần thiết. Quá trình tải lần đầu có thể mất một chút thời gian.

## Tài liệu tham khảo

- CVRPLIB: http://vrp.atd-lab.inf.puc-rio.br/
- Augerat et al. datasets: http://vrp.atd-lab.inf.puc-rio.br/index.php/en/
- Christofides, Mingozzi and Toth datasets: http://vrp.atd-lab.inf.puc-rio.br/index.php/en/
- Uchoa et al. X-n Series datasets: http://vrp.atd-lab.inf.puc-rio.br/index.php/en/

Dự án này được phát triển bởi Nguyễn Anh Quân và Vũ Quốc Long.

## Công cụ chuyển đổi dữ liệu CVRPLIB

Để sử dụng các bộ dữ liệu từ CVRPLIB trong phần mềm này, bạn có thể chuyển đổi từ định dạng .vrp sang định dạng JSON phù hợp bằng công cụ `convert_cvrplib.py`.

### Cách sử dụng:

1. **Chuyển đổi một file VRP đơn lẻ:**
   ```
   python convert_cvrplib.py -i path/to/file.vrp -o output/file.json
   ```

2. **Chuyển đổi toàn bộ thư mục:**
   ```
   python convert_cvrplib.py -i path/to/directory -o output/directory
   ```

3. **Nạp file đã chuyển đổi vào phần mềm:**
   - Khởi động phần mềm với `python main.py`
   - Chọn thuật toán (Ant Colony hoặc Genetic)
   - Nhấn nút "Tải bài toán"
   - Duyệt đến file JSON đã chuyển đổi và mở

### Cấu trúc dữ liệu JSON:

```json
{
  "capacity": 206,
  "depot": {
    "x": 365.0,
    "y": 689.0
  },
  "customers": [
    {
      "id": 2,
      "x": 146.0,
      "y": 180.0,
      "demand": 38
    },
    // ...các khách hàng khác
  ]
}
```

### Lưu ý:
- Thuật toán CVRP trong phần mềm này hoạt động tốt nhất với các bài toán có kích thước vừa và nhỏ (dưới 200 khách hàng)
- Các instance lớn hơn có thể mất nhiều thời gian để giải quyết
- Thư mục `json_datasets` chứa các bài toán CVRP từ thư mục `datasets` đã được chuyển đổi
