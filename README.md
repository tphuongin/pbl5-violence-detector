# PBL5 - HỆ THỐNG PHÁT HIỆN HÀNH VI BẠO LỰC REAL-TIME TRÊN THIẾT BỊ EDGE (JETSON)

Đây là đồ án môn học **PBL5 (Project-Based Learning 5)** thuộc ngành Công nghệ Thông tin - Trường Đại học Bách khoa - Đại học Đà Nẵng (DUT).
Hệ thống sử dụng các mô hình học sâu (Deep Learning) tối ưu hóa thời gian để phát hiện hành vi bạo lực thời gian thực từ Camera giám sát, sau đó kích hoạt các cảnh báo vật lý (còi hú, gọi điện thoại khẩn cấp qua mạng di động) và quản lý tập trung thông qua Web Dashboard.

---

## 📌 Kiến trúc Tổng quan Hệ thống

Hệ thống được thiết kế theo mô hình **Edge-to-Cloud-to-Client** chia làm 3 thành phần chính:

```mermaid
graph TD
    subgraph Thiết bị Edge Jetson [Jetson Edge Device]
        Cam[Camera RTSP / USB / Clip] -->|Khung hình JPEG| Inf[Bộ suy diễn AI MoViNet/MobileNet TSM]
        Inf -->|Xác suất thô| Filter[Bộ lọc cảnh báo 3 lớp chống nhiễu]
        Filter -->|Điều khiển GPIO| Buzzer[Còi báo động vật lý]
        Filter -->|Lệnh AT qua Serial| GSM[Mạch SIM800L gọi điện thoại]
        HTTP[Aiohttp Server Port 8001] -->|Kích hoạt web_reset_event| Filter
    end

    subgraph Máy chủ Backend [FastAPI Backend Server]
        API[FastAPI Router] <--> DB[(Cơ sở dữ liệu MySQL)]
        API <-->|Truyền tải WebSocket| WS[WebSocket Relayer]
        API -->|Gửi yêu cầu POST /mute| HTTP
    end

    subgraph Giao diện Web [React Web UI]
        React[ReactJS Web Dashboard] <-->|WS Stream & Real-time Metadata| WS
        React -->|Tác vụ Quản lý & Tắt còi| API
    end
```

### Nguyên lý hoạt động:
1. **Thu nhận hình ảnh**: Thiết bị Edge (Jetson Nano/TX2) đọc luồng video từ Camera (USB/RTSP) hoặc tệp video chạy thử nghiệm qua OpenCV.
2. **Xử lý AI & Lọc nhiễu**: Các khung hình được đưa vào mô hình học sâu để phân loại hành vi. Kết quả xác suất thô được đưa qua **Bộ lọc cảnh báo 3 lớp** nhằm loại bỏ cảnh báo giả.
3. **Phản ứng tức thời (Edge Action)**: Khi bạo lực được xác nhận, Jetson sẽ trực tiếp kích hoạt chân GPIO ra còi báo động vật lý và gửi tập lệnh AT qua Serial sang Module GSM SIM800L để thực hiện cuộc gọi khẩn cấp tới số điện thoại cấu hình sẵn.
4. **Đồng bộ hóa & Truyền stream**: Thiết bị Jetson nén khung hình dạng JPEG gửi lên FastAPI Backend qua giao thức WebSocket cùng với các tham số nhận diện realtime. Backend sẽ chuyển tiếp luồng ảnh này đến người dùng đang theo dõi trên Web Dashboard.
5. **Điều khiển từ xa (Mute/Reset)**: Nếu phát hiện cảnh báo sai hoặc bạo lực đã được xử lý xong, nhân viên giám sát có thể bấm nút **Tắt còi** trên Web. Yêu cầu này đi qua Backend và được gửi tới cổng HTTP `8001` trên Jetson để tắt còi, hủy cuộc gọi khẩn cấp và khởi tạo thời gian mù 5 giây.

---

## ⚡ Các Tính năng Nổi bật

* **Nhận diện Bạo lực Real-time**: Sử dụng các mô hình video tiên tiến (MoViNet-A0-TSM / MobileNetV2-TSM) xử lý trực tiếp trên thiết bị nhúng Jetson với độ trễ thấp và FPS tối ưu.
* **Bộ lọc Cảnh báo 3 Lớp (3-Layer Alert Filter)**: Thuật toán độc quyền giúp giảm thiểu tối đa các cảnh báo giả từ nhiễu động nhất thời của mô hình AI.
* **Gọi điện Cảnh báo Di động**: Tích hợp module phần cứng GSM SIM800L để gọi trực tiếp tới số điện thoại của quản lý/cảnh sát khi phát hiện sự cố.
* **Còi báo động vật lý**: Kích hoạt còi hú kết nối trực tiếp với chân GPIO trên thiết bị Edge.
* **Stream Video độ trễ cực thấp**: Truyền tải khung hình trực tiếp từ Jetson lên Web Client qua WebSocket, không cần cài đặt các máy chủ streaming RTSP phức tạp.
* **Quản trị Camera thông minh (Admin Dashboard)**: Cho phép thêm, sửa, xóa camera, cấu hình địa chỉ IP của từng camera và số điện thoại nhận cuộc gọi khẩn cấp trực tiếp từ giao diện Web.
* **Tắt còi báo động khẩn cấp thủ công**: Tích hợp nút tắt còi thiết kế cao cấp ở góc dưới bên phải màn hình Live Stream. Hỗ trợ tắt còi, dừng cuộc gọi đang thực hiện, reset trạng thái AI và áp dụng 5 giây miễn dịch (Grace Period) để tránh nhiễu lặp lại.
* **Phân tích Video Offline**: Trang phân tích độc lập cho phép tải lên clip video từ máy tính, server FastAPI sẽ gửi lệnh yêu cầu thiết bị Jetson phân tích và trả về video kết quả được vẽ bounding box và biểu đồ xác suất chi tiết.

---

## 🛠️ Công nghệ Sử dụng

### 1. Edge AI Client (Jetson)
* **Ngôn ngữ**: Python
* **Thư viện AI**: PyTorch, Torchvision, CUDA/TensorRT (tùy cấu hình Jetson)
* **Xử lý ảnh/Luồng**: OpenCV, GStreamer
* **Giao tiếp phần cứng**: 
  * `RPi.GPIO` hoặc Jetson.GPIO (Điều khiển chân còi hú).
  * `pySerial` (Điều khiển tập lệnh AT cho SIM800L).
* **HTTP Server**: `aiohttp` (Lắng nghe tín hiệu reset từ Backend).

### 2. Backend Server
* **Ngôn ngữ & Khung**: Python, FastAPI
* **Cơ sở dữ liệu**: MySQL (ORM SQLAlchemy)
* **Xác thực**: JWT (JSON Web Tokens)
* **Kết nối**: WebSockets (Đồng bộ luồng stream và thông số realtime), Httpx (Giao tiếp với Jetson).

### 3. Frontend Web Client
* **Khung**: ReactJS (Vite)
* **Giao diện**: Vanilla CSS (Hiệu ứng kính mờ Glassmorphism, thiết kế Dark Mode cao cấp, Responsive 2 cột).
* **Bộ icon**: Lucide React
* **Kết nối**: WebSocket Client, Fetch API.

---

## 📐 Thuật toán Lọc Cảnh báo 3 Lớp (3-Layer Alert Filtering)

Để đảm bảo hệ thống không báo động sai khi mô hình chỉ phát hiện 1-2 khung hình bạo lực ngẫu nhiên (nhiễu động), chúng tôi áp dụng bộ lọc 3 lớp:

1. **Lớp 1: Ngưỡng Xác suất Thô (Raw Threshold)**:
   Mô hình AI dự đoán xác suất bạo lực $P_{\text{raw}}$ của từng frame. Frame được coi là nghi ngờ bạo lực nếu:
   $$P_{\text{raw}} \ge \text{Ngưỡng xác suất (Thường là 0.70)}$$

2. **Lớp 2: Bộ lọc Làm mượt Exponential Moving Average (EMA)**:
   Để tránh việc xác suất nhảy vọt đột ngột, ta áp dụng công thức làm mượt theo thời gian:
   $$P_{\text{smooth}} = \alpha \cdot P_{\text{raw}} + (1 - \alpha) \cdot P_{\text{smooth\_prev}}$$
   *Với $\alpha$ (thường là 0.25) điều chỉnh độ nhạy bén và độ làm mịn.*

3. **Lớp 3: Xác nhận theo Cửa sổ thời gian (Window-based Confirmation)**:
   Hệ thống duy trì một bộ đếm số khung hình vượt ngưỡng liên tiếp (`confirm_count`).
   * **Kích hoạt còi & cuộc gọi**: Khi `confirm_count` đạt tới giới hạn cấu hình `confirm_needed` (ví dụ: 6 frame liên tục có $P_{\text{smooth}} \ge 0.75$), hệ thống mới xác nhận hành vi bạo lực và kích hoạt còi vật lý cùng cuộc gọi.
   * **Reset AI tự động**: Nếu có liên tiếp `8 frame` liên tục được dự đoán là bình thường (Normal), mô hình sẽ tự động làm trống bộ đệm trạng thái (mô hình TSM có trạng thái ẩn giữa các frame) để giải phóng bộ nhớ và tránh kẹt tín hiệu cũ.

### 🛑 Cơ chế Tắt còi & Khoảng thời gian mù (5s Grace Period)
Khi người dùng bấm nút "Tắt còi" trên Web UI:
1. Giao diện gửi yêu cầu `POST /api/cameras/{id}/mute` tới Backend.
2. Backend gửi tín hiệu đến Jetson Client.
3. Jetson Client thực hiện:
   * Kéo chân GPIO Buzzer xuống `LOW` ngay lập tức.
   * Gửi lệnh AT ngắt cuộc gọi sang module SIM800L.
   * Gọi hàm `model.reset_states()` để xóa sạch bộ đệm ẩn của mô hình AI.
   * Kích hoạt bộ đếm thời gian miễn dịch trong **5 giây**. Trong 5 giây này, mọi nhận diện bạo lực của AI đều bị bỏ qua, giúp cảnh quay xung quanh ổn định trở lại hoặc tạo điều kiện cho các nhân vật di chuyển ra ngoài vùng quét mà không kích hoạt chu kỳ cảnh báo mới.

---

## 🗄️ Cấu trúc Cơ sở Dữ liệu (MySQL Schema)

Cơ sở dữ liệu gồm 4 bảng chính được liên kết chặt chẽ:

1. **`USERS`**: Lưu thông tin người dùng được phân quyền.
   * `UserID` (PK), `Username`, `PasswordHash`, `Role` (admin / operator), `CreatedAt`.
2. **`CAMERAS`**: Quản lý thông tin và cấu hình camera.
   * `CameraID` (PK), `CameraName`, `CameraIP` (địa chỉ IP của Jetson), `CameraPhoneNum` (SĐT nhận cảnh báo khi có bạo lực), `CameraStatus` (Online/Offline), `CreatedAt`.
3. **`VIOLENCE_HISTORY`**: Lưu trữ lịch sử các vụ bạo lực được phát hiện.
   * `HistoryID` (PK), `CameraID` (FK), `Timestamp`, `Confidence` (độ tin cậy), `VideoUrl` (đường dẫn lưu trữ video bạo lực trên Cloudinary), `IsConfirmed` (được xác nhận hay là cảnh báo sai).
4. **`CALLS`**: Nhật ký các cuộc gọi cảnh báo đã thực hiện.
   * `CallID` (PK), `CameraID` (FK), `PhoneNumber` (số đã gọi), `Timestamp`, `Status` (Thành công / Thất bại).

---

## 📂 Cấu trúc Mã nguồn

```text
violence_detector/
├── backend/                  # Mã nguồn FastAPI Backend
│   ├── main.py               # File chạy chính của API & WebSocket
│   ├── database.py           # Kết nối MySQL bằng SQLAlchemy
│   ├── models.py             # Định nghĩa cấu trúc bảng Database
│   ├── schemas.py            # Pydantic Schemas phục vụ Request/Response Validation
│   ├── stream_service.py     # Dịch vụ quản lý stream và phân phối WebSocket
│   └── upload_service.py     # Dịch vụ tải video bạo lực lên Cloudinary
│
├── frontend/                 # Mã nguồn React Frontend (Vite)
│   ├── src/
│   │   ├── pages/            # Các trang giao diện (CameraPage, VideoAnalysisPage,...)
│   │   │   ├── CameraPage.jsx      # Giám sát camera và nút Tắt còi khẩn cấp
│   │   │   └── CameraPage.css      # CSS định vị tuyệt đối cho nút Mute
│   │   ├── components/       # Các components dùng chung (VideoPlayer, Header,...)
│   │   ├── services/         # apiService.js quản lý các kết nối HTTP/WebSocket
│   │   └── styles.css        # Stylesheet tổng thể của ứng dụng
│
└── models/                   # Mã nguồn chạy AI trên thiết bị nhúng Jetson
    ├── mobilenet_jetson_client.py    # Client nhận diện dùng MobileNetV2-TSM
    └── movinet/
        └── movinet_jetson_client.py  # Client nhận diện dùng MoViNet-A0-TSM
```

---

## 🚀 Hướng dẫn Vận hành và Chạy thử

### Bước 1: Chuẩn bị Cơ sở dữ liệu MySQL
1. Khởi động MySQL Server trên máy chủ.
2. Tạo một database mới tên là `pbl5`:
   ```sql
   CREATE DATABASE pbl5 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```

### Bước 2: Cấu hình và Chạy Backend FastAPI
1. Di chuyển vào thư mục backend:
   ```bash
   cd backend
   ```
2. Tạo tệp cấu hình `.env` dựa trên `.env.example` và điền đầy đủ các thông tin:
   * Kết nối DB: `DATABASE_URL=mysql+pymysql://username:password@localhost:3306/pbl5`
   * Cấu hình Cloudinary để lưu video sự kiện bạo lực.
   * Cấu hình số điện thoại mặc định, cổng HTTP kết nối Jetson (`JETSON_HTTP_PORT=8001`).
3. Khởi tạo dữ liệu mẫu (Tạo sẵn tài khoản admin/operator và các camera mẫu):
   ```bash
   python seed_data.py
   ```
4. Chạy server backend:
   ```bash
   python main.py
   ```
   *Mặc định API chạy tại `http://localhost:8000`.*

### Bước 3: Cấu hình và Chạy Frontend React
1. Di chuyển vào thư mục frontend:
   ```bash
   cd ../frontend
   ```
2. Cài đặt các thư viện phụ thuộc:
   ```bash
   npm install
   ```
3. Chạy server phát triển (Development):
   ```bash
   npm run dev
   ```
   *Mở trình duyệt truy cập địa chỉ được hiển thị trên terminal (ví dụ: `http://localhost:5173`). Đăng nhập bằng tài khoản admin đã được tạo sẵn qua file seed.*

### Bước 4: Chạy Client nhận diện AI trên Jetson
1. Đảm bảo thiết bị đã kết nối Camera (USB hoặc RTSP) và mạch GSM SIM800L (nếu có cuộc gọi).
2. Di chuyển vào thư mục models và chạy client AI tương ứng (Ví dụ chạy MoViNet):
   ```bash
   cd ../models/movinet
   python movinet_jetson_client.py --camera_id 1 --rtsp rtsp://username:password@ip_camera
   ```
   *(Nếu chạy thử nghiệm bằng Webcam cục bộ, hãy thay địa chỉ RTSP bằng số `0` hoặc đường dẫn tới tệp video thử nghiệm)*.