# Hệ thống Camera với LCD Display và MQTT

Hệ thống này bao gồm:
- **Server** (chạy trên MacBook): Nhận frame từ camera, xử lý và gửi thông tin sản phẩm qua MQTT
- **Client** (chạy trên Raspberry Pi): Gửi frame lên server và hiển thị thông tin sản phẩm lên LCD

## 📋 Yêu cầu hệ thống

### Hardware:
- Raspberry Pi với camera
- LCD 16x2 với giao tiếp I2C (địa chỉ 0x27)
- MacBook/Server trong cùng mạng

### Software:
- Python 3.7+
- Các package trong `requirements.txt`

## 🚀 Cài đặt và Chạy

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình Mạng

Đảm bảo Raspberry Pi và MacBook ở cùng mạng WiFi và có thể ping nhau:

```bash
# Trên Raspberry Pi, kiểm tra kết nối tới MacBook
ping 172.20.10.12

# Trên MacBook, kiểm tra kết nối tới Raspberry Pi
ping [RASPBERRY_PI_IP]
```

### 3. Chạy Server (trên MacBook)

```bash
cd client_lcd
python serverpl.py
```

Server sẽ:
- **Chạy MQTT Broker** trên localhost:1883
- Lắng nghe frame từ client trên port 5555
- Gửi thông tin sản phẩm ngẫu nhiên qua MQTT mỗi 3-8 giây
- Hiển thị frame nhận được

### 4. Chạy Client (trên Raspberry Pi)

```bash
python client_lcd/clientpl.py
```

Client sẽ:
- Kết nối tới server và gửi frame
- Nhận thông tin sản phẩm qua MQTT
- Hiển thị thông tin lên LCD:
  - Dòng 1: Tên sản phẩm
  - Dòng 2: Tổng tiền (VND)

## ⚙️ Cấu hình

### Biến môi trường (tùy chọn):

```bash
# MQTT Broker - Server dùng localhost, Client dùng IP của MacBook
# Server (MacBook):
export MQTT_BROKER="localhost"
# Client (Raspberry Pi):
export MQTT_BROKER="172.20.10.12"

# MQTT Port (mặc định 1883)
export MQTT_PORT=1883

# MQTT Topic (mặc định "product/info")
export MQTT_TOPIC="product/info"

# Server IP (cho client kết nối gửi frame)
export server_ip="172.20.10.12"

# Server Port (mặc định 5555)
export port=5555
```

## 📁 Cấu trúc File

```
rasp_client_ppl6/
├── requirements.txt          # Dependencies
├── client_lcd/
│   ├── serverpl.py          # Server xử lý frame và gửi MQTT
│   └── clientpl.py          # Client gửi frame và hiển thị LCD
└── README.md                # Tài liệu hướng dẫn
```

## 🔧 Khắc phục sự cố

### 1. LCD không hiển thị:
- Kiểm tra kết nối I2C: `i2cdetect -y 1`
- Đảm bảo địa chỉ LCD là 0x27
- Kiểm tra kết nối dây

### 2. MQTT không kết nối:
- Đảm bảo firewall không chặn port 1883
- Kiểm tra IP broker có đúng không
- Thử ping broker từ client

### 3. Camera không hoạt động:
- Kiểm tra camera được kết nối đúng
- Đảm bảo Picamera2 được cài đặt

### 4. ImageZMQ không kết nối:
- Kiểm tra cùng mạng WiFi
- Đảm bảo port 5555 không bị chặn
- Kiểm tra IP server có đúng không

## 📊 Chức năng hoạt động

1. **Client** (Raspberry Pi) chụp frame từ camera và gửi lên **Server** (MacBook)
2. **Server** nhận frame và giả lập xử lý AI (có/không có sản phẩm)
3. **Server** định kỳ gửi thông tin sản phẩm ngẫu nhiên qua MQTT
4. **Client** nhận thông tin qua MQTT và hiển thị lên LCD

## 🎯 Tùy chỉnh

### Thêm sản phẩm mới:
Sửa danh sách `PRODUCTS` trong `client_lcd/serverpl.py`:

```python
PRODUCTS = [
    "iPhone 15 Pro Max",
    "Samsung Galaxy S24",
    # Thêm sản phẩm mới vào đây
]
```

### Thay đổi khoảng thời gian gửi MQTT:
Sửa `random.randint(3, 8)` trong hàm `mqtt_sender_thread()`

### Thay đổi định dạng hiển thị LCD:
Sửa hàm `display_on_lcd()` trong `client_lcd/clientpl.py`

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Kết nối mạng giữa các thiết bị
2. Cấu hình IP đúng
3. Các port không bị chặn
4. Dependencies đã cài đầy đủ
