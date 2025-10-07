import cv2
import time
import imagezmq
import numpy as np

# --- CẤU HÌNH ---
# Địa chỉ IP và Cổng của ImageZMQ Hub (FastAPI Server)
# Thay 'SERVER_IP' bằng địa chỉ IP thực tế của máy chủ FastAPI của bạn
SERVER_IP = '127.0.0.1' 
SERVER_PORT = 5555
SERVER_URL = f"tcp://{SERVER_IP}:{SERVER_PORT}"

# Tên client/thiết bị này (ví dụ: 'RPi_Camera_1')
SENDER_NAME = 'Client_PC'

# --- KHỞI TẠO ---
# 1. Khởi tạo kết nối ImageZMQ Sender
sender = imagezmq.ImageSender(connect_to=SERVER_URL)
print(f"Kết nối tới server: {SERVER_URL}")

# 2. Khởi tạo Camera (sử dụng camera mặc định số 0)
# Thêm camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) 
# và camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
# để tăng tốc độ nếu cần.
camera = cv2.VideoCapture(0)

# Kiểm tra nếu camera không mở được
if not camera.isOpened():
    print("Lỗi: Không thể truy cập camera.")
    exit()

# Thiết lập một số thuộc tính camera (tùy chọn)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

time.sleep(2.0) # Đợi camera ấm lên
print("Bắt đầu gửi luồng ảnh...")

try:
    while True:
        # 1. Chụp ảnh
        ret, frame = camera.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ camera.")
            break

        # 2. Xử lý ảnh (Tùy chọn: Thay đổi kích thước, chuyển xám, v.v.)
        # Ví dụ: Giảm kích thước ảnh để tăng tốc độ truyền
        # frame = cv2.resize(frame, (320, 240)) 
        
        # 3. Mã hóa ảnh sang định dạng JPEG
        # Việc gửi ảnh JPEG (nén) nhanh hơn nhiều so với gửi ảnh thô (uncompressed)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # Chất lượng 90%
        # Trả về: ret (true/false) và jpg_buffer (mảng byte)
        ret, jpg_buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Chuyển đổi mảng byte sang NumPy array (cho imagezmq)
        jpg_buffer_array = np.array(jpg_buffer).tobytes()

        # 4. Gửi ảnh và chờ phản hồi (Blocking call)
        # Gửi Tên client và Buffer ảnh (đã nén)
        reply = sender.send_jpg(SENDER_NAME, jpg_buffer_array)
        
        # 5. Xử lý phản hồi từ server (ví dụ: 'OK')
        # print(f"Server phản hồi: {reply.decode('utf-8')}")

        # Tùy chọn: Hiển thị ảnh tại client (Debug)
        cv2.imshow('Image Client', frame)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("\nĐã dừng client.")
    
finally:
    # 6. Dọn dẹp
    camera.release()
    cv2.destroyAllWindows()
    sender.close()
    print("Client đã đóng kết nối.")