import socket
import json
import time

# ================== CẤU HÌNH ==================
HOST = '0.0.0.0'  # Lắng nghe trên tất cả các interface mạng
PORT = 5001       # Port phải khớp với SERVER_PORT ở client (main.py)

def handle_client(conn, addr):
    """
    Xử lý kết nối từ một client.
    Nhận, parse và in dữ liệu JSON.
    """
    print(f"[INFO] Đã kết nối bởi {addr}")
    
    # Buffer để lưu dữ liệu nhận được từ client
    buffer = b""
    try:
        while True:
            # Nhận dữ liệu từ client, mỗi lần 1024 bytes
            data = conn.recv(1024)
            if not data:
                # Nếu không nhận được dữ liệu, client đã đóng kết nối
                break
            
            buffer += data
            
            # Client gửi mỗi JSON payload kết thúc bằng một ký tự xuống dòng ('\n')
            # Ta xử lý buffer nếu tìm thấy ký tự này
            if b'\n' in buffer:
                # Tách message và phần còn lại trong buffer
                message, buffer = buffer.split(b'\n', 1)
                
                try:
                    # Decode chuỗi bytes thành string UTF-8
                    payload_str = message.decode('utf-8')
                    # Parse chuỗi JSON thành dictionary
                    payload = json.loads(payload_str)
                    
                    # Xử lý dữ liệu (ở đây chỉ in ra console)
                    device_id = payload.get("device_id", "N/A")
                    timestamp = payload.get("timestamp", 0)
                    detections = payload.get("detections", [])
                    
                    print("-" * 50)
                    print(f"Time: {time.ctime(timestamp)} | Device: {device_id}")
                    if detections:
                        print(f"Found {len(detections)} objects:")
                        for i, det in enumerate(detections):
                            class_name = det.get('class_name', 'unknown')
                            score = det.get('score', 0)
                            print(f"  {i+1}. Class: {class_name}, Score: {score:.2f}")
                    else:
                        print("No objects detected.")
                    print("-" * 50)

                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"[ERROR] Lỗi khi xử lý dữ liệu từ {addr}: {e}")
    finally:
        print(f"[INFO] Đóng kết nối từ {addr}")
        conn.close()

def main():
    """Hàm chính để khởi chạy server."""
    # Tạo socket TCP/IP (AF_INET cho IPv4, SOCK_STREAM cho TCP)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Cho phép tái sử dụng địa chỉ ngay lập tức để tránh lỗi "Address already in use"
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"[INFO] Server TCP đang lắng nghe trên {HOST}:{PORT}...")
        
        while True:
            conn, addr = server_socket.accept()
            handle_client(conn, addr) # Xử lý mỗi client một cách tuần tự

if __name__ == "__main__":
    main()