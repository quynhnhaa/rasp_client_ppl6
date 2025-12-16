import os
import time
import json
import socket
import numpy as np
import cv2
import ncnn
from picamera2 import Picamera2

# ================== CẤU HÌNH ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Tự động xác định đường dẫn model (thư mục model cùng cấp với file main.py)
MODEL_DIR = os.path.join(SCRIPT_DIR, "no_mosaic_sgd_0284_ncnn_model")
PARAM_PATH = os.path.join(MODEL_DIR, "model.ncnn.param") # Sửa tên file nếu cần
BIN_PATH   = os.path.join(MODEL_DIR, "model.ncnn.bin")   # Sửa tên file nếu cần
CLASSES_PATH = os.path.join(SCRIPT_DIR, "class_to_id.txt") # File chứa tên các lớp

# Tên blob input/output trong file .param (bắt buộc xem trong .param để sửa cho đúng)
INPUT_BLOB_NAME  = "in0"              # thường là 'images'
OUTPUT_BLOB_NAMES = ["out0"]          # hoặc ['output'], hoặc ['output0','output1','output2']

# Kích thước input khi export (imgsz=320 => 320x320)
INPUT_SIZE = (320, 320)

# Đọc tên lớp từ file
try:
    CLASS_NAMES = []
    with open(CLASSES_PATH, "r", encoding="utf-8") as f: # Đảm bảo đọc UTF-8
        for line in f:
            class_name = line.strip() # Xóa khoảng trắng và ký tự xuống dòng
            if class_name: # Chỉ thêm nếu dòng không rỗng
                CLASS_NAMES.append(class_name)

    NUM_CLASSES = len(CLASS_NAMES)
    print(f"[INFO] Đã đọc {NUM_CLASSES} lớp từ: {CLASSES_PATH}")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"[ERROR] Lỗi khi đọc file class '{CLASSES_PATH}': {e}. Sử dụng giá trị mặc định.")
    CLASS_NAMES = ["product"]
    NUM_CLASSES = 1

# Ngưỡng lọc
SCORE_THRESH = 0.25
NMS_THRESH   = 0.45

# Server nhận kết quả
SERVER_IP = "192.168.1.10" # SỬA IP CHO ĐÚNG
SERVER_PORT = 5001        # Port cho kết nối TCP
DEVICE_ID = "raspi-zero-2w-01"

# ================== YOLO11 + NCNN ==================

class YOLO11NCNN:
    def __init__(self,
                 param_path,
                 bin_path,
                 input_size=(320, 320),
                 num_classes=1,
                 score_thresh=0.25,
                 nms_thresh=0.45,
                 input_blob_name="images",
                 output_blob_names=None,
                 class_names=None):
        self.input_w, self.input_h = input_size
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.input_blob_name = input_blob_name
        self.output_blob_names = output_blob_names or ["output0"]
        self.class_names = class_names or []

        self.net = ncnn.Net()
        opt = self.net.opt
        opt.use_vulkan_compute = False  # Pi Zero 2 W không có Vulkan
        opt.num_threads = 2             # 2–3 thread là hợp lý

        if self.net.load_param(param_path):
            raise RuntimeError(f"load_param failed: {param_path}")
        if self.net.load_model(bin_path):
            raise RuntimeError(f"load_model failed: {bin_path}")

    def _letterbox(self, frame_bgr):
        """Resize + pad (letterbox) -> RGB [H,W,3], đồng thời trả về scale, pad_x, pad_y."""
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        in_w, in_h = self.input_w, self.input_h

        scale = min(in_w / w0, in_h / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)

        dx = (in_w - nw) // 2
        dy = (in_h - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw, :] = resized

        return canvas, scale, dx, dy, w0, h0

    @staticmethod
    def _iou(box1, box2):
        """Tính IoU giữa 2 box [x1,y1,x2,y2] (trên cùng hệ tọa độ)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h

        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

        if area1 + area2 - inter <= 0:
            return 0.0
        return inter / (area1 + area2 - inter)

    def _nms(self, dets):
        """
        dets: list [x1,y1,x2,y2,score,cls_id] trong hệ toạ độ INPUT (sau letterbox).
        NMS theo từng class.
        """
        if not dets:
            return []

        # Tách theo class
        dets_by_cls = {}
        for d in dets:
            cls_id = d[5]
            dets_by_cls.setdefault(cls_id, []).append(d)

        final_dets = []
        for cls_id, dets_cls in dets_by_cls.items():
            dets_cls = sorted(dets_cls, key=lambda x: x[4], reverse=True)
            keep = []

            while dets_cls:
                best = dets_cls.pop(0)
                keep.append(best)
                dets_cls = [
                    d for d in dets_cls
                    if self._iou(best, d) < self.nms_thresh
                ]

            final_dets.extend(keep)

        return final_dets

    def detect(self, frame_bgr):
        """
        frame_bgr: numpy [H,W,3] BGR từ Picamera2
        return: list dicts {class_id, class_name, score, bbox:[x1,y1,x2,y2]} trên ảnh gốc.
        """
        img_rgb, scale, dx, dy, orig_w, orig_h = self._letterbox(frame_bgr)

        # Chuẩn bị input cho NCNN
        mat_in = ncnn.Mat.from_pixels(
            img_rgb, ncnn.Mat.PixelType.PIXEL_RGB,
            self.input_w, self.input_h
        )
        mean_vals = (0.0, 0.0, 0.0)
        norm_vals = (1/255.0, 1/255.0, 1/255.0)
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        ex = self.net.create_extractor()
        ex.input(self.input_blob_name, mat_in)

        all_preds = []

        # Lấy tất cả output blob, ghép lại
        for name in self.output_blob_names:
            out = ncnn.Mat()
            ret = ex.extract(name, out)
            if ret != 0:
                # Không tìm thấy blob này, bỏ qua
                continue

            out_np = np.array(out)

            # Tổng số giá trị của out
            total_vals = out_np.size
            values_per_det = 4 + self.num_classes  # [x,y,w,h] + scores cho num_classes

            if total_vals % values_per_det != 0:
                # Shape không chia hết, có thể bạn set NUM_CLASSES sai
                print(f"[WARN] Output '{name}' size {total_vals} không chia hết cho 4+NUM_CLASSES={values_per_det}")
                continue

            preds = out_np.reshape(-1, values_per_det)  # (N, 4+num_classes)
            all_preds.append(preds)

        if not all_preds:
            return []

        all_preds = np.concatenate(all_preds, axis=0)  # (N_total, 4 + num_classes)

        # Decode: xywh + class_scores -> x1y1x2y2 + class_id + score (trong hệ letterbox 320x320)
        raw_dets = []
        for row in all_preds:
            x_c, y_c, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:]

            cls_id = int(np.argmax(class_scores))
            score = float(class_scores[cls_id])

            if score < self.score_thresh:
                continue

            # Chuyển xywh -> x1y1x2y2 trên ảnh letterbox (320x320)
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2

            raw_dets.append([x1, y1, x2, y2, score, cls_id])

        # NMS trên hệ toạ độ letterbox
        nms_dets = self._nms(raw_dets)

        results = []
        for x1, y1, x2, y2, score, cls_id in nms_dets:
            # Map từ letterbox về ảnh gốc
            x1 = (x1 - dx) / scale
            y1 = (y1 - dy) / scale
            x2 = (x2 - dx) / scale
            y2 = (y2 - dy) / scale

            # Clip vào khung ảnh
            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(0, min(orig_w - 1, x2))
            y2 = max(0, min(orig_h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            class_name = (
                self.class_names[cls_id]
                if 0 <= cls_id < len(self.class_names)
                else str(cls_id)
            )

            results.append({
                "class_id": cls_id,
                "class_name": class_name,
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        return results

# ================== GỬI DỮ LIỆU LÊN SERVER ==================

def send_detections_to_server(detections, server_ip, server_port):
    """
    detections: list các dict từ YOLO11NCNN.detect
    Gửi dữ liệu qua TCP socket.
    """
    payload = {
        "device_id": DEVICE_ID,
        "timestamp": time.time(),
        "detections": detections,
    }
    
    # Chỉ gửi nếu có detection
    if not detections:
        return

    try:
        # Tạo socket mới cho mỗi lần gửi để đơn giản hóa việc xử lý lỗi kết nối
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0) # Đặt timeout 2 giây
            s.connect((server_ip, server_port))
            
            # Chuyển payload thành JSON string, rồi encode sang bytes và thêm ký tự xuống dòng
            message = json.dumps(payload).encode('utf-8') + b'\n'
            s.sendall(message)
            # print(f"Sent {len(detections)} detections to server.") # Bỏ comment để debug
    except Exception as e:
        print(f"Send to server error: {e}")

# ================== MAIN: PICAMERA2 LOOP ==================

def main():
    yolo = YOLO11NCNN(
        PARAM_PATH,
        BIN_PATH,
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
        score_thresh=SCORE_THRESH,
        nms_thresh=NMS_THRESH,
        input_blob_name=INPUT_BLOB_NAME,
        output_blob_names=OUTPUT_BLOB_NAMES,
        class_names=CLASS_NAMES,
    )

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)  # chờ camera ổn định

    print("Start capturing & detecting... (Ctrl+C để dừng)")

    try:
        while True:
            t0 = time.time()
            frame = picam2.capture_array()  # numpy BGR
            detections = yolo.detect(frame)

            # In ra console để debug
            if detections:
                print(f"{len(detections)} detections:")
                for d in detections:
                    print(d)

            # Gửi lên server (có thể chỉ gửi khi có detection nếu muốn)
            send_detections_to_server(detections, SERVER_IP, SERVER_PORT)

            # Hạn chế tần số xử lý (ví dụ ~5 FPS)
            t1 = time.time()
            dt = t1 - t0
            if dt < 0.2:
                time.sleep(0.2 - dt)

    except KeyboardInterrupt:
        print("Stopped by user")

    picam2.stop()


if __name__ == "__main__":
    main()