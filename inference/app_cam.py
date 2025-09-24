from inference import get_model
import cv2

# Load model từ Roboflow
model = get_model(
    model_id="vietnamese-productions-classification/19",
    api_key="gMIGrMPVZUiwmUML8smO"
)

# Mở camera laptop (0 = camera mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chạy inference, confidence > 0.4
    results = model.infer(frame, confidence=0.4, overlap=0.5)[0]

    # Vẽ kết quả
    for pred in results.predictions:
        x, y = int(pred.x), int(pred.y)
        w, h = int(pred.width), int(pred.height)
        conf = float(pred.confidence)
        class_name = pred.class_name

        # Tính toạ độ bounding box
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Vẽ khung chữ nhật
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Vẽ nhãn + confidence
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("Camera Detections", frame)

    # Nhấn phím q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
