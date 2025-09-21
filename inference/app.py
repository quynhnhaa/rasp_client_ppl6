from inference import get_model
import cv2
import os

# Đường dẫn ảnh
image_file = "/Users/quynhnhaa/Documents/Ahn/Study/Year4.1/PBL6/ImageZQ/Vietnamese Productions Classification/test/images/3029c2ea91aa921f78eb94422600891e_jpg.rf.e3a3ea412713fb17fbf26ce44013d2b4.jpg"
image = cv2.imread(image_file)

# Load model từ Roboflow
model = get_model(
    model_id="vietnamese-productions-classification/19",
    api_key="gMIGrMPVZUiwmUML8smO"
)

# Chạy inference, lọc confidence > 0.4
results = model.infer(image, confidence=0.4, overlap=0.5)[0]

# Vẽ kết quả lên ảnh
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
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Vẽ nhãn + confidence
    label = f"{class_name}: {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Hiển thị ảnh
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
