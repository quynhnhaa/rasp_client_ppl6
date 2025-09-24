from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import sys
from multiprocessing import Process, Queue

def camera_process(queue: Queue):
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 640)}
    )
    picam2.configure(config)
    picam2.start()

    while True:
        frame = picam2.capture_array()
        if not queue.full():
            queue.put(frame)

def inference_process(queue: Queue, model_path: str):
    model = YOLO(model_path)

    while True:
        if not queue.empty():
            frame = queue.get()
            results = model.predict(
                source=frame, show=False, save=False, verbose=False
            )
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv11 Real-time", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yolo11.py <model.pt>")
        sys.exit(1)

    model_path = sys.argv[1]
    
    frame_queue = Queue(maxsize=2)

    
    p1 = Process(target=camera_process, args=(frame_queue,))
    p2 = Process(target=inference_process, args=(frame_queue, model_path))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
