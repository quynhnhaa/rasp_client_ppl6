#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <csignal>
#include <atomic>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// NCNN
#include "net.h"

// ZeroMQ (cppzmq)
#include <zmq.hpp>

// Cấu hình
const std::string SERVER_IP = "192.168.1.10"; // <-- THAY ĐỔI IP SERVER CỦA BẠN
const int SERVER_PORT = 5555;
const std::string CAMERA_NAME = "raspi_cam_cpp";
const int CAMERA_WIDTH = 640;
const int CAMERA_HEIGHT = 640;
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float CONF_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;
const int JPEG_QUALITY = 30; // Chất lượng JPEG (0-100)
const int QUEUE_SIZE = 2; // Giới hạn kích thước queue để tiết kiệm RAM

const std::vector<std::string> CLASS_NAMES = {"product"};

// Hàng đợi an toàn cho đa luồng
template<typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() < max_size_) {
            queue_.push(std::move(value));
        }
        cond_.notify_one();
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    T wait_and_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{ return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

private:
    size_t max_size_ = QUEUE_SIZE;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};

// Biến toàn cục để xử lý tín hiệu dừng (Ctrl+C)
std::atomic<bool> stop_flag(false);

void signal_handler(int signum) {
    std::cout << "\nCaught signal " << signum << ". Shutting down..." << std::endl;
    stop_flag = true;
}

// Struct để lưu trữ kết quả nhận diện
struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

// Luồng 1: Lấy frame từ camera
void camera_worker(ThreadSafeQueue<cv::Mat>& frame_queue) {
    cv::VideoCapture cap;
    // Sử dụng backend V4L2 với libcamera
    cap.open(0, cv::CAP_V4L2); 
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open camera." << std::endl;
        stop_flag = true;
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    cap.set(cv::CAP_PROP_FPS, 30);

    std::cout << "[INFO] Camera thread started." << std::endl;
    while (!stop_flag) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "[WARN] Could not read frame from camera." << std::endl;
            continue;
        }
        // Picamera2 mặc định là BGR, không cần chuyển đổi
        frame_queue.push(frame);
    }
    cap.release();
    std::cout << "[INFO] Camera thread stopped." << std::endl;
}

// Luồng 2: Xử lý và nhận diện
void inference_worker(ThreadSafeQueue<cv::Mat>& frame_queue, ThreadSafeQueue<cv::Mat>& result_queue, ncnn::Net& net) {
    std::cout << "[INFO] Inference thread started." << std::endl;
    while (!stop_flag) {
        cv::Mat frame = frame_queue.wait_and_pop();
        if (frame.empty()) continue;

        int w = frame.cols;
        int h = frame.rows;

        // 1. Pre-processing
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, ncnn::Mat::PIXEL_BGR, w, h, INPUT_WIDTH, INPUT_HEIGHT);
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in.substract_mean_normalize(0, norm_vals);

        // 2. Inference
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);

        // 3. Post-processing
        std::vector<Detection> detections;
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        for (int i = 0; i < out.h; i++) {
            const float* values = out.row(i);
            float score = values[4]; // Chỉ có 1 class, score nằm ở index 4
            if (score > CONF_THRESHOLD) {
                float cx = values[0] * w;
                float cy = values[1] * h;
                float width = values[2] * w;
                float height = values[3] * h;

                int left = static_cast<int>(cx - width / 2);
                int top = static_cast<int>(cy - height / 2);
                
                boxes.push_back(cv::Rect(left, top, static_cast<int>(width), static_cast<int>(height)));
                scores.push_back(score);
                class_ids.push_back(0); // Chỉ có 1 class
            }
        }

        // Non-Maximum Suppression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

        for (int idx : indices) {
            Detection det;
            det.box = boxes[idx];
            det.score = scores[idx];
            det.class_id = class_ids[idx];
            detections.push_back(det);
        }

        // 4. Vẽ kết quả lên frame
        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = CLASS_NAMES[det.class_id] + ": " + cv::format("%.2f", det.score);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
            cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        result_queue.push(frame);
    }
    std::cout << "[INFO] Inference thread stopped." << std::endl;
}

// Luồng 3: Gửi dữ liệu qua mạng
void sender_worker(ThreadSafeQueue<cv::Mat>& result_queue) {
    zmq::context_t context(1);
    zmq::socket_t sender(context, zmq::socket_type::req);
    
    std::string server_address = "tcp://" + SERVER_IP + ":" + std::to_string(SERVER_PORT);
    std::cout << "[INFO] Connecting to server at " << server_address << "..." << std::endl;
    
    try {
        sender.connect(server_address);
    } catch (const zmq::error_t& e) {
        std::cerr << "[ERROR] ZeroMQ connect error: " << e.what() << std::endl;
        stop_flag = true;
        return;
    }
    
    std::cout << "[INFO] Sender thread started." << std::endl;
    bool first_frame_sent = false;

    while (!stop_flag) {
        cv::Mat frame = result_queue.wait_and_pop();
        if (frame.empty()) continue;

        // Nén ảnh thành JPEG
        std::vector<uchar> jpg_buffer;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY};
        cv::imencode(".jpg", frame, jpg_buffer, params);

        // Gửi tên camera
        sender.send(zmq::buffer(CAMERA_NAME), zmq::send_flags::sndmore);
        // Gửi dữ liệu ảnh
        sender.send(zmq::buffer(jpg_buffer));

        // Chờ xác nhận từ server
        zmq::message_t reply;
        if(sender.recv(reply, zmq::recv_flags::none)){
            if (!first_frame_sent) {
                std::cout << "[INFO] Successfully sent first frame to server." << std::endl;
                first_frame_sent = true;
            }
        }
    }
    sender.close();
    context.close();
    std::cout << "[INFO] Sender thread stopped." << std::endl;
}

int main() {
    // Đăng ký xử lý tín hiệu Ctrl+C
    signal(SIGINT, signal_handler);

    // 1. Load NCNN model
    ncnn::Net net;
    if (net.load_param("no_mosaic_sgd_0284_ncnn_model/model.ncnn.param") != 0) {
        std::cerr << "[ERROR] Failed to load model.ncnn.param" << std::endl;
        return -1;
    }
    if (net.load_model("no_mosaic_sgd_0284_ncnn_model/model.ncnn.bin") != 0) {
        std::cerr << "[ERROR] Failed to load model.ncnn.bin" << std::endl;
        return -1;
    }
    std::cout << "[INFO] NCNN model loaded successfully." << std::endl;

    // 2. Tạo các hàng đợi
    ThreadSafeQueue<cv::Mat> frame_queue;
    ThreadSafeQueue<cv::Mat> result_queue;

    // 3. Khởi chạy các luồng
    std::thread t1(camera_worker, std::ref(frame_queue));
    std::thread t2(inference_worker, std::ref(frame_queue), std::ref(result_queue), std::ref(net));
    std::thread t3(sender_worker, std::ref(result_queue));

    // 4. Chờ các luồng kết thúc
    t1.join();
    t2.join();
    t3.join();

    std::cout << "[INFO] All threads finished. Exiting." << std::endl;
    return 0;
}
