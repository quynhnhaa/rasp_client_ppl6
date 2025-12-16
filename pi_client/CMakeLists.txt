cmake_minimum_required(VERSION 3.10)
project(pi_client_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tìm OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Tìm NCNN (Giả sử bạn đã cài đặt hoặc có sẵn trong một thư mục)
# Bạn cần thay đổi đường dẫn này tới thư mục build của NCNN
set(NCNN_DIR "/path/to/your/ncnn/build/install/lib/cmake/ncnn") 
find_package(ncnn REQUIRED)
include_directories(${ncnn_INCLUDE_DIRS})

# Tìm ZeroMQ
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)
include_directories(${ZMQ_INCLUDE_DIRS})

# Thêm thư viện cppzmq (header-only)
# Bạn cần tải file zmq.hpp và đặt nó ở nơi trình biên dịch có thể tìm thấy
# Ví dụ: /usr/local/include hoặc trong thư mục dự án
# include_directories(/path/to/cppzmq/include)

add_executable(main main.cpp)

target_link_libraries(main
    ${OpenCV_LIBS}
    ${ncnn_LIBS}
    ${ZMQ_LIBRARIES}
    pthread # Cần cho std::thread
)
