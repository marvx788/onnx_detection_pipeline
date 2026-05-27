cmake_minimum_required(VERSION 3.10)
project(onnx_detection_gpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Check for ORT_ROOT defined in Dockerfile
if(NOT DEFINED ENV{ORT_ROOT})
    message(FATAL_ERROR "ORT_ROOT environment variable not set. Please set it to the ONNX Runtime installation directory.")
endif()

set(ORT_ROOT $ENV{ORT_ROOT})

# ONNX Runtime headers and library
include_directories(${ORT_ROOT}/include)
link_directories(${ORT_ROOT}/lib)

# Create executable
add_executable(yolo_onnx_app main2.cpp)

# Link libraries
target_link_libraries(yolo_onnx_app
    ${OpenCV_LIBS}
    onnxruntime
)
