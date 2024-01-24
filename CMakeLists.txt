cmake_minimum_required(VERSION 3.16) # 他妈的问题出在我只有/usr/share/cmake-3.16/Modules/FindCUDA.cmake，只能指定3.16.其他都错
project(light_glue_onnx  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)


# set(ONNXRUNTIME_ROOTDIR /home/lin/Projects/onnxruntime)
# message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")
# include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
# link_directories("/home/lin/Projects/onnxruntime/build/Linux/Release")


# 查找 CUDA
# find_package(CUDA)

# if(CUDA_FOUND)
#     message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
#     # 如果找到 CUDA，使用 GPU 版本的 ONNX Runtime
#     set(ONNXRUNTIME_ROOTDIR /home/lin/Projects/onnxruntime-linux-x64-gpu-1.16.3)
#     message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")
#     include_directories(${ONNXRUNTIME_ROOTDIR}/include)
#     link_directories(${ONNXRUNTIME_ROOTDIR}/lib)
# else()
#     # 如果没有找到 CUDA，使用 CPU 版本的 ONNX Runtime
#     set(ONNXRUNTIME_ROOTDIR /home/lin/Projects/onnxruntime-linux-x64-1.16.3)
#     message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")
#     include_directories(${ONNXRUNTIME_ROOTDIR}/include)
#     link_directories(${ONNXRUNTIME_ROOTDIR}/lib)
# endif()




find_package(CUDA REQUIRED)
set(ONNXRUNTIME_ROOTDIR /home/lin/Projects/onnxruntime-linux-x64-gpu-1.16.3)
message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")
include_directories(${ONNXRUNTIME_ROOTDIR}/include)
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)

find_package(spdlog REQUIRED)
message(spdlog_VERSION: ${spdlog_VERSION})

set(OpenCV_CUDA_MODULE_PATH "")
find_package(OpenCV REQUIRED QUIET)
message(Opencv version: ${OpenCV_VERSION})
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)

# logger lib
add_library(logger src/log/logger.cpp)
target_link_libraries(logger spdlog::spdlog)

add_library(image_process src/image_process.cpp)
target_link_libraries(image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(combine_onnx_runner src/combined_onnx_runner.cpp)
target_link_libraries(combine_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_library(decoupled_onnx_runner src/decoupled_onnx_runner.cpp)
target_link_libraries(decoupled_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)


add_executable(onnx_test test/onnx_test.cpp)
target_link_libraries(onnx_test onnxruntime logger spdlog::spdlog)

add_executable(combined_test src/combined_test.cpp)
target_link_libraries(combined_test combine_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)

add_executable(decoupled_test src/decoupled_test.cpp)
target_link_libraries(decoupled_test decoupled_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger spdlog::spdlog)