cmake_minimum_required(VERSION 3.16) # 他妈的问题出在我只有/usr/share/cmake-3.16/Modules/FindCUDA.cmake，只能指定3.16.其他都错
project(light_glue_onnx  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions(-DRELEASE)
endif()

# SPDLOG 在编译时会有一些警告，这里忽略掉
add_definitions(-Wno-deprecated-declarations)

find_package(CUDA REQUIRED)
set(ONNXRUNTIME_ROOTDIR thirdparty/onnxruntime-linux-x64-gpu-1.16.3)
message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")
include_directories(${ONNXRUNTIME_ROOTDIR}/include)
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

set(OpenCV_CUDA_MODULE_PATH "")
find_package(OpenCV REQUIRED QUIET)
message(Opencv version: ${OpenCV_VERSION})
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(
${CMAKE_SOURCE_DIR}/include
${CMAKE_SOURCE_DIR}/thirdparty
)

# logger lib
add_library(logger src/logger/logger.cpp)
target_link_libraries(logger  )

add_library(image_process src/image_process.cpp)
target_link_libraries(image_process ${OpenCV_LIBS} logger  )

add_library(combine_onnx_runner src/combined_onnx_runner.cpp)
target_link_libraries(combine_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger  )

add_library(decoupled_onnx_runner src/decoupled_onnx_runner.cpp)
target_link_libraries(decoupled_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger  )

add_library(extractor src/extractor/extractor.cpp)
target_link_libraries(extractor onnxruntime image_process ${OpenCV_LIBS} logger  )

add_library(matcher src/matcher/matcher.cpp)
target_link_libraries(matcher onnxruntime image_process ${OpenCV_LIBS} logger  )

add_executable(onnx_test node/onnx_test.cpp)
target_link_libraries(onnx_test onnxruntime logger  )

add_executable(combined_test node/combined_test.cpp)
target_link_libraries(combined_test combine_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger  )

add_executable(decoupled_test node/decoupled_test.cpp)
target_link_libraries(decoupled_test decoupled_onnx_runner onnxruntime image_process ${OpenCV_LIBS} logger  )

add_executable(extractor_test node/extractor_test.cpp)
target_link_libraries(extractor_test extractor onnxruntime image_process ${OpenCV_LIBS} logger  pthread)

add_executable(matcher_test node/matcher_test.cpp)
target_link_libraries(matcher_test matcher extractor onnxruntime image_process ${OpenCV_LIBS} logger  pthread)

################### Unit Test ##############################
# # Find Google Test
# find_package(GTest REQUIRED)

# # Unit Test
# file(GLOB_RECURSE TEST_SOURCES node/*_test.cpp)
# add_executable(unit_test ${TEST_SOURCES})
# target_link_libraries(unit_test GTest::GTest GTest::Main)
#############################################################