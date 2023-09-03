cmake_minimum_required(VERSION 3.17)
project(test_tflite)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(OpenCV REQUIRED core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS = ${OpenCV_LIBS}")

include_directories(${CMAKE_CURRENT_LIST_DIR}/tflite-2.12.0/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/tflite-2.12.0/include/drwav)
include_directories(${CMAKE_CURRENT_LIST_DIR}/tflite-2.12.0/include/abseil-cpp-20220623.1)
include_directories(${CMAKE_CURRENT_LIST_DIR}/tflite-2.12.0/include/flatbuffers-22.10.26/include)

# 获取 macOS 版本号
execute_process(COMMAND sw_vers -productVersion OUTPUT_VARIABLE macOS_VERSION)
string(STRIP ${macOS_VERSION} macOS_VERSION)
# 提取前两位数字
string(REGEX MATCH "^[0-9]+" macOS_VERSION_MAJOR "${macOS_VERSION}")

message("macOS_VERSION = ${macOS_VERSION}")
message("macOS_VERSION_MAJOR = ${macOS_VERSION_MAJOR}")

add_library(tflite SHARED IMPORTED)
set_target_properties(tflite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/tflite-2.12.0/tflite2.12.0_lib/mac-os/os${macOS_VERSION_MAJOR}/libtensorflowlite.dylib)
#add_library(tflite_gpu SHARED IMPORTED)
#set_target_properties(tflite_gpu PROPERTIES IMPORTED_LOCATION
#        /Users/yang/opt/tensorflow/bazel-out/applebin_macos-darwin_x86_64-opt-ST-4d5b244fa833/bin/tensorflow/lite/delegates/gpu/tensorflow_lite_gpu_dylib.dylib)
#link_directories(${CMAKE_CURRENT_LIST_DIR}/tflite-2.10.0/tflite2.10.0_lib/mac-os/gpu)

add_executable(test_tflite main.cpp)

target_link_libraries(test_tflite ${OpenCV_LIBS} tflite)
#target_link_libraries(test_tflite ${OpenCV_LIBS} tflite tflite_gpu)