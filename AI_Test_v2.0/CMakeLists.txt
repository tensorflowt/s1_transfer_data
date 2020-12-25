cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_MACOSX_RPATH 0)

project(AI_Test)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-msse4.2 -fPIC -Ofast -O3")
add_definitions(-std=c++11)

set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)


set(CMAKE_CUDA_COMPILER "C/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin/nvcc")

## For opencv
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/opencv/build/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/boost_1_70_0/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/opencv/build/x64/vc15/lib
                 ${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/boost_1_70_0/lib)

## For logging infor
option(ENABLE_LOG_DEBUG "Enbale log debug message" OFF)
option(ENABLE_LOG_INFO "Enbale log info message" ON)
option(ENABLE_LOG_ERROR "Enbale log error message" ON)

if(${ENABLE_LOG_DEBUG})
    add_definitions(-DENABLE_LOG_DEBUG)
endif()
if(${ENABLE_LOG_INFO})
    add_definitions(-DENABLE_LOG_INFO)
endif()
if(${ENABLE_LOG_ERROR})
    add_definitions(-DENABLE_LOG_ERROR)
endif()

## For dnn inference engine
option(ENABLE_AI_ENGINE_OBJECT_DETECTOR "Enbale object detector" ON)
option(ENABLE_AI_ENGINE_CLASSIFIER "Enbale classifier" OFF)
option(ENABLE_AI_ENGINE_CLASSIFIER_VID "Enbale classifier video" OFF)
option(ENABLE_AI_ENGINE_SEMANTIC_SEGMENTATION "Enbale semantic segmentation" OFF)
option(ENABLE_AI_ENGINE_MOT_TRACKER "Enbale mot tracker" OFF)

option(ENABLE_DARKNET "Enbale darknet" OFF)
option(ENABLE_TFLITE "Enbale Tflite" OFF)
option(ENABLE_TENSORFLOW "Enbale tensorflow" OFF)
option(ENABLE_MXNET "Enbale mxnet" OFF)
option(ENABLE_TORCH "Enbale torch" OFF)
option(ENABLE_TENSORRTX "Enbale tensorrtx" ON)

message("================= Config =================")
link_directories("C/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64")

if(${ENABLE_TENSORFLOW})
    message("-- Enable Tensorflow")
    add_definitions(-DENABLE_TENSORFLOW)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/tensorflow-2.0/include/
                        ${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/tensorflow-2.0/include/eigen
                        ${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/tensorflow-2.0/include/absl
                        )
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/tensorflow-2.0/lib)
endif()

if(${ENABLE_MXNET})
    message("-- Enable MXNet")
    add_definitions(-DENABLE_MXNET)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/mxnet/include)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/mxnet/lib)
endif()

if(${ENABLE_TORCH})
    message("-- Enable Torch")
    add_definitions(-DENABLE_TORCH)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/libtorch/include/)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/libtorch/include/torch/csrc/api/include)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/libtorch/lib)
endif()

if(${ENABLE_TFLITE})
    message("-- Enable Tflite")
    add_definitions(-DENABLE_TFLITE)
    include_directories(/Users/shenfei/sandbox/software/tensorflow_1.14/
                        /Users/shenfei//sandbox/software/tensorflow_1.14/tensorflow/contrib/makefile/downloads/flatbuffers/include/)
    link_directories(/Users/shenfei/sandbox/software/tensorflow_1.14/bazel-bin/tensorflow/lite)
endif()

if(${ENABLE_DARKNET})
    message("-- Enable darknet")
    add_definitions(-DENABLE_DARKNET)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/darknet/include
                        /usr/local/cuda-10.0/include)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/darknet/lib)
endif()

if(${ENABLE_TENSORRTX})
    message("-- Enable Tensorrtx")
    add_definitions(-DENABLE_TENSORRTX)
	find_package(CUDA REQUIRED)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/tensorrt-7.0.0.11/include
                        C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/AI_3rdparty/x86_64/tensorrt-7.0.0.11/lib)
	set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_30;code=sm_30)
	enable_language(CUDA) 
  
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")
	cuda_add_library(yololayer SHARED ${CMAKE_CURRENT_SOURCE_DIR}/AI_Engine/object_detector/src/yololayer.cu ${CMAKE_CURRENT_SOURCE_DIR}/AI_Engine/object_detector/inc/yololayer.h)
    target_link_libraries(yololayer "nvinfer" "cudart")
  
endif()

message("================= Config =================")

add_subdirectory(AI_Engine)
message(${AI_ENGINE_INC})        
include_directories(${AI_ENGINE_INC})
