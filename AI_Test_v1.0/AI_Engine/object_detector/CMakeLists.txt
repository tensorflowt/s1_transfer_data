include_directories(./inc)
aux_source_directory(./src src_cpp)

set(OpenCV_DIR D:\\opencv_4.3\\source\\opencv\\build)

find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)
## For opencv
#include_directories(D:\\project\\AI_Test\\AI_3rdparty\\x86_64\\opencv\\build\\include\\
#                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include"
#                    "D:/project/AI_Test/AI_3rdparty/x86_64/boost_1_70_0/include")
#link_directories(D:/project/AI_Test/AI_3rdparty/x86_64/opencv/build/x64/vc15/lib/
#                 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64"
#                 "D:/project/AI_Test/AI_3rdparty/x86_64/boost_1_70_0/lib")
#link_directories(D:/project/AI_Test/build/lib/Debug)
				 
set(target object_detector)
add_library(${target} SHARED ${src_cpp})

set(dependencies opencv_core opencv_highgui opencv_imgproc sak_common)


				 
if(${ENABLE_TORCH})
    set(dependencies ${dependencies} torch)
endif()

if(${ENABLE_TENSORFLOW})
    set(dependencies ${dependencies} tensorflow_cc tensorflow_framework)
endif()

if(${ENABLE_TFLITE})
    set(dependencies ${dependencies} tensorflowlite)
endif()

if(${ENABLE_DARKNET})
    set(dependencies ${dependencies} darknet)
endif()

if(${ENABLE_TENSORRTX})
    message("build with tensorrt.")
    add_definitions(-DENABLE_TRT)
    find_package(CUDA REQUIRED)
    include_directories(${PROJECT_SOURCE_DIR}/yolov5/)
    link_directories(${PROJECT_SOURCE_DIR}/yolov5/build)
    set(dependencies ${dependencies} cudart nvinfer yololayer)
	add_subdirectory(yolov5)
endif()

target_link_libraries(${target} ${dependencies})

add_subdirectory(demo)

## Install the target
INSTALL(TARGETS ${target}  DESTINATION lib64)
