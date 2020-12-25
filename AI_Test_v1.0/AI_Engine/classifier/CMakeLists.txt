
include_directories(./inc)
aux_source_directory(./src src_cpp)

set(target classifier)
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

if(${ENABLE_MXNET})
    set(dependencies ${dependencies} mxnet)
endif()

if(${ENABLE_DARKNET})
    set(dependencies ${dependencies} darknet)
endif()

target_link_libraries(${target} ${dependencies})


add_subdirectory(demo)

## Install the target
INSTALL(TARGETS ${target}  DESTINATION lib64)
