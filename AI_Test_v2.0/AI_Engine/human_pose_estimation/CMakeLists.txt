include_directories(./inc)
aux_source_directory(./src src_cpp)

set(target human_pose_estimation)
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

target_link_libraries(${target} ${dependencies})


add_subdirectory(demo)

## Install the target
INSTALL(TARGETS ${target}  DESTINATION lib64)