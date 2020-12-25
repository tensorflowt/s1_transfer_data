include_directories(./inc)
include_directories(./src)
aux_source_directory(./src src_cpp)

set(target mot_tracker)
add_library(${target} SHARED ${src_cpp})

target_link_libraries(${target}  opencv_core opencv_highgui opencv_imgproc opencv_video sak_common object_detector)

## Install the target
INSTALL(TARGETS ${target}  DESTINATION lib64)
