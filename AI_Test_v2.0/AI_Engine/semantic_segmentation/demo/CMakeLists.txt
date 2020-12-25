
set(target demo_segmentation_image)
add_executable(${target} demo_segmentation_image.cpp)
target_link_libraries(${target}  opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common semantic_segmentation)


