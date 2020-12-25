
set(target demo_tf_od_image)
add_executable(${target} demo_tf_od_image.cpp)
target_link_libraries(${target} object_detector opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common)

set(target demo_tf_od_list)
add_executable(${target} demo_tf_od_list.cpp)
target_link_libraries(${target} object_detector opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common)

set(target demo_tf_od_image_batch_effdet)
add_executable(${target} demo_tf_od_image_batch_effdet.cpp)
target_link_libraries(${target} object_detector opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common)

if (${ENABLE_TENSORRTX})
    set(target demo_trtx_od_image_yolov5)
    add_executable(${target} demo_trtx_od_image_yolov5.cpp)
    target_link_libraries(${target} object_detector opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common)
endif()

if (${ENABLE_DARKNET})
    set(target demo_darknet)
    add_executable(${target} demo_darknet.cpp)
    target_link_libraries(${target} darknet opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common)
endif()
