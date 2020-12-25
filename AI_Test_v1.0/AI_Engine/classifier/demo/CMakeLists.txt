
set(target demo_classifier_image)
add_executable(${target} demo_classifier_image.cpp)
target_link_libraries(${target} classifier opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common)

set(target demo_classifier_image_list)
add_executable(${target} demo_classifier_image_list.cpp)
target_link_libraries(${target} classifier opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common)

set(target demo_extract_feature_image)
add_executable(${target} demo_extract_feature_image.cpp)
target_link_libraries(${target} classifier opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common)
