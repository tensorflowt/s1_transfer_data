
set(target demo_classifier_video)
add_executable(${target} demo_classifier_video.cpp)
target_link_libraries(${target} classifier_video opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common opencv_video opencv_videoio)
