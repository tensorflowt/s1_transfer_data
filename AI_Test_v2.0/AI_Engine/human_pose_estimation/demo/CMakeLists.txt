set(target demo_alphapose_estimation) 
add_executable(${target} demo_alphapose_estimation.cpp) 
target_link_libraries(${target}  opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common human_pose_estimation object_detector)


set(target demo_poseestimation_image)
add_executable(${target} demo_poseestimation_image.cpp)
target_link_libraries(${target}  opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs sak_common human_pose_estimation)


set(target demo_alphapose_video)
add_executable(${target} demo_alphapose_video.cpp) 
target_link_libraries(${target}  opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio sak_common human_pose_estimation object_detector)
