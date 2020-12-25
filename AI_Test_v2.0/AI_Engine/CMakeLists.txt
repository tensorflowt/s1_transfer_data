## For opencv
include_directories(D:/Project/AI_Test/AI_3rdparty/x86_64/opencv/include/
                    D:/Project/AI_Test/AI_3rdparty/x86_64/boost_1_70_0/include)
link_directories(D:/Project/AI_Test/AI_3rdparty/x86_64/opencv/lib/
                 D:/Project/AI_Test/AI_3rdparty/x86_64/boost_1_70_0/lib)
				 
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/ai_engine)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/object_detector/inc 
                ${CMAKE_CURRENT_SOURCE_DIR}/semantic_segmentation/inc 
                ${CMAKE_CURRENT_SOURCE_DIR}/classifier/inc 
                ${CMAKE_CURRENT_SOURCE_DIR}/classifier_video/inc 
                ${CMAKE_CURRENT_SOURCE_DIR}/mot_tracker/inc 
                ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
                ${CMAKE_CURRENT_SOURCE_DIR}/human_pose_estimation/inc 
                )
message(include_dir)   
set(AI_ENGINE_INC ${include_dir} PARENT_SCOPE)
include_directories(${include_dir})

## For dnn inference engine
add_subdirectory(common)
if(ENABLE_AI_ENGINE_CLASSIFIER)
    add_subdirectory(classifier)
endif()

if(ENABLE_AI_ENGINE_CLASSIFIER_VID)
    add_subdirectory(classifier_video)
endif()

if(ENABLE_AI_ENGINE_OBJECT_DETECTOR)
    add_subdirectory(object_detector)
endif()

if(ENABLE_AI_ENGINE_SEMANTIC_SEGMENTATION)
    add_subdirectory(semantic_segmentation)
endif()

if(ENABLE_AI_ENGINE_MOT_TRACKER)
    add_subdirectory(mot_tracker)
endif()

if(ENABLE_AI_ENGINE_HUMAN_POSE_ESTIMATION)
    add_subdirectory(human_pose_estimation)
endif()  
