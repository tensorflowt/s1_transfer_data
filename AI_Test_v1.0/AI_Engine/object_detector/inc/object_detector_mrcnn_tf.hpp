#ifndef __OBJECT_DETECTOR_MRCNN_TF_HPP__
#define __OBJECT_DETECTOR_MRCNN_TF_HPP__

#include "object_detector_tf.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TENSORFLOW
#include "tensorflow/core/public/session.h"

class ObjectDetectorMrcnnTF : public ObjectDetectorTF
{
public:
    ObjectDetectorMrcnnTF();
    virtual ~ObjectDetectorMrcnnTF();

public:
    virtual int predict();
    virtual int setImage(const cv::Mat &img, bool keep_aspect=false);

protected:
    int setTensor(tensorflow::Tensor &in_tensor);
};

#else

class ObjectDetectorMrcnnTF : public ObjectDetectorTF
{
public:
    ObjectDetectorMrcnnTF()=default;
    virtual ~ObjectDetectorMrcnnTF()=default;

public:
    virtual int predict(){ 
        LOG_ERROR("Please implement ObjectDetectorMrcnnTF");
        return -1; 
    }
    virtual int setImage(const cv::Mat &img, bool keep_aspect=false){
        LOG_ERROR("Please implement ObjectDetectorMrcnnTF");
        return -1; 
    }
};


#endif // ENABLE_TENSORFLOW

#endif // __OBJECT_DETECTOR_MRCNN_TF_HPP__
