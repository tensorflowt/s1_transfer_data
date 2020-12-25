#ifndef __OBJECT_DETECTOR_TF_HPP__
#define __OBJECT_DETECTOR_TF_HPP__

#include "object_detector.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TENSORFLOW
#include "tensorflow/core/public/session.h"

class ObjectDetectorTF : public ObjectDetector
{
public:
    ObjectDetectorTF();
    virtual ~ObjectDetectorTF();

public:
    virtual int init(const ODParam &param);
    virtual int predict();

protected:
    int setTensor(tensorflow::Tensor &in_tensor);
    
protected:
    std::unique_ptr<tensorflow::Session> m_session;
};

#else

class ObjectDetectorTF : public ObjectDetector
{
public:
    ObjectDetectorTF()=default;
    virtual ~ObjectDetectorTF()=default;

public:
    virtual int init(const ODParam &param){ 
        LOG_ERROR("Please implement ObjectDetectorTF");
        return -1; 
    }
    virtual int predict(){ 
        LOG_ERROR("Please implement ObjectDetectorTF");
        return -1; 
    }
};

#endif // ENABLE_TENSORFLOW

#endif // __OBJECT_DETECTOR_TF_HPP__
