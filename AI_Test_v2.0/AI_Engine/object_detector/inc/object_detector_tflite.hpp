#ifndef __OBJECT_DETECTOR_TFLITE_HPP__
#define __OBJECT_DETECTOR_TFLITE_HPP__

#include "object_detector.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TFLITE
#include "tensorflow/lite/kernels/register.h"

class ObjectDetectorTflite : public ObjectDetector
{
public:
    ObjectDetectorTflite();
    virtual ~ObjectDetectorTflite();

public:
    virtual int init(const ODParam &param);
    virtual int predict();

private:
    int setTensor();
    
private:
    std::unique_ptr<tflite::Interpreter> m_interpreter;
    // model must be a member variable, otherwise it would be released when call invoke() in other functions
    std::unique_ptr<tflite::FlatBufferModel> m_model;
};

#else

class ObjectDetectorTflite : public ObjectDetector
{
public:
    ObjectDetectorTflite()=default;
    virtual ~ObjectDetectorTflite()=default;

public:
    virtual int init(const ODParam &param){ 
        LOG_ERROR("Please implement ObjectDetectorTflite");
        return -1; 
    }
    virtual int predict(){ 
        LOG_ERROR("Please implement ObjectDetectorTflite");
        return -1; 
    };
};

#endif // ENABLE_TENSORFLOW

#endif // __OBJECT_DETECTOR_TFLITE_HPP__
