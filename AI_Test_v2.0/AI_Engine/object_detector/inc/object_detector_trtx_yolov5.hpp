#ifndef __OBJECT_DETECTOR_TRTX_YOLOV5_HPP__
#define __OBJECT_DETECTOR_TRTX_YOLOV5_HPP__

#include "object_detector.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TENSORRTX
#include "NvInfer.h"
#include "logging.h"

class __declspec(dllexport) ObjectDetectorTRTX_YOLOV5 : public ObjectDetector
//class ObjectDetectorTRTX_YOLOV5 : public ObjectDetector
{
public:
    ObjectDetectorTRTX_YOLOV5();
    virtual ~ObjectDetectorTRTX_YOLOV5();

public:
    virtual int init(const ODParam &param);
    virtual int predict();
    virtual int setImage(const cv::Mat &img, bool keep_aspect=false);

protected:
    int setTensor(float *data);

private:
    nvinfer1::IRuntime* m_trt_runtime;
    nvinfer1::ICudaEngine* m_trt_engine;
    nvinfer1::IExecutionContext* m_trt_context;
    Logger m_trt_logger;
    int m_output_size;
    int m_class_num;
    void *m_gpu_buffers[2];
    cudaStream_t m_cudastream;
};

#else

class ObjectDetectorTRTX_YOLOV5 : public ObjectDetector
{
public:
    ObjectDetectorTRTX_YOLOV5()=default;
    virtual ~ObjectDetectorTRTX_YOLOV5()=default;

public:
    virtual int init(const ODParam &param){
        LOG_ERROR("Please implement ObjectDetectorTRTX_YOLOV5::init");
        return -1;
    }
    virtual int predict(){ 
        LOG_ERROR("Please implement ObjectDetectorTRTX_YOLOV5::predict");
        return -1; 
    }
};

						   

#endif // ENABLE_TENSORRTX

#endif // __OBJECT_DETECTOR_TRTX_YOLOV5_HPP__
