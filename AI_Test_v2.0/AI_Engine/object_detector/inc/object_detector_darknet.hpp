#ifndef __OBJECT_DETECTOR_DARKNET_HPP__
#define __OBJECT_DETECTOR_DARKNET_HPP__

#include "object_detector.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_DARKNET
#define GPU
#define CUDNN
#define OPENCV
#include "darknet.h"
class ObjectDetectorDarknet : public ObjectDetector
{
public:
    ObjectDetectorDarknet();
    virtual ~ObjectDetectorDarknet();

public:
    virtual int setImage(const cv::Mat &img, bool keep_aspect=true);
    virtual int init(const ODParam &param);
    virtual int predict();

private:
    int setTensor();

private:
    // network pointer
    network *m_network;
    // 
    image m_tensor;
    float m_nms_thresh;

    int     m_offset_x;
    int     m_offset_y;
};

#else

class ObjectDetectorDarknet : public ObjectDetector
{
public:
    ObjectDetectorDarknet()=default;
    virtual ~ObjectDetectorDarknet()=default;

public:
    virtual int init(const ODParam &param){ 
        LOG_ERROR("Please implement ObjectDetectorDarknet");
        return -1; 
    }
    virtual int predict(){ 
        LOG_ERROR("Please implement ObjectDetectorDarknet");
        return -1; 
    }
};

#endif // ENABLE_DARKNET

#endif // __OBJECT_DETECTOR_DARKNET_HPP__
