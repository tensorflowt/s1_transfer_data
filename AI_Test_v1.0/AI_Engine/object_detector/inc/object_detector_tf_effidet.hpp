#ifndef __OBJECT_DETECTOR_TF_EFFIDET_HPP__
#define __OBJECT_DETECTOR_TF_EFFIDET_HPP__

#include "object_detector.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TENSORFLOW
#include "tensorflow/core/public/session.h"

class ObjectDetectorTF_EFFIDET : public ObjectDetector
{
public:
    ObjectDetectorTF_EFFIDET();
    virtual ~ObjectDetectorTF_EFFIDET();

public:
    virtual int init(const ODParam &param);
    virtual int predict();


public:
    int setTensor(tensorflow::Tensor &in_tensor);
    
    // batch mode
    int setBatchTensor(tensorflow::Tensor& in_tensor);
    int setBatchImage(const std::vector<cv::Mat> &batch_imgs, bool keep_aspect);
    int predictBatch();

    /*
    Func: 
        batchObject -- fetch detection result
    Input: 
        idx -- image index
        obj -- detected objects
    Output:
        status -- 0:sucess -1:failed
    */    
    int batchObject(int idx, std::vector<ObjectDetection> &obj);

    // for batch mode
    int m_cur_batch_size; 
    int m_max_batch_size;
    std::vector<cv::Mat>    m_batch_imgs;
    std::vector<std::pair<int,int>>  m_origin_img_size;
    std::vector<std::pair<float,float>>  m_resize_scale;
    std::vector<std::vector<ObjectDetection>>   m_batch_objs;

protected:
    std::unique_ptr<tensorflow::Session> m_session;
};

#else

class ObjectDetectorTF_EFFIDET : public ObjectDetector
{
public:
    ObjectDetectorTF_EFFIDET()=default;
    virtual ~ObjectDetectorTF_EFFIDET()=default;

public:
    virtual int init(const ODParam &param){ 
        LOG_ERROR("Please implement ObjectDetectorTF_EFFIDET::init");
        return -1; 
    }


    virtual int predict(){ 
        LOG_ERROR("Please implement ObjectDetectorTF_EFFIDET::predict");
        return -1; 
    }
};

#endif // ENABLE_TENSORFLOW

#endif // __OBJECT_DETECTOR_TF_EFFIDET_HPP__
