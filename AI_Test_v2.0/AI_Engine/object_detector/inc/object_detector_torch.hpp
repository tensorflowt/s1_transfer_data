#ifndef __OBJECT_DETECTOR_TORCH_HPP__
#define __OBJECT_DETECTOR_TORCH_HPP__

#include "object_detector.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TORCH
#include "torch/torch.h"
#include "torch/script.h"

class ObjectDetectorTorch : public ObjectDetector
{
public:
    ObjectDetectorTorch();
    virtual ~ObjectDetectorTorch();

public:
    virtual int init(const ODParam &param);
    virtual int predict();

    // for post process
    int postProcess(torch::Tensor &src_tensor);

    float m_conf_thresh = 0;
    int m_device_id = -1;

protected:
    std::unique_ptr<torch::jit::script::Module> m_module;
};

#else

class ObjectDetectorTorch : public ObjectDetector
{
public:
    ObjectDetectorTorch()=default;
    virtual ~ObjectDetectorTorch()=default;

public:
    virtual int init(const ODParam &param){ 
        LOG_ERROR("Please implement ObjectDetectorTorch::init");
        return -1; 
    }


    virtual int predict(){ 
        LOG_ERROR("Please implement ObjectDetectorTorch::predict");
        return -1; 
    }
};

#endif // ENABLE_TORCH

#endif // __OBJECT_DETECTOR_TORCH_HPP__
