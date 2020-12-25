#ifndef __SEMANTIC_SEGMENTATION_T7_HPP__
#define __SEMANTIC_SEGMENTATION_T7_HPP__

#include "semantic_segmentation.hpp"
#include "sak_log.hpp"
#include "opencv2/opencv.hpp"

#ifdef ENABLE_TORCH
#include "torch/torch.h"
#include "torch/script.h"

class SemanticSegT7 : public SemanticSeg
{
public:
    SemanticSegT7();
    virtual ~SemanticSegT7();

public:
    virtual int init(const SegParam &param);
    virtual int predict();

protected:
    void* m_module_wrapper;
    int m_device_id;
};

#else

class SemanticSegT7 : public SemanticSeg
{
public:
    SemanticSegT7()=default;
    virtual ~SemanticSegT7()=default;

public:
    virtual int init(const SegParam &param) { 
        LOG_ERROR("Please implement SemanticSegT7");
        return -1;
    }
    virtual int predict() { 
        LOG_ERROR("Please implement SemanticSegT7");
        return -1; 
    }
};

#endif // ENABLE_TORCH

#endif // __SEMANTIC_SEGMENTATION_T7_HPP__
