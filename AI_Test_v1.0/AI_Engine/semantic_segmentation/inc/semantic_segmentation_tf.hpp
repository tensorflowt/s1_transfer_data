#ifndef __SEMANTIC_SEGMENTATION_TF_HPP__
#define __SEMANTIC_SEGMENTATION_TF_HPP__

#include "semantic_segmentation.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TENSORFLOW
#include "tensorflow/core/public/session.h"

class SemanticSegTF : public SemanticSeg
{
public:
    SemanticSegTF();
    virtual ~SemanticSegTF();

public:
    virtual int init(const SegParam &param);
    virtual int predict();

protected:
    int setTensor(tensorflow::Tensor &in_tensor);
    
private:
    std::unique_ptr<tensorflow::Session> m_session;
};

#else

class SemanticSegTF : public SemanticSeg
{
public:
    SemanticSegTF()=default;
    virtual ~SemanticSegTF()=default;

public:
    virtual int init(const SegParam &param) { 
        LOG_ERROR("Please implement SemanticSegTF");
        return -1;
    }
    virtual int predict() { 
        LOG_ERROR("Please implement SemanticSegTF");
        return -1; 
    }
};

#endif // ENABLE_TENSORFLOW

#endif // __SEMANTIC_SEGMENTATION_TF_HPP__
