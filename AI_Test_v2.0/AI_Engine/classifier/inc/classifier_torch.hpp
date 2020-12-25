#ifndef __CLASSIFIER_TORCH_HPP__
#define __CLASSIFIER_TORCH_HPP__

#include "classifier.hpp"
#include "sak_log.hpp"

#ifdef ENABLE_TORCH
#include "torch/torch.h"
#include "torch/script.h"

class ClassifierT7 : public Classifier
{
public:
    ClassifierT7();
    virtual ~ClassifierT7();

public:
    virtual int init(const CLSParam &param);
    virtual int predict(float &score, int &id, std::string &name);
    virtual int predict(std::map<std::string, float> &resM);

protected:
    torch::jit::script::Module m_module;
    int m_device_id;
    cv::Mat m_img_float;
};

#else

class ClassifierT7 : public Classifier
{
public:
    ClassifierT7()=default;
    virtual ~ClassifierT7()=default;

public:
    virtual int init(const CLSParam &param){ 
        LOG_ERROR("Please implement ClassifierT7");
        return -1; 
    }
    virtual int predict(float &score, int &id, std::string &name){ 
        LOG_ERROR("Please implement ClassifierT7");
        return -1; 
    }
    virtual int predict(std::map<std::string, float> &resM){
        LOG_ERROR("Please implement ClassifierT7");
        return -1; 
    }
};

#endif // ENABLE_TENSORFLOW

#endif // __CLASSIFIER_TF_HPP__
