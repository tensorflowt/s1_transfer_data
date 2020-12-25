#ifndef __CLASSIFIER_TF_HPP__
#define __CLASSIFIER_TF_HPP__

#include "classifier.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TENSORFLOW
#include "tensorflow/core/public/session.h"

class ClassifierTF : public Classifier
{
public:
    ClassifierTF();
    virtual ~ClassifierTF();

public:
    virtual int init(const CLSParam &param);
    virtual int predict(float &score, int &id, std::string &name);
    virtual int predict(std::map<std::string, float> &res);

protected:
    int setTensor(tensorflow::Tensor &in_tensor);
    
protected:
    std::unique_ptr<tensorflow::Session> m_session;
    std::string m_input_name;
    std::string m_output_name;
};

#else

class ClassifierTF : public Classifier
{
public:
    ClassifierTF()=default;
    virtual ~ClassifierTF()=default;

public:
    virtual int init(const CLSParam &param){ 
        LOG_ERROR("Please implement ClassifierTF");
        return -1; 
    }
    virtual int predict(float &score, int &id, std::string &name){ 
        LOG_ERROR("Please implement ClassifierTF");
        return -1; 
    }
    virtual int predict(std::map<std::string, float> &res){
        LOG_ERROR("Please implement ClassifierTF");
        return -1; 
    }
};

#endif // ENABLE_TENSORFLOW

#endif // __CLASSIFIER_TF_HPP__
