#ifndef __CLASSIFIER_TF_HPP__
#define __CLASSIFIER_TF_HPP__

#include "classifier.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_MXNET
#include <mxnet/c_predict_api.h>

class ClassifierMXNET : public Classifier
{
public:
    ClassifierMXNET();
    virtual ~ClassifierMXNET();

public:
    virtual int init(const CLSParam &param);
    int extractFeature(cv::Mat &fea); 

    // TODO implement predict function  
    virtual int predict(float &score, int &id, std::string &name){ return -1; };
    virtual int predict(std::map<std::string, float> &res) { return -1; };

protected:
    int loadFile(const std::string &fname, std::vector<char> &buf);
    int setTensor(std::vector<float> &input);

protected:
    PredictorHandle m_handle;
};

#else

class ClassifierMXNET : public Classifier
{
public:
    ClassifierMXNET()=default;
    virtual ~ClassifierMXNET()=default;

public:
    int extractFeature(cv::Mat &fea){ return -1;}
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
