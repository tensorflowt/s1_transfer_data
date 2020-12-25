#ifndef __CLASSIFIER_VID_I3D_HPP__
#define __CLASSIFIER_VID_I3D_HPP__

#include "classifier_video.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TENSORFLOW
#include "tensorflow/core/public/session.h"

class ClassifierVidI3D : public ClassifierVid
{
public:
    ClassifierVidI3D();
    virtual ~ClassifierVidI3D();

public:
    virtual int init(const VidCLSParam &param);
    virtual int predict(float &score, int &id, std::string &name);
    virtual int predict(std::map<std::string, float> &res);

protected:
    int setTensor(tensorflow::Tensor &in_tensor);
    
protected:
    std::unique_ptr<tensorflow::Session> m_session;
};

#else

class ClassifierVidI3D : public ClassifierVid
{
public:
    ClassifierVidI3D()=default;
    virtual ~ClassifierVidI3D()=default;

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
