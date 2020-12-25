#ifndef __CLASSIFIER_HPP__
#define __CLASSIFIER_HPP__

#include <string>
#include <map>
#include "opencv2/opencv.hpp"

typedef struct CLS_PARAM_TAG{
public:
    CLS_PARAM_TAG(){
        gpu_id = -1;
        deploy = "None";
        model = "None";
        label_map = "None";
        net_w = 0;
        net_h = 0;
        gpu_fraction = 0.1;
        input_name = "input";
        output_name = "MobilenetV2/Predictions/Reshape_1";
    }
public:
    std::string deploy;
    std::string model;
    std::string label_map;
    int gpu_id;
    int net_w;
    int net_h;
    float gpu_fraction;
    std::string input_name;
    std::string output_name;
}CLSParam;

class Classifier
{
public:
    Classifier(){
    }
    virtual ~Classifier() = default;

public:
    virtual int init(const CLSParam &param) = 0;
    virtual int predict(float &score, int &id, std::string &name) = 0;
    virtual int predict(std::map<std::string, float> &res)=0;
    virtual int setImage(const cv::Mat &img, bool keep_ratio=false, bool use_rgb=true);
    
protected:
    int readLabelMap(const std::string &fileName);
    int warmUp(){
        m_img = cv::Mat(m_net_h, m_net_w, CV_8UC3);
        cv::randu(m_img, cv::Scalar::all(0), cv::Scalar::all(255));

        float score;
        int id;
        std::string name;
        predict(score, id, name);
        return 0;
    }

protected:
    cv::Mat m_img;
    int     m_net_w;
    int     m_net_h;
    std::map<int, std::string> m_label_map;
};


#endif
