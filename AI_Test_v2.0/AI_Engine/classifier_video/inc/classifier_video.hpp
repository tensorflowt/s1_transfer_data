#ifndef __CLASSIFIER_VID_HPP__
#define __CLASSIFIER_VID_HPP__

#include <string>
#include <map>
#include "opencv2/opencv.hpp"

typedef struct VID_CLS_PARAM_TAG{
public:
    VID_CLS_PARAM_TAG(){
        gpu_id = -1;
        deploy = "None";
        model = "None";
        label_map = "None";
        input_node = "None";
        output_node = "None";
        net_w = 0;
        net_h = 0;
        net_t = 0;
        gpu_fraction = 0.1;
    }
public:
    std::string deploy;
    std::string model;
    std::string label_map;
    std::string input_node;
    std::string output_node;
    int gpu_id;
    int net_w;
    int net_h;
    int net_t;
    float gpu_fraction;
}VidCLSParam;

class ClassifierVid
{
public:
    ClassifierVid(){
    }
    virtual ~ClassifierVid() = default;

public:
    virtual int init(const VidCLSParam &param) = 0;
    virtual int predict(float &score, int &id, std::string &name) = 0;
    virtual int predict(std::map<std::string, float> &res)=0;
    virtual int setImage(std::vector<cv::Mat> &imgV);
    
protected:
    int readLabelMap(const std::string &fileName);
    int warmUp(){
        for (int i=0; i<m_net_t; i++){
            cv::Mat img = cv::Mat(m_net_h, m_net_w, CV_8UC3);
            cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
            m_imgV.push_back(img);
        }
        
        float score;
        int id;
        std::string name;
        predict(score, id, name);
        return 0;
    }

protected:
    std::vector<cv::Mat> m_imgV;
    std::string m_output_node;
    std::string m_input_node;

    int     m_net_w;
    int     m_net_h;
    int     m_net_t;
    std::map<int, std::string> m_label_map;
};


#endif
