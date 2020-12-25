#ifndef __SEMANTIC_SEGMENTATION_HPP__
#define __SEMANTIC_SEGMENTATION_HPP__

#include <string>
#include <map>
#include "opencv2/opencv.hpp"

typedef struct SEG_PARAM_TAG{
public:
    SEG_PARAM_TAG(){
        gpu_id = -1;
        deploy = "None";
        model = "None";
        label_map = "None";
        net_w = 0;
        net_h = 0;
        output_type="INT32";
        gpu_fraction = 0.1f;
    }
public:
    std::string deploy;
    std::string model;
    std::string label_map;
    std::string output_type;
    int gpu_id;
    int net_w;
    int net_h;
    float gpu_fraction;
}SegParam;

class SemanticSeg
{
public:
    SemanticSeg(){
        m_scale_x = 0;
        m_scale_y = 0;
    }
    virtual ~SemanticSeg() = default;

public:
    virtual int init(const SegParam &param) = 0;
    virtual int predict() = 0;
    virtual int setImage(const cv::Mat &img, bool keep_aspect=false);

    // segmentation result
    virtual int segData(cv::Mat &mask, int height, int width){
        cv::resize(m_mask(cv::Rect(0, 0, m_img_real_w, m_img_real_h)), mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
        return 0;
    }
protected:
    int readLabelMap(const std::string &fileName);
    int warmUp(){
        m_img = cv::Mat(m_net_h, m_net_w, CV_8UC3);
        cv::randu(m_img, cv::Scalar::all(0), cv::Scalar::all(255));
        predict();
    }

protected:
    cv::Mat m_img;
    float   m_scale_x;
    float   m_scale_y;
    int m_img_real_w;
    int m_img_real_h;
    int m_net_w;
    int m_net_h;
    std::map<int, std::string> m_label_map;
    cv::Mat m_mask;
    std::string m_output_type;
};

#endif //__SEMANTIC_SEGMENTATION_HPP__
