#ifndef __HUMAN_POSE_ESTIMATION_HPP__
#define __HUMAN_POSE_ESTIMATION_HPP__

#include <string>
#include <map>
#include "opencv2/opencv.hpp"
# include<vector>
typedef cv::Point_<float> Point2f;

typedef struct Valpoints{
    std::vector<Point2f> preds;
    std::vector<float> scores;
}Valpoints;

typedef struct HPES_PARAM_TAG{
public:
    HPES_PARAM_TAG(){
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
}HpesParam;

class HPestimation
{
public:
    HPestimation(){
        m_scale_x = 0;
        m_scale_y = 0;
    }
    virtual ~HPestimation() = default;

public:
    virtual int init(const HpesParam &param) = 0;
    virtual int predict() = 0;
    virtual int setImage(const cv::Mat &img, cv::Rect rect);
    virtual int posData(Valpoints &points);
    

protected:
 //   int readLabelMap(const std::string &fileName);
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
    cv::Rect m_rect;
    std::map<int, std::string> m_label_map;
    Valpoints m_points; 
    std::string m_output_type;
};

#endif //__SEMANTIC_SEGMENTATION_HPP__
