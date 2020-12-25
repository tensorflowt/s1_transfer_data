#ifndef __OBJECT_DETECTOR_HPP__
#define __OBJECT_DETECTOR_HPP__

#include <string>
#include <map>
#include "opencv2/opencv.hpp"
#include "object_detection.hpp"

typedef struct OD_PARAM_TAG{
public:
    OD_PARAM_TAG(){
        gpu_id = -1;
        deploy = "None";
        model = "None";
        label_map = "None";
        net_w = 0;
        net_h = 0;
        nms_thresh = 0;
        gpu_fraction = 0.1f;
        conf_thresh = 0;
        // for batch mode
        max_batch_size = 1;
    }
public:
    std::string deploy;
    std::string model;
    std::string label_map;
    int gpu_id;
    int net_w;
    int net_h;
    float nms_thresh;
    float gpu_fraction;
    float conf_thresh;
    // for batch mode
    int max_batch_size;
}ODParam;

class __declspec(dllexport) ObjectDetector
{
public:
    ObjectDetector(){
        m_scale_x = 0;
        m_scale_y = 0;
        m_origin_img_w = 0;
        m_origin_img_h = 0;
    }
    virtual ~ObjectDetector() = default;

public:
    virtual int init(const ODParam &param) = 0;
    virtual int predict() = 0;
    virtual int setImage(const cv::Mat &img, bool keep_aspect=false);
    
    // detection result
    virtual int object(int idx, ObjectDetection &obj){
        if (idx >= m_objs.size())
            return -1;
        obj = m_objs[idx];
        return 0;
    }
    virtual int objectNum(){ 
        return m_objs.size(); 
    }
    
protected:
    int readLabelMap(const std::string &fileName);
    int warmUp(){
        int warm_times = 2;
        for(int i = 0; i< warm_times; i++)
        {
            m_img = cv::Mat(m_net_h, m_net_w, CV_8UC3);
            cv::randu(m_img, cv::Scalar::all(0), cv::Scalar::all(255));
            predict();
        }
        return 0;
    }

protected:
    // input image for network
    cv::Mat m_img;
    // resize scale for width/height
    float   m_scale_x;
    float   m_scale_y;
    // input width/height
    int     m_net_w;
    int     m_net_h;
    // original image width/height
    int     m_origin_img_w;
    int     m_origin_img_h;
    // label map: <i,label_0>,...
    std::map<int, std::string> m_label_map;
    // object result
    std::vector<ObjectDetection> m_objs;
    // detect object score & nms threshold
    float conf_thresh;
    float nms_thresh;
};


#endif
