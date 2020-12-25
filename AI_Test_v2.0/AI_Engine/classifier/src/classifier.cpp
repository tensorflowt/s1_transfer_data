#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "classifier.hpp"
#include "sak_utils.hpp"
#include "sak_log.hpp"
using namespace std;

int Classifier::setImage(const cv::Mat &img, bool keep_ratio, bool use_rgb){
    if (0){
        cv::Rect roi;
        roi.x = int(img.cols*0.06f + 0.5f);
        roi.y = int(img.rows*0.06f + 0.5f);
        roi.width = int(img.cols*0.875f + 0.5f);
        roi.height = int(img.rows*0.875f + 0.5f);
        cv::resize(img(roi), m_img,
                   cv::Size(m_net_w, m_net_h), 0, 0,
                   cv::INTER_LINEAR);
    }
    else if (true == keep_ratio){
        float scale_x = float(img.cols)/m_net_w;
        float scale_y = float(img.rows)/m_net_h;
        
        int scale_img_w = m_net_w;
        int scale_img_h = m_net_h;
        if (scale_x > scale_y){
            scale_y = scale_x;
            scale_img_h = int(img.rows/scale_y)/2*2;
        }
        else {
            scale_x = scale_y;
            scale_img_w = int(img.cols/scale_x)/2*2;
        }
        m_img = cv::Mat(m_net_h, m_net_w, CV_8UC3, cv::Scalar(0, 0, 0)); 
        cv::Rect roi(0,0,scale_img_w, scale_img_h);
        cv::resize(img, m_img(roi),
                   cv::Size(scale_img_w, scale_img_h), 0, 0,
                   cv::INTER_LINEAR);
    }
    else
        cv::resize(img, m_img,
                   cv::Size(m_net_w, m_net_h), 0, 0,
                   cv::INTER_LINEAR);
    if (use_rgb)
        cv::cvtColor(m_img, m_img, cv::COLOR_BGR2RGB);
    return 0;
}

int Classifier::readLabelMap(const string &fileName){
    // Read file into a string
    ifstream t(fileName.c_str());
    if (t.bad()){
        LOG_ERROR("Load label file failed");
        return -1;
    }
    
    string line;
    LOG_INFO("Loaded "<<fileName);
    while (getline(t, line))
    {
        vector<string> eles = split(line, ':');
        if (eles.size() != 2)
            break;
        m_label_map.insert(pair<int, string>(atoi(eles[0].c_str()), eles[1]));
    }
    return 0;
}
