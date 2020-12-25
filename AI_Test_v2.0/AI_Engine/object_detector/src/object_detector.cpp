#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "object_detector.hpp"
#include "sak_utils.hpp"
#include "sak_log.hpp"
using namespace std;

int ObjectDetector::setImage(const cv::Mat &img, bool keep_aspect){
    
    m_origin_img_w = img.cols;
    m_origin_img_h = img.rows;
    if (true == keep_aspect){
        m_scale_x = float(img.cols)/m_net_w;
        m_scale_y = float(img.rows)/m_net_h;
        
        int scale_img_w = m_net_w;
        int scale_img_h = m_net_h;
        if (m_scale_x > m_scale_y){
            m_scale_y = m_scale_x;
            scale_img_h = int(img.rows/m_scale_y)/2*2;
        }
        else {
            m_scale_x = m_scale_y;
            scale_img_w = int(img.cols/m_scale_x)/2*2;
        }
        m_img = cv::Mat(m_net_h, m_net_w, CV_8UC3, cv::Scalar(127, 127, 127));
        cv::Rect roi(0,0,scale_img_w, scale_img_h);
        cv::resize(img, m_img(roi),
                   cv::Size(scale_img_w, scale_img_h), 0, 0,
                   cv::INTER_LINEAR);
        cv::cvtColor(m_img, m_img, cv::COLOR_BGR2RGB);
    }
    else {
        cv::resize(img, m_img,
                   cv::Size(m_net_w, m_net_h), 0, 0,
                   cv::INTER_LINEAR);
        m_scale_x = float(img.cols)/m_net_w;
        m_scale_y = float(img.rows)/m_net_h;
        cv::cvtColor(m_img, m_img, cv::COLOR_BGR2RGB);
    }
    return 0;
}



int ObjectDetector::readLabelMap(const string &fileName){
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
