#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "classifier_video.hpp"
#include "sak_utils.hpp"
#include "sak_log.hpp"
using namespace std;

int ClassifierVid::setImage(std::vector<cv::Mat> &imgV){
    m_imgV.clear();
    if (imgV.size() < m_net_t)
        return -1;
    
    for (int i=0; i<m_net_t; i++){
        // resize to 256 first
        cv::Mat img;
        cv::resize(imgV[i], img, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Rect roi;
        roi.x = int((img.cols-m_net_w)/2);
        roi.y = int((img.rows-m_net_h)/2);
        roi.width = m_net_w;
        roi.height = m_net_h;
        m_imgV.push_back(img(roi));
    }
    return 0;
}

int ClassifierVid::readLabelMap(const string &fileName){
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
