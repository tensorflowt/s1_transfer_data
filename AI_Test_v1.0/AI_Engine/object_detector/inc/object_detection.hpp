#ifndef __OBJECT_DETECTION_HPP__
#define __OBJECT_DETECTION_HPP__

#include "opencv2/opencv.hpp"
class ObjectDetection
{
protected:
    float m_x;
    float m_y;
    float m_w;
    float m_h;
    float m_confidence;
    int   m_class_id;
    std::string m_name;

    int m_attr_id;
    float m_attr_conf;
    std::string m_attr_name;
    
public:
    ObjectDetection(){
        m_x = 0 , m_y = 0, m_w = 0, m_h = 0;
        m_confidence = 0;
        m_class_id = -1;
        m_name = "None";

        m_attr_id = -1;
        m_attr_conf = 0;
        m_attr_name = "None";
    }
    ObjectDetection(float x, float y, float w, float h, float conf, 
                    int class_id, std::string name,  
                    float attr_id=-1, float attr_score=0, std::string attr_name="None"){
        m_x = x, m_y = y, m_w = w, m_h = h;
        m_confidence = conf;
        m_class_id = class_id;
        m_name = name;
        
        m_attr_id = attr_id;
        m_attr_conf = attr_score;
        m_attr_name = attr_name;
    }

public:
    ObjectDetection& operator=(ObjectDetection other)
    {
        cv::Rect roi = other.getRect();
        m_x = roi.x;
        m_y = roi.y;
        m_w = roi.width;
        m_h = roi.height;
        m_confidence = other.getScore();
        m_class_id = other.getClassId();
        m_name = other.getName();

        m_attr_id = other.getAttrId();
        m_attr_conf = other.getAttrScore();
        m_attr_name = other.getAttrName();
        return *this;
    }
public:
    inline cv::Rect    getRect()  { return cv::Rect(m_x, m_y, m_w, m_h);}
    inline int         getClassId() { return m_class_id; }
    inline float       getScore() { return m_confidence; }
    inline std::string getName()  { return m_name; }
public:
    inline int         getAttrId() { return m_attr_id; }
    inline float       getAttrScore() { return m_attr_conf; }
    inline std::string getAttrName()  { return m_attr_name; }
public:
    inline void setRect(cv::Rect &roi){ 
        m_x = roi.x;
        m_y = roi.y;
        m_w = roi.width;
        m_h = roi.height;
    }

    inline void setName(std::string name){ 
        m_name = name;
    }
};

#endif
