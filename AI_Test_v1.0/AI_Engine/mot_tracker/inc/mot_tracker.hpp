#ifndef __MOT_TRACKER_HPP__
#define __MOT_TRACKER_HPP__

#include "KalmanTracker.h"
#include "Hungarian.h"
#include "object_detection.hpp"	

typedef struct TraceTag
{
    int64_t id;
    int status; // 0-dead 1-activate 2-disappear
    cv::Rect bbox;
    std::string name;
    std::vector<cv::Rect> history;
}Trace;

class MotTracker{
public:
    MotTracker() {
        m_cur_frame_time = 0;
    }
    virtual ~MotTracker() = default;
    
public:
    virtual int process(std::vector<ObjectDetection> &objs)=0;
    
    inline int traceNum() {
        return m_traces.size();
    }
    
    inline int trace(int idx, Trace &tra){
        if (idx > m_traces.size())
            return -1;
        tra = m_traces[idx];
        return 0;
    }
    
    inline int objectNum(){
        return  m_objs.size();
    }
    
    virtual int object(int idx, ObjectDetection &obj){
        if (idx >= m_objs.size())
            return -1;
        obj = m_objs[idx];
        return  0;
    }
protected:
    std::vector<ObjectDetection> m_objs;
    std::vector<Trace> m_traces;
    int64_t m_cur_frame_time;
};

#endif
