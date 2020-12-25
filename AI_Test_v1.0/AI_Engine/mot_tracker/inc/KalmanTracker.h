#ifndef __KALMAN_TRACKER_H__
#define __KALMAN_TRACKER_H__

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <deque>
#include <vector>

class KalmanTracker
{
public:
    KalmanTracker();
    KalmanTracker(cv::Rect_<float> initRect);
	~KalmanTracker();
    
public:
	cv::Rect_<float> predict();
	void update(cv::Rect_<float> stateMat);

    // Set Active
    // can be called with update
    /*
      inline void active(cv::Rect &box){
        m_age_dead = 0;
        m_age_total += 1;
        m_age_continue +=1;

        m_correctBox = box;
        m_detBox = box;
        if (m_trace.size() >= m_trace_max_len)
            m_trace.pop_front();
        m_trace.push_back(m_correctBox);
        }
    */

    int setTraceMaxLen(int len){
        m_trace_max_len = len;
    }
    
    // Status 
    int age(){
	return m_age_total;
    }
    int deadAge(){
        return m_age_dead;
    }
    void deadAgeAdd(){
        m_age_dead += 1;
    }
    
    inline bool isDead(){
        if (m_age_dead>=15)
            return true;
        return false;
    }

    inline bool isActive(){
        if (m_age_total>=5 && m_age_continue>=3)
            return true;
        return false;
    }

    // Result
    inline cv::Rect_<float> trackPos(){
        return m_correctBox;
    }

    inline void setTrackId(int64_t id){
        m_id = id;
    }

    inline int trackId(){ 
        return m_id;
    }

    inline int trace(std::vector<cv::Rect> &tra){ 
        tra.clear();
        std::deque<cv::Rect>::iterator it = m_trace.begin();
        while (it != m_trace.end())
            tra.push_back(*it++);
        return tra.size();
    }
    
    inline void setName(const std::string &name){
        m_name = name;
        return ;
    }
    inline std::string name(){
        return m_name;
    }

protected:
    void initKF(cv::Rect_<float> stateMat);
	cv::Rect_<float> state2Rect(float cx, float cy, float s, float r);

protected:
    int64_t m_age_total;
    int64_t m_age_continue;
    int64_t m_age_dead;
	int64_t m_id;
    std::string m_name;

	cv::KalmanFilter m_kf;
	cv::Mat m_measurement;
    cv::Rect_<float> m_predBox;
    cv::Rect_<float> m_correctBox;
    cv::Rect_<float> m_detBox;
    std::deque<cv::Rect> m_trace;
    int m_trace_max_len;
    
protected:
	static int64_t g_kf_count;
};

#endif
