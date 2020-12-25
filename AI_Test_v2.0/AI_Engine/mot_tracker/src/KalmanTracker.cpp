#include "KalmanTracker.h"
int64_t KalmanTracker::g_kf_count = 0;

using namespace cv;
KalmanTracker::KalmanTracker(){
    initKF(cv::Rect_<float>());
    m_age_total = 0;
    m_age_continue = 0;
    m_age_dead = 0;
    m_id = g_kf_count;
    m_name = "unknown";
    m_trace_max_len = 50;
    m_trace.clear();
}
KalmanTracker::KalmanTracker(cv::Rect_<float> initRect){
    m_trace.clear();
    m_age_total = 0;
    m_age_continue = 0;
    m_age_dead = 0;
    m_id = g_kf_count;
    g_kf_count++;
    m_name = "unknown";
    m_trace_max_len = 50;
    initKF(initRect);
}

KalmanTracker::~KalmanTracker(){
    m_trace.clear();
}

void KalmanTracker::initKF(cv::Rect_<float> stateMat){
	int stateNum = 7;
	int measureNum = 4;
	m_kf.init(stateNum, measureNum, 0);
	m_measurement = Mat(measureNum, 1, CV_32F, Scalar(0));
	m_kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
                             1, 0, 0, 0, 1, 0, 0,
                             0, 1, 0, 0, 0, 1, 0,
                             0, 0, 1, 0, 0, 0, 1,
                             0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0,
                             0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 1);
	setIdentity(m_kf.measurementMatrix);
	setIdentity(m_kf.processNoiseCov, Scalar::all(1e-2));
	setIdentity(m_kf.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(m_kf.errorCovPost, Scalar::all(1));
	
	// initialize state vector with bounding box in [cx,cy,s,r] style
	m_kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	m_kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	m_kf.statePost.at<float>(2, 0) = stateMat.area();
	m_kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;

    //
    m_correctBox = stateMat;
    m_trace.push_back(m_correctBox);
}

// Predict the estimated bounding box.
cv::Rect_<float> KalmanTracker::predict()
{
	// predict
	Mat p = m_kf.predict();
	m_age_total += 1;
    m_age_continue +=1;
    m_age_dead += 1;

    if (m_age_dead > 1)
        m_age_continue = 0;
    
    m_predBox = state2Rect(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));
	return m_predBox;
}

// Update the state vector with observed bounding box.
void KalmanTracker::update(cv::Rect_<float> stateMat){
	m_age_dead = 0;
    
	// measurement
	m_measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	m_measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	m_measurement.at<float>(2, 0) = stateMat.area();
	m_measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

	// update
	m_kf.correct(m_measurement);
    Mat s = m_kf.statePost;
	m_correctBox = state2Rect(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
    m_detBox = stateMat;
    if (m_trace.size() >= m_trace_max_len)
        m_trace.pop_front();
    m_trace.push_back(m_correctBox);
    return;
}

// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
cv::Rect_<float> KalmanTracker::state2Rect(float cx, float cy, float s, float r){
	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);
    
	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;
    
	return cv::Rect_<float>(x, y, w, h);
}
