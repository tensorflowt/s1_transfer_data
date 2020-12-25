#ifndef __MOT_TRACKER_SORT_HPP__
#define __MOT_TRACKER_SORT_HPP__

#include "opencv2/opencv.hpp"
#include "mot_tracker.hpp"
#include "object_detection.hpp"	
#include "sak_log.hpp"

class MotTrackerSORT : public MotTracker{

public:
    MotTrackerSORT();
    virtual ~MotTrackerSORT();

public:
    virtual int process(std::vector<ObjectDetection> &objs);

protected:
    int updateTracks(std::vector<ObjectDetection> &obj);
    int calcCostMatrix(std::vector<cv::Rect> &pbboxes, std::vector<cv::Rect> &dbboxes, std::vector<std::vector<double> > &iou_matrix);
    double calcIOU(cv::Rect &bbox0, cv::Rect &bbox1);
    double calcGIOU(cv::Rect &bbox0, cv::Rect &bbox1); // general iou
    double calcDist(cv::Rect &bbox0, cv::Rect &bbox1);
    
    bool isOverlapedWithCurTracker(cv::Rect &box);
    
protected:
    float m_iou_thresh;
    float m_giou_thresh;
    float m_dist_thresh;
    
    std::vector<KalmanTracker *> m_trackers;
    HungarianAlgorithm m_hungAlgo;
};

#endif
