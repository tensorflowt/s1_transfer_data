#include "mot_tracker_sort.hpp"

MotTrackerSORT::MotTrackerSORT(){
    m_iou_thresh = 0.05;
    m_dist_thresh = 100;
}

MotTrackerSORT::~MotTrackerSORT(){
    for (auto it=m_trackers.begin(); it!=m_trackers.end(); it++)
        delete (*it);
}

int MotTrackerSORT::process(std::vector<ObjectDetection> &objs){
    m_cur_frame_time += 1;
    m_traces.clear();
    m_objs.clear();
    m_objs=objs;

    // Do tracking 
    updateTracks(m_objs);
    return 0;
}

double MotTrackerSORT::calcIOU(cv::Rect &bbox0, cv::Rect &bbox1){
    float in = (bbox0 & bbox1).area();
    float un = bbox0.area() + bbox1.area() - in;
    if (un < DBL_EPSILON)
        return 0;
    return (double)(in/un);
}

double MotTrackerSORT::calcGIOU(cv::Rect &bbox0, cv::Rect &bbox1){
    float in = (bbox0 & bbox1).area();
    float un = bbox0.area() + bbox1.area() - in;
    cv::Rect bbox_c = (bbox0 | bbox1);
    double iou_term = un < 1 ? 0: double(in/un);
    double giou_term = 0;
    
    if (bbox_c.area() > 1)
        giou_term = (bbox_c.area() - un)/float(bbox_c.area());
    else{
        giou_term = 0.0;
    }
    return iou_term - giou_term;
}

double MotTrackerSORT::calcDist(cv::Rect &bbox0, cv::Rect &bbox1){
    float x0 = bbox0.x+bbox0.width/2;
    float y0 = bbox0.y+bbox0.height/2;
    float x1 = bbox1.x+bbox1.width/2;
    float y1 = bbox1.y+bbox1.height/2;
    float dist_x = x1-x0;
    float dist_y = y1-y0;
    float dist = (double)(sqrt(dist_x*dist_x+dist_y*dist_y));
    if (dist>m_dist_thresh)
        dist = m_dist_thresh;
    
    return dist;
}

int::MotTrackerSORT::calcCostMatrix(std::vector<cv::Rect> &pred_bboxes, std::vector<cv::Rect> &det_bboxes, std::vector<std::vector<double> > &cost_matrix){
    int track_num = pred_bboxes.size();
    int det_num = det_bboxes.size();
    cost_matrix.clear();
    cost_matrix.resize(track_num, std::vector<double>(det_num, 0));

    for (int i=0; i<track_num; i++) {
        for (int j=0; j<det_num; j++){
            //cost_matrix[i][j] = 1 - calcIOU(pred_bboxes[i], det_bboxes[j]);
            cost_matrix[i][j] = calcDist(pred_bboxes[i], det_bboxes[j]);
        }
    }
    return 0;
}

bool MotTrackerSORT::isOverlapedWithCurTracker(cv::Rect &box){
    
    bool flag = false;
    for (auto it=m_trackers.begin(); it!=m_trackers.end(); it++)
    {
        cv::Rect tBox = (*it)->trackPos();
        double iou = calcIOU(tBox, box);
        if (iou>=0.3){
            flag = true;
            break;
        }
    }
    return flag;
}

int MotTrackerSORT::updateTracks(std::vector<ObjectDetection> &obj){
    std::vector<cv::Rect> det_bboxes;
    std::vector<cv::Rect> pred_bboxes;
    for (unsigned int i=0; i<obj.size(); i++)
        det_bboxes.push_back(obj[i].getRect());
    
    // Init all tracker with dets when no tracks
    if (0 == m_trackers.size()){
        for (unsigned int i=0; i < det_bboxes.size(); i++) {
            KalmanTracker *tracker = new KalmanTracker(det_bboxes[i]);
            m_trackers.push_back(tracker);
        }
        return 0;
    }
    
    // Get the predict box
    for(unsigned int i=0; i<m_trackers.size(); i++){
        cv::Rect_<float> box = m_trackers[i]->predict();
        pred_bboxes.push_back(box);
    }
    
    // Do assignment
    int track_num = pred_bboxes.size();
    int det_num = det_bboxes.size();
    std::vector<int> assignment;
    std::vector<std::vector<double> > cost_matrix;
    calcCostMatrix(pred_bboxes, det_bboxes, cost_matrix);
    m_hungAlgo.Solve(cost_matrix, assignment);

    std::vector<int> matched_det_idx;
    for (unsigned int i=0; i<assignment.size(); i++){
        //if (-1 != assignment[i] && 1-cost_matrix[i][assignment[i]]<m_iou_thresh)
        if (-1 != assignment[i] && cost_matrix[i][assignment[i]] >= (m_dist_thresh-1e-6))
            assignment[i] = -1;
        
        if (-1 != assignment[i]){
            matched_det_idx.push_back(assignment[i]);
            m_trackers[i]->update(det_bboxes[assignment[i]]);
        }
    }

    // create new trackers for unmatched detections
    for (int i=0; i<det_num; i++)
    {
        if (std::find(matched_det_idx.begin(), matched_det_idx.end(), i) != matched_det_idx.end())
            continue;
        
        // new tracker should not overlap with current tracker
        if (isOverlapedWithCurTracker(det_bboxes[i]))
            continue;
        
        KalmanTracker *tracker = new KalmanTracker(det_bboxes[i]);
        m_trackers.push_back(tracker);
    }
    
    // extract the trace and delete the dead tracker
    for (auto it = m_trackers.begin(); it != m_trackers.end();)
    {
        Trace trace;
        if ((*it)->isDead()){
            trace.status = 0;
        }
        else if ((*it)->isActive())
            trace.status = 1;
        else
            trace.status = 2;

        if ((*it)->isDead()){
            delete (*it);
            it = m_trackers.erase(it);
            continue;
        }
        
        trace.bbox = (*it)->trackPos();
        trace.id = (*it)->trackId();
        trace.name = (*it)->name();
        (*it)->trace(trace.history);
        m_traces.push_back(trace);
        it++;
    }
    return 0;
}
