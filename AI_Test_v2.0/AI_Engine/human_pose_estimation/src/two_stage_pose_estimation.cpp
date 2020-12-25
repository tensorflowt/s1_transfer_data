#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "two_stage_pose_estimation.hpp"
#include "sak_utils.hpp"
#include "sak_log.hpp"

using namespace std;


cv::Mat padding_img(int H, int W, cv::Mat roi_img, cv::Rect rect) {
	cv::Mat image = cv::Mat::zeros(H, W, CV_8UC3);
	roi_img.copyTo(image(rect));
	return image; 
}


Point2f  get_3d_point(Point2f a, Point2f b) {
	Point2f direct = a - b;
	Point2f final = Point2f(b.x - direct.y, b.y + direct.x);
		return final;
}

cv::Mat crop_box(cv::Mat img, cv::Rect rect,int H, int W) {
	int crop_x1 = cv::max(0, rect.x);
	int crop_y1 = cv::max(0, rect.y);
	int crop_x2 = cv::min(img.cols - 1, rect.x + rect.width - 1);
	int crop_y2 = cv::min(img.rows - 1, rect.y + rect.height - 1);
	int h = rect.height;
	int w = rect.width; 
	float lenH = std::max(h, w*H/W);
	float lenW = lenH * W / H;
	int pad_h = (lenH - h) / 2;
	int pad_w = (lenW - w) / 2;
	cv::Mat roi_img = img(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
	cv::Mat pad_img = padding_img(img.rows, img.cols,roi_img, rect);
	Point2f srcTri[3];
	srcTri[0] = Point2f(crop_x1 - pad_w, crop_y1 - pad_h);
	srcTri[1] = Point2f(crop_x2 + pad_w, crop_y2 + pad_h);
	srcTri[2] = get_3d_point(srcTri[0], srcTri[1]);
	Point2f dstTri[3];
	dstTri[0] = Point2f(0.0, 0.0);
	dstTri[1] = Point2f(W - 1, H - 1);
	dstTri[2] = get_3d_point(dstTri[0], dstTri[1]); 
	cv::Mat  affine_img = cv::Mat::zeros(H, W, CV_8UC3);
	cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
    cv::warpAffine(pad_img, affine_img, warp_mat, affine_img.size());
	return affine_img; 
}

cv::Rect expansion(cv::Mat img, cv::Rect rect, float scale_ratio) {
	int height = img.rows;
	int width = img.cols;
	int x1 = rect.x;
	int y1 = rect.y; 
	int x2 = rect.x + rect.width - 1;
	int y2 = rect.y + rect.height - 1;
	int new_x1 = std::max(0, (int)(x1-rect.width*scale_ratio/2));
	int new_y1 = std::max(0, (int)(y1 - rect.height*scale_ratio / 2));
	int new_x2 = std::max(std::min((int)(width - 1), 
		(int)(x2+rect.width*scale_ratio/2)), new_x1 + 5);
	int new_y2 = std::max(std::min((int)(height - 1),
		(int)(y2 + rect.height*scale_ratio / 2)), new_y1 + 5);
	cv::Rect rect1(new_x1, new_y1, new_x2 - new_x1 + 1, new_y2 - new_y1 + 1);
	return rect1;
}




int HPestimation::setImage(const cv::Mat &img, cv::Rect rect){
    m_img_real_h = img.rows; 
    m_img_real_w = img.cols; 
    m_scale_x = float(m_img_real_w)/896;
    m_scale_y = float(m_img_real_h)/854; 

    cv::Rect resize_rect(int(rect.x/m_scale_x), int(rect.y/m_scale_y), 
    int(rect.width/m_scale_x), int(rect.height/m_scale_y));
    float scaleRate = 0.2; 
    cv::Mat resize_img = cv::Mat(854, 896, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::resize(img, resize_img, cv::Size(896, 854), 0, 0, cv::INTER_LINEAR);
    m_rect = expansion(resize_img, resize_rect, scaleRate);
    m_img = crop_box(resize_img, m_rect, m_net_h, m_net_w);
	cv::cvtColor(m_img, m_img, cv::COLOR_BGR2RGB);
    return 0; 
}



/*
int HPestimation::readLabelMap(const string &fileName){
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
*/

int HPestimation:: posData(Valpoints &points){
	vector<Point2f> keypoints;
	vector<float> scores = m_points.scores;  
	vector<Point2f> resize_points = m_points.preds; 
    points.scores = scores; 
	for(int i=0; i<resize_points.size(); i++){
		Point2f point = resize_points[i];
        Point2f orign_point = Point2f(point.x*m_scale_x, point.y*m_scale_y);
		keypoints.push_back(orign_point);
	}
	points.preds = keypoints;
    return 0;
}


