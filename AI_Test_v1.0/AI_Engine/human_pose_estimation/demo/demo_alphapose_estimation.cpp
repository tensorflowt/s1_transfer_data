#include <iostream>
#include <string> 
#include "opencv2/opencv.hpp"
#include "object_detection.hpp"
#include "object_detector.hpp"
#include "object_detector_tf.hpp"
#include "object_detector_tflite.hpp"
#include "object_detector_darknet.hpp"
#include "object_detector_mrcnn_tf.hpp"
#include "object_detector_tf_effidet.hpp"
#include "two_stage_pose_estimation.hpp"
#include "pose_estimation_alphapose.hpp"
#include "sak_utils.hpp"


cv::Mat draw_line(cv::Mat img, std::vector<Valpoints> allpoints){
	std::vector<std::vector<int>> l_pair = { {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6},
	{5, 7}, {7, 9}, {6, 8}, {8, 10}, {17, 11}, {17, 12},{11, 13}, {12, 14}, {13, 15},
	{14, 16}
	};
	std::vector<cv::Scalar> p_color = { cv::Scalar(0, 255, 255), cv::Scalar(0, 191, 255),cv::Scalar(0, 255, 102),
		cv::Scalar(0, 77, 255), cv::Scalar(0, 255, 0), cv::Scalar(77,255,255), cv::Scalar(77, 255, 204), 
		cv::Scalar(77,204,255), cv::Scalar(191, 255, 77), cv::Scalar(77,191,255), cv::Scalar(191, 255, 77), 
		cv::Scalar(204,77,255), cv::Scalar(77,255,204), cv::Scalar(191,77,255), 
		cv::Scalar(77,255,191), cv::Scalar(127,77,255), cv::Scalar(77,255,127), cv::Scalar(0, 255, 255) };
	std::vector<cv::Scalar> line_color = { cv::Scalar(0, 215, 255), cv::Scalar(0, 255, 204), 
		cv::Scalar(0, 134, 255), cv::Scalar(0, 255, 50), cv::Scalar(77,255,222), 
		cv::Scalar(77,196,255), cv::Scalar(77,135,255), cv::Scalar(191,255,77), cv::Scalar(77,255,77),
					cv::Scalar(77,222,255), cv::Scalar(255,156,127),
					cv::Scalar(0,127,255), cv::Scalar(255,127,77), cv::Scalar(0,77,255), 
		cv::Scalar(255,77,36) };
        for (int k=0; k<allpoints.size(); k++){
		Valpoints hum = allpoints[k];
		std::vector<int>  part_line;
		std::vector<float> scores = hum.scores;
		std::vector<Point2f> points = hum.preds;
		float new_score = (hum.scores[5] + hum.scores[6]) / 2;
		Point2f new_point = Point2f((int)((points[5].x + points[6].x) / 2),
			(int)((points[5].y + points[6].y) / 2));
		scores.push_back(new_score);
		points.push_back(new_point);
		
		for (int j=0; j < scores.size(); j++) {

			if (scores[j] < 0.1){
				continue;
			}
			else {
				part_line.push_back(j);
				cv::circle(img, points[j], 2, p_color[j], -1);

			}

		};
		for (int p=0; p < l_pair.size(); p++) {
			int start = l_pair[p][0];
			int end = l_pair[p][1];
			std::vector<int>::iterator its = std::find(part_line.begin(), part_line.end(), start);
			std::vector<int>::iterator ite = std::find(part_line.begin(), part_line.end(), end);
			if (its != part_line.end() && ite != part_line.end()) {
				cv::line(img, points[start], points[end], line_color[p], 1);
			}
		};
        }
	return img;
}







int parse_config(std::string path, ODParam &detect_param, HpesParam &pose_param){
    cv::FileStorage fs(path.c_str(), cv::FileStorage::READ);
    pose_param.deploy = (std::string)fs["POSE_DEPLOY"];
    pose_param.model = (std::string)fs["POSE_MODEL"];
    detect_param.label_map = (std::string)fs["LABEL_MAP"];
    pose_param.net_w = (int)fs["POSE_W"];
    pose_param.net_h = (int)fs["POSE_H"];

    detect_param.deploy = (std::string)fs["DETECT_DEPLOY"];
    detect_param.model = (std::string)fs["DETECT_MODEL"];
    detect_param.net_w = (int)fs["DETECT_W"];
    detect_param.net_h = (int)fs["DETECT_H"];

    // get the absolute path
    int found = path.find_last_of("/\\");
    detect_param.deploy = path.substr(0,found) + "/" + detect_param.deploy;
    detect_param.model = path.substr(0,found) + "/" + detect_param.model;
    detect_param.label_map = path.substr(0,found) + "/" + detect_param.label_map;
    pose_param.deploy = path.substr(0,found) + "/" + pose_param.deploy;
    pose_param.model = path.substr(0,found) + "/" + pose_param.model;
    return 0;
}

int main(int argc, char **argv){
    //1. load config
    double time_beg = time_stamp();
     const std::string keys =
    "{help h usage ? |      | print this message   }"
    "{gpu  g         |-1    | gpu id, default -1}"
    "{@config        |.     | path to config file  }"
    "{@image         |.     | path to image file   }"
    "{@savep         |.     | path to save file}"
    ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("alphapose_estimation_image");
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
    int gpu_id = parser.get<int>("gpu");
    float th = 0.6;
    std::string config = parser.get<std::string>(0);
    std::string fileP = parser.get<std::string>(1);
    std::string saveP = parser.get<std::string>(2);
    std::string line;
    std::ifstream infile(fileP.c_str());
    if(!infile.is_open()){
        std::cout<<"Open image list file failed"<<std::endl;
        return -1;
    }

    ODParam detect_param;
    HpesParam pose_param;
    parse_config(config, detect_param, pose_param);
    detect_param.gpu_id = gpu_id;
	pose_param.gpu_id = gpu_id; 
    double time_end = time_stamp();
    std::cout<<"Load config time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;


    // 2. init model and warmup 
    time_beg = time_stamp();
    ObjectDetector *detector = 0;
    HPestimation *estimator = 0;
    detector = new ObjectDetectorTF_EFFIDET();
     std::cout<<"0"<<std::endl;
    estimator = new EstimAlphaPose();
    if (0 != detector->init(detect_param)){
        std::cout<<"Init Detector failed"<<std::endl;
        return -1;
    }
    if (0 != estimator->init(pose_param)){
        std::cout<<"Init Detector failed"<<std::endl;
        return -1;
    }
    time_end = time_stamp();
    std::cout<<"Init net time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;

    // 3. deal img 
    double time_total_s=time_stamp();
    int img_num = 0;
    cv::Mat imgM;
    int obj_num = 0;
   
     while (std::getline(infile, line)){
        // 1. Load image
        std::string img_name = std::to_string(img_num);
        std::string fileP = saveP + '/' + img_name+ ".jpg";
        std::vector<Valpoints> allpoints; 
        std::vector<cv::Rect> allrects;
        time_beg = time_stamp();
        imgM = cv::imread(line.c_str());
        time_end = time_stamp();
        std::cout<<"Load image time: "
                <<(time_end-time_beg)/1000<<"ms"<<std::endl;

        // 2. Reisze image
        time_beg = time_stamp();
        if (0 != detector->setImage(imgM,false)){
            std::cout<<"Set Image failed"<<std::endl;
            return -1;
        }
        time_end = time_stamp();
        std::cout<<"Resize image to target size time: "
                <<(time_end-time_beg)/1000<<"ms"<<std::endl;

        // 3. Detect Forward
        time_beg = time_stamp();
        obj_num = detector->predict();
        if (obj_num < 0){
            std::cout<<"Predict failed"<<std::endl;
            return -1;
        }
        time_end = time_stamp();
        std::cout<<" Detect Predict inference time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;
        
        // 4. Get Bbox
        time_beg = time_stamp();
        for (int i=0; i < obj_num; i++){
            ObjectDetection obj;
            if (0 != detector->object(i, obj))
                std::cout<<"Get object failed: "<<i<<std::endl;
            if (obj.getScore() < th)
                continue;
            std::string name = obj.getName();
            if (name !="person"){
                continue;
            }
           cv::Rect roi = obj.getRect();
           allrects.push_back(roi);
        }           
        if (0==allrects.size()){ 
            cv::imwrite(fileP, imgM);
            std::cout << "No person in the pic"<< std::endl;
            continue;

        }
        time_end = time_stamp(); 
        std::cout<<" Select Bbox time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;
        // 5. pose img set and infere 
        time_beg = time_stamp();
        for (int j=0; j<allrects.size(); j++){
            cv::Rect rect = allrects[j]; 
            if (0 != estimator->setImage(imgM, rect)){
                std::cout<<"Set Image failed"<<std::endl;
                return -1;
               }
            if (0 != estimator->predict()){
                std::cout<<"Predict failed"<<std::endl;
                return -1;
               }
            Valpoints points; 
            estimator->posData(points);
            allpoints.push_back(points);
        }
        time_end = time_stamp();
        std::cout<<"Pose inference time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;

        //6 draw pose point in the img
        time_beg = time_stamp();       
        cv::Mat img = draw_line(imgM, allpoints);
        std::cout<<fileP<<std::endl; 
        cv::imwrite(fileP, img);
        time_end = time_stamp();
        std::cout<<"Draw line time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;
        img_num += 1;
    }
    double time_total_e=time_stamp();
    std::cout<<"Resize image and predict average time: "
             <<(time_total_e - time_total_s)/(1000*img_num)<<"ms"<<std::endl;


}