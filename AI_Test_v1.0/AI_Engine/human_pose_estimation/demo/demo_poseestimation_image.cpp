#include <iostream>
#include <string> 

#include "opencv2/opencv.hpp"
#include "two_stage_pose_estimation.hpp"
#include "pose_estimation_alphapose.hpp"
#include "sak_utils.hpp"


cv::Mat draw_line(cv::Mat img, Valpoints valpoints){
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
		Valpoints hum = valpoints;
		std::vector<int>  part_line;
		std::vector<float> scores = hum.scores;
		std::vector<Point2f> points = hum.preds;
		float new_score = (hum.scores[5] + hum.scores[6]) / 2;
		Point2f new_point = Point2f((int)((points[5].x + points[6].x) / 2),
			(int)((points[5].y + points[6].y) / 2));
		std::cout << "scores" << new_score << "points" << new_point << std::endl;
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
	return img;
}



int parse_config(std::string path, HpesParam &param){
    cv::FileStorage fs(path.c_str(), cv::FileStorage::READ);
    param.deploy = (std::string)fs["NET_DEPLOY"];
    param.model = (std::string)fs["NET_MODEL"];
 //   param.label_map = (std::string)fs["LABEL_MAP"];
    param.net_w = (int)fs["NET_IN_W"];
    param.net_h = (int)fs["NET_IN_H"];

    // get the absolute path
    int found = path.find_last_of("/\\");
    param.deploy = path.substr(0,found) + "/" + param.deploy;
    param.model = path.substr(0,found) + "/" + param.model;
 //   param.label_map = path.substr(0,found) + "/" + param.label_map;
    return 0;
}

int main(int argc, char **argv){
    // command line parser
    const std::string keys =
    "{help h usage ? |      | print this message   }"
    "{mode m         |0     | 0 for tensorflow, 1 for torch}"
    "{gpu  g         |-1    | gpu id, default -1}"
    "{@config        |.     | path to config file  }"
    "{@image         |.     | path to image file   }"
    ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("demo_poseestimation_image");
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    int mode = parser.get<int>("mode");
    int gpu_id = parser.get<int>("gpu");

    std::string config = parser.get<std::string>(0);
    std::string image = parser.get<std::string>(1);
    cv::Mat imgM = cv::imread(image);
    // the rect should be given
    cv::Rect  rect(372, 33, 74, 259);
    
    //Read config file
    HpesParam param;
    parse_config(config, param);
    param.gpu_id = gpu_id;
	

    HPestimation *handle = 0;
    if (1==mode)
        handle = new EstimAlphaPose();
    else
        return -1;
    
    if (0 != handle->init(param)){
        std::cout<<"Init Detector failed"<<std::endl;
        return -1;
    }

    

    if (0 != handle->setImage(imgM, rect)){
        std::cout<<"Set Image failed"<<std::endl;
        return -1;
    }

    double time_beg = time_stamp();
    if (0 != handle->predict()){
        std::cout<<"Predict failed"<<std::endl;
        return -1;
    }
    double time_end = time_stamp();
    std::cout<<"Predict inference time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;
    
    Valpoints points; 
    handle->posData(points);
    
    // draw landmark on image
    cv::Mat img = draw_line(imgM, points);
    cv::imwrite("./result.jpg", img);
    //cv::imshow("result", imgM);
    //cv::waitKey(0);    
    delete handle;
    return 0;
}