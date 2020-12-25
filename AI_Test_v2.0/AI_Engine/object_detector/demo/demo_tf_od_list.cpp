#include <iostream>
#include <string> 
#include <fstream>

#include "opencv2/opencv.hpp"
#include "object_detection.hpp"
#include "object_detector.hpp"
#include "object_detector_tf.hpp"
#include "object_detector_tflite.hpp"
#include "object_detector_darknet.hpp"
#include "object_detector_mrcnn_tf.hpp"

#include "sak_utils.hpp"

int parse_config(std::string path, ODParam &param){
    cv::FileStorage fs(path.c_str(), cv::FileStorage::READ);
    param.deploy = (std::string)fs["NET_DEPLOY"];
    param.model = (std::string)fs["NET_MODEL"];
    param.label_map = (std::string)fs["LABEL_MAP"];
    param.net_w = (int)fs["NET_IN_W"];
    param.net_h = (int)fs["NET_IN_H"];

    // get the absolute path
    int found = path.find_last_of("/\\");
    param.deploy = path.substr(0,found) + "/" + param.deploy;
    param.model = path.substr(0,found) + "/" + param.model;
    param.label_map = path.substr(0,found) + "/" + param.label_map;
    return 0;
}

int main(int argc, char **argv){
    // command line parser
    const std::string keys =
    "{help h usage ? |      | print this message   }"
    "{mode m         |0     | 0 for tensorflow, 1 for tflite, 2 for darknet}"
    "{gpu  g         |-1    | gpu id, default -1}"
    "{thresh  t      |0     | threshold for detection}"        
    "{@config        |.     | path to config file  }"
    "{@image_list    |.     | path to image list file   }"
    ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("demo_tf_od_list");
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    int mode = parser.get<int>("mode");
    int gpu_id = parser.get<int>("gpu");
    float th = parser.get<float>("thresh");
    std::string config = parser.get<std::string>(0);
    std::string imageP = parser.get<std::string>(1);

    //Read config file
    ODParam param;
    parse_config(config, param);
    param.gpu_id = gpu_id;
    ObjectDetector *detector = 0;
    if (0==mode)
        detector = new ObjectDetectorTF();
    else if (1==mode)
        detector = new ObjectDetectorTflite();
    else if (2==mode)
        detector = new ObjectDetectorDarknet();
    else if (3==mode)
        detector = new ObjectDetectorMrcnnTF();
	
    if (0 != detector->init(param)){
        std::cout<<"Init Detector failed"<<std::endl;
        return -1;
    } else {
        std::cout<<"Init Detector success"<<std::endl;
    }

    // load image list
    std::string line;
    std::ifstream infile(imageP.c_str());
    if(!infile.is_open()){
        std::cout<<"Open image list file failed"<<std::endl;
        return -1;
    }
    
    double time_beg, time_end, time_total=0;
    int img_num = 0;
    cv::Mat imgM;
    while (std::getline(infile, line)){
        std::cout<<line<<std::endl;
        imgM = cv::imread(line.c_str());
        
        time_beg = time_stamp();
        if (0 != detector->setImage(imgM)){
            std::cout<<"Set Image failed"<<std::endl;
            return -1;
        }
        
        int obj_num = detector->predict();
        if (obj_num < 0){
            std::cout<<"Predict failed"<<std::endl;
            return -1;
        }
        time_end = time_stamp();
        time_total += (time_end-time_beg)/1000;
        
        for (int i=0; i<obj_num; i++){
            ObjectDetection obj;
            if (0 != detector->object(i, obj))
                std::cout<<"Get object failed: "<<i<<std::endl;
            
            char label[1024] = {0};
            std::string name = obj.getName();
            cv::Rect roi = obj.getRect();
            if (obj.getScore() < th)
                continue;
            sprintf(label, "%s:%0.1f", name.c_str(), obj.getScore());
            cv::rectangle(imgM, roi, cv::Scalar(0, 255, 255));
            cv::putText(imgM, label, cv::Point(roi.x, roi.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
        }

        int found = line.find_last_of("/\\");
        char saveP[1024] = {0};
        sprintf(saveP, "res_%s", line.substr(found+1, line.size()).c_str());
        cv::imwrite(saveP, imgM);
    }
    std::cout<<"Predict average time: "
             <<time_total/img_num<<"ms"<<std::endl;
    //cv::imshow("result", imgM);
    //cv::waitKey(0);    
    delete detector;
    return 0;
}
