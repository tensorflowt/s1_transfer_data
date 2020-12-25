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
#include "object_detector_tf_effidet.hpp"
#include "object_detector_torch.hpp"
#include "object_detector_trtx_yolov5.hpp"


#include "sak_utils.hpp"

int parse_config(std::string path, ODParam &param){
    cv::FileStorage fs(path.c_str(), cv::FileStorage::READ);
    param.deploy = (std::string)fs["NET_DEPLOY"];
    param.model = (std::string)fs["NET_MODEL"];
    param.label_map = (std::string)fs["LABEL_MAP"];
    param.net_w = (int)fs["NET_IN_W"];
    param.net_h = (int)fs["NET_IN_H"];
    param.max_batch_size = (int)fs["MAX_BATCH_SIZE"];

    // get the absolute path
    int found = path.find_last_of("/\\");
    param.deploy = path.substr(0,found) + "/" + param.deploy;
    param.model = path.substr(0,found) + "/" + param.model;
    param.label_map = path.substr(0,found) + "/" + param.label_map;
    return 0;
}

std::string to_string(float val) {
    char buf[200];
    sprintf(buf, "%.3f", val);
    return std::string(buf);
}

int main(int argc, char **argv){
    // command line parser
    const std::string keys =
    "{help h usage ? |      | print this message   }"
    "{mode m         |0     | 0 for tensorflow, 1 for tflite, 2 for darknet}"
    "{gpu  g         |-1    | gpu id, default -1}"
    "{th   t         |0     | threshold for detection, default 0}"
    "{@config        |.     | path to config file  }"
    "{@image list    |.     | path to image list file   }"
    "{@output dir    |.     | directory of saving result images   }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("demo_tf_od_image");
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    int mode = parser.get<int>("mode");
    int gpu_id = parser.get<int>("gpu");
    float th = parser.get<float>("th");


    std::string config = parser.get<std::string>(0);
    /*
    std::string image = parser.get<std::string>(1);
    cv::Mat imgM = cv::imread(image);
    */

    std::string fileP = parser.get<std::string>(1);
    std::string outDir = parser.get<std::string>(2);
    std::string line;
    std::ifstream infile(fileP.c_str());
    if(!infile.is_open()){
        std::cout<<"Open image list file failed"<<std::endl;
        return -1;
    }

    // 1. Read config file
    double time_beg = time_stamp();
    ODParam param;
    parse_config(config, param);
    param.gpu_id = gpu_id;
    double time_end = time_stamp();
    std::cout<<"Load config time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;


    // 2. Init network and warm-up
    time_beg = time_stamp();
    
    ObjectDetector *detector = 0;
    if (0==mode)
        detector = new ObjectDetectorTF();
    else if (1==mode)
        detector = new ObjectDetectorTflite();
    else if (2==mode)
        detector = new ObjectDetectorDarknet();
    else if (3==mode)
        detector = new ObjectDetectorMrcnnTF();
    else if (4==mode)
        detector = new ObjectDetectorTF_EFFIDET();
    else if (5==mode)
        detector = new ObjectDetectorTorch();
    else if (6==mode)
        detector = new ObjectDetectorTRTX_YOLOV5();
	
    if (0 != detector->init(param)){
        std::cout<<"Init Detector failed"<<std::endl;
        return -1;
    }
    time_end = time_stamp();
    std::cout<<"Init net time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;

    double time_total_s = time_stamp();
    int img_num = 0;
    cv::Mat imgM;
    int obj_num = 0;
    while (std::getline(infile, line)){

        // 1. Load image
        time_beg = time_stamp();
        std::string img_path = line.c_str();
        imgM = cv::imread(img_path);
        time_end = time_stamp();
        std::cout<<"Load image time: "
                <<(time_end-time_beg)/1000<<"ms"<<std::endl;

        // 2. Resize image
        time_beg = time_stamp();
        if (0 != detector->setImage(imgM,false)){
            std::cout<<"Set Image failed"<<std::endl;
            return -1;
        }
        time_end = time_stamp();
        std::cout<<"Resize image to target size time: "
                <<(time_end-time_beg)/1000<<"ms"<<std::endl;

        // 3. Forward
        time_beg = time_stamp();
        obj_num = detector->predict();
        if (obj_num < 0){
            std::cout<<"Predict failed"<<std::endl;
            return -1;
        }
        time_end = time_stamp();
        std::cout<<"Predict time: "
            <<(time_end-time_beg)/1000<<"ms"<<std::endl;
        
        /*
        // 4. Draw image
        time_beg = time_stamp();
        ObjectDetection obj;
        cv::Mat rst_img = imgM;
        for (int obj_idx=0; obj_idx<obj_num; obj_idx++)
        {
            int status = detector->object(obj_idx, obj);
            std::string obj_info = obj.getName() + ":" + to_string(obj.getScore());
            cv::Point top_left(obj.getRect().x,obj.getRect().y);
            cv::rectangle(rst_img,obj.getRect(),cv::Scalar(0,255,0),2);
            cv::putText(rst_img, obj_info, top_left, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
        }
        std::string save_path = outDir + '/' + img_path.substr(img_path.find_last_of("/\\") + 1);
        cv::imwrite(save_path,rst_img);
        time_end = time_stamp();
        std::cout<<"Writing image time: "
            <<(time_end-time_beg)/1000<<"ms"<<std::endl;
        */
        img_num += 1;
    }
    double time_total_e=time_stamp();
    std::cout<<"Average time for each image [Read img + resize and copy to device + predict]: "
            <<(time_total_e - time_total_s)/(1000*img_num)<<"ms"<<std::endl;

    
    delete detector;
    return 0;
}
