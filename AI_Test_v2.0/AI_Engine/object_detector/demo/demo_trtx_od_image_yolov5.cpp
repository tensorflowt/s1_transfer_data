#include "stdio.h"
#include "stdlib.h"   
#include "time.h"  
#include <iostream>
#include <fstream>
#include <string> 
#include <chrono>
#include "opencv2/opencv.hpp"
#include "sak_utils.hpp"
#include "object_detector_trtx_yolov5.hpp"
/*
#include "object_detector_tf.hpp"
#include "object_detector_tflite.hpp"
#include "object_detector_darknet.hpp"
#include "object_detector_mrcnn_tf.hpp"
#include "object_detector_tf_effidet.hpp"
#include "object_detector_torch.hpp"
*/


int parse_config(std::string path, ODParam &param){
    cv::FileStorage fs(path.c_str(), cv::FileStorage::READ);
    param.deploy = (std::string)fs["NET_DEPLOY"];
    param.model = (std::string)fs["NET_MODEL"];
    param.label_map = (std::string)fs["LABEL_MAP"];
    param.wts_file = (std::string)fs["WTS_FILE"];
    param.net_w = (int)fs["NET_IN_W"];
    param.net_h = (int)fs["NET_IN_H"];
    param.class_num = (int)fs["CLASS_NUM"];
    param.max_batch_size = (int)fs["MAX_BATCH_SIZE"];
    param.nms_thresh = (float)fs["NMS_THRESH"];
    param.conf_thresh = (float)fs["CONF_THRESH"];
    std::cout << "conf_thresh: " << param.conf_thresh << std::endl;
    std::cout << "nms_thresh: " << param.nms_thresh << std::endl;


    // get the absolute path
    int found = path.find_last_of("/\\");
    param.deploy = path.substr(0,found) + "/" + param.deploy;
    param.model = path.substr(0,found) + "/" + param.model;
    param.label_map = path.substr(0,found) + "/" + param.label_map;
    param.wts_file = path.substr(0,found) + "/" + param.wts_file;
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

    std::cout << "mode: " << mode << std::endl;
    std::cout << "gpu: " << gpu_id << std::endl;
    std::string config = parser.get<std::string>(0);
    std::string fileP = parser.get<std::string>(1);
    std::string outDir = parser.get<std::string>(2);
    std::string line;

    std::ifstream infile(fileP.c_str());
    if(!infile.is_open()){
       std::cout<<"Open image list file failed"<<std::endl;
       return -1;
    }

    // 1. Read config file
    auto time_beg1 = std::chrono::system_clock::now();
    ODParam param;

#if 1
    parse_config(config, param);
#else
    param.deploy = (std::string)fs["NET_DEPLOY"];
    param.model = "D:\\Project\\AI_Test\\build\\AI_Engine\\object_detector\\yolov5\\yolov5s.engine";
    param.label_map = (std::string)fs["LABEL_MAP"];
    param.net_w = 608;
    param.net_h = 608;
    param.conf_thresh = 0.8;
    param.nms_thresh = 0.5;
    param.max_batch_size = (int)fs["MAX_BATCH_SIZE"];
#endif

    param.gpu_id = gpu_id;
    auto time_end1 = std::chrono::system_clock::now();
    std::cout << "Load config time : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end1 - time_beg1).count() << "ms" << std::endl;

    // 2. Init network and warm-up
    auto time_beg2 = std::chrono::system_clock::now();

    ObjectDetector *detector = 0;
    if (6==mode)
    //    detector = new ObjectDetectorTF();
    //else if (1==mode)
    //    detector = new ObjectDetectorTflite();
    //else if (2==mode)
    //    detector = new ObjectDetectorDarknet();
    //else if (3==mode)
    //    detector = new ObjectDetectorMrcnnTF();
    //else if (4==mode)
    //    detector = new ObjectDetectorTF_EFFIDET();
    //else if (5==mode)
    //    detector = new ObjectDetectorTorch();
    //else if (6==mode)
        detector = new ObjectDetectorTRTX_YOLOV5();

    if (0 != detector->init(param)){
        std::cout<<"Init Detector failed"<<std::endl;
        return -1;
    }
    
    auto time_end2 = std::chrono::system_clock::now();
    std::cout << "Init net time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end2 - time_beg2).count() << "ms" << std::endl;

    double time_total_s = time_stamp();
    int img_num = 0;
    cv::Mat imgM;
    int obj_num = 0;
    float time_cost_total = 0;

    while (std::getline(infile,line)){

        // 1. Load image
        //auto time_beg3 = std::chrono::system_clock::now();
        std::string img_path = line.c_str();
        std::cout << "Image path: "<< img_path << std::endl;
        imgM = cv::imread(img_path);
        //auto time_end3 = std::chrono::system_clock::now();
        //float time_cost3 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end3 - time_beg3).count();
        //time_cost_total += time_cost3;
        //std::cout << "Load image time: " << time_cost3 << "ms" << std::endl;

        // 2. Resize image
        auto time_beg4 = std::chrono::system_clock::now();
        if (0 != detector->setImage(imgM,false)){
            std::cout<<"Set Image failed"<<std::endl;
            return -1;
        }
        //auto time_end4 = std::chrono::system_clock::now();
        //float time_cost4 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end4 - time_beg4).count();
        //time_cost_total += time_cost4;
        //std::cout << "Resize image to target size time: " << time_cost4 << "ms" << std::endl;

        // 3. Forward
        auto time_beg5 = std::chrono::system_clock::now();
        obj_num = detector->predict();
        if (obj_num < 0){
            std::cout<<"Predict failed"<<std::endl;
            return -1;
        }
        auto time_end5 = std::chrono::system_clock::now();
        float time_cost5 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end5 - time_beg5).count();
        time_cost_total += time_cost5;
        std::cout << "Predict time [set image to tensor + do predict + nms + coordinate transformation]: " << time_cost5 << "ms" << std::endl;

        // 4. Draw image
        //auto time_beg6 = std::chrono::system_clock::now();
        ObjectDetection obj;
        cv::Mat rst_img = imgM;
        for (int obj_idx=0; obj_idx<obj_num; obj_idx++){
            int status = detector->object(obj_idx, obj);
            std::string obj_info = obj.getName() + ":" + to_string(obj.getScore());
            cv::Point top_left(obj.getRect().x,obj.getRect().y);
            cv::rectangle(rst_img,obj.getRect(),cv::Scalar(0,255,0),2);
            cv::putText(rst_img, obj_info, top_left, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
        }
        std::string save_path = outDir + '/' + img_path.substr(img_path.find_last_of("/\\") + 1);
        cv::imwrite(save_path,rst_img);
        //auto time_end6 = std::chrono::system_clock::now();
        //float time_cost6 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end6 - time_beg6).count();
        //time_cost_total += time_cost6;
        //std::cout << "Writing image time: " << time_cost6 << "ms" << std::endl;

        img_num += 1;
    }
    double time_total_e=time_stamp();
    std::cout<<"Average time for each image [set image to tensor + do predict + nms + coordinate transformation]: "
            <<(time_cost_total)/(img_num)<<"ms"<<std::endl;

    
    delete detector;
    return 0;
}
