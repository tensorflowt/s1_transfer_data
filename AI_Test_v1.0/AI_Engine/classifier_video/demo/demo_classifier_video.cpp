#include <iostream>
#include <string> 

#include "opencv2/opencv.hpp"
#include "classifier_video.hpp"
#include "classifier_video_i3d.hpp"
#include "sak_utils.hpp"

int parse_config(std::string path, VidCLSParam &param){
    cv::FileStorage fs(path.c_str(), cv::FileStorage::READ);
    param.deploy = (std::string)fs["NET_DEPLOY"];
    param.model = (std::string)fs["NET_MODEL"];
    param.label_map = (std::string)fs["LABEL_MAP"];
    param.net_w = (int)fs["NET_IN_W"];
    param.net_h = (int)fs["NET_IN_H"];
    param.net_t = (int)fs["NET_IN_T"];
    param.input_node = (std::string)fs["INPUT_NODE"];
    param.output_node = (std::string)fs["OUTPUT_NODE"];

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
    "{mode m         |0     | 0 for tensorflow }"
    "{gpu  g         |-1    | gpu id, default -1}"
    "{th   t         |0     | threshold for detection, default 0}"
    "{@config        |.     | path to config file  }"
    "{@image         |.     | path to video file   }"
    ;

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
    std::string image = parser.get<std::string>(1);
    std::vector<cv::Mat> imgV;
    
    //opencv read video
    cv::VideoCapture cap(image.c_str()); 
    if(!cap.isOpened()){
        std::cout << "Open video failed"<<std::endl;
        return -1;
    }

    cv::Mat frame;
    while(true){
        cap >> frame;
        if (frame.empty())
            break;
        imgV.push_back(frame.clone());
    }
    cap.release();

    
    //Read config file
    VidCLSParam param;
    parse_config(config, param);
    param.gpu_id = gpu_id;
	
    ClassifierVid *clsH = 0;
    if (0==mode)
        clsH = new ClassifierVidI3D();
    
    if (0 != clsH->init(param)){
        std::cout<<"Init Classifier failed"<<std::endl;
        return -1;
    }
    
    if (0 != clsH->setImage(imgV)){
        std::cout<<"Set Image failed"<<std::endl;
        return -1;
    }

    double time_beg = time_stamp();
    int id;
    float score;
    std::string name;
    int flag = clsH->predict(score, id, name);
    if (flag < 0){
        std::cout<<"Predict failed"<<std::endl;
        return -1;
    }
    double time_end = time_stamp();
    std::cout<<name<<", "<<score<<std::endl;
    std::cout<<"Predict inference time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;

    delete clsH;
    return 0;
}
