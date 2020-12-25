#include <iostream>
#include <string> 

#include "opencv2/opencv.hpp"
#include "classifier.hpp"
#include "classifier_mxnet.hpp"
#include "sak_utils.hpp"

int parse_config(std::string path, CLSParam &param){
    cv::FileStorage fs(path.c_str(), cv::FileStorage::READ);
    std::cout<<path<<std::endl;
    param.deploy = (std::string)fs["NET_DEPLOY"];
    param.model = (std::string)fs["NET_MODEL"];
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
    "{gpu  g         |-1    | gpu id, default -1}"
    "{@config        |.     | path to config file  }"
    "{@image         |.     | path to image file   }"
    ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("demo_extract_feature_image");
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    int gpu_id = parser.get<int>("gpu");
    
    std::string config = parser.get<std::string>(0);
    std::string image = parser.get<std::string>(1);
    cv::Mat imgM = cv::imread(image);
    
    //Read config file
    CLSParam param;
    parse_config(config, param);
    param.gpu_id = gpu_id;
	
    ClassifierMXNET *clsH = 0;

    clsH = new ClassifierMXNET();
    if (0 != clsH->init(param)){
        std::cout<<"Init Classifier failed"<<std::endl;
        return -1;
    }
    
    if (0 != clsH->setImage(imgM, true)){
        std::cout<<"Set Image failed"<<std::endl;
        return -1;
    }

    double time_beg = time_stamp();
    cv::Mat fea;
    int flag = clsH->extractFeature(fea);
    if (flag < 0){
        std::cout<<"Extract feature failed"<<std::endl;
        return -1;
    }
    std::cout<<fea.cols<<" "<<fea.rows<<std::endl;
    for (int r=0; r<fea.rows; r++){
        for (int c=0; c<fea.cols; c++){
            std::cout<<fea.at<float>(r, c)<<std::endl;
        }
    }
    double time_end = time_stamp();
    std::cout<<"Predict inference time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;
    
    delete clsH;
    return 0;
}
