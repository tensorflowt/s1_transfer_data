#include <iostream>
#include <string> 

#include "opencv2/opencv.hpp"
#include "classifier.hpp"
#include "classifier_tf.hpp"
#include "sak_utils.hpp"

int parse_config(std::string path, CLSParam &param){
    cv::FileStorage fs(path.c_str(), cv::FileStorage::READ);
    param.deploy = (std::string)fs["NET_DEPLOY"];
    param.model = (std::string)fs["NET_MODEL"];
    param.label_map = (std::string)fs["LABEL_MAP"];
    param.net_w = (int)fs["NET_IN_W"];
    param.net_h = (int)fs["NET_IN_H"];
    if (!fs["NET_OUTPUT"].empty())
        param.output_name = (std::string)fs["NET_OUTPUT"];
    if (!fs["NET_INPUT"].empty())
        param.input_name = (std::string)fs["NET_INPUT"];
    
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
        "{th   t         |0     | threshold for detection, default 0}"
        "{bgr            |      | use bgr mode, default rgb mode}"
        "{@config        |.     | path to config file  }"
        "{@image_list    |.     | path to image file   }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("demo_classifier_image_list");
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    int mode = parser.get<int>("mode");
    int gpu_id = parser.get<int>("gpu");
    float th = parser.get<float>("th");
    bool use_rgb = !(parser.get<bool>("bgr"));

    std::string config = parser.get<std::string>(0);
    std::string imageP = parser.get<std::string>(1);
    
    //Read config file
    CLSParam param;
    parse_config(config, param);
    param.gpu_id = gpu_id;
	
    Classifier *clsH = 0;
    if (0==mode)
        clsH = new ClassifierTF();
    
    if (0 != clsH->init(param)){
        std::cout<<"Init Classifier failed"<<std::endl;
        return -1;
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
        std::cout<<line;
        imgM = cv::imread(line.c_str());
        
        if (0 != clsH->setImage(imgM, false, use_rgb)){
            std::cout<<"Set Image failed"<<std::endl;
            return -1;
        }
        
        time_beg = time_stamp();
        int id;
        float score;
        std::string name;
        int flag = clsH->predict(score, id, name);
        if (flag < 0){
            std::cout<<"Predict failed"<<std::endl;
            return -1;
        }
        time_end = time_stamp();
        std::cout<<","<<name<<","<<score<<std::endl;
        time_total += (time_end-time_beg)/1000;
        img_num += 1;
    }
    std::cout<<"Predict average time: "<<time_total/img_num<<"ms"<<std::endl;
    delete clsH;
    return 0;
}
