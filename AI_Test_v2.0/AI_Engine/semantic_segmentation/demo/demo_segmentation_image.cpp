#include <iostream>
#include <string> 

#include "opencv2/opencv.hpp"
#include "semantic_segmentation.hpp"
#include "semantic_segmentation_tf.hpp"
#include "semantic_segmentation_torch.hpp"
#include "sak_utils.hpp"

int parse_config(std::string path, SegParam &param){
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
    "{mode m         |0     | 0 for tensorflow, 1 for torch}"
    "{gpu  g         |-1    | gpu id, default -1}"
    "{@config        |.     | path to config file  }"
    "{@image         |.     | path to image file   }"
    ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("demo_segmentation_image");
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    int mode = parser.get<int>("mode");
    int gpu_id = parser.get<int>("gpu");

    std::string config = parser.get<std::string>(0);
    std::string image = parser.get<std::string>(1);
    cv::Mat imgM = cv::imread(image);
    
    //Read config file
    SegParam param;
    parse_config(config, param);
    param.gpu_id = gpu_id;
	
    SemanticSeg *handle = 0;
    if (0==mode)
        handle = new SemanticSegTF();
    else if (1==mode)
        handle = new SemanticSegT7();
    else
        return -1;
    
    if (0 != handle->init(param)){
        std::cout<<"Init Detector failed"<<std::endl;
        return -1;
    }
    
    if (0 != handle->setImage(imgM, true)){
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
    
    cv::Mat mask;
    handle->segData(mask, imgM.rows, imgM.cols);
    
    // draw seg on image
    unsigned char *pImg = imgM.data;
    unsigned char *pMask = mask.data;
    for (int i=0; i<imgM.cols*imgM.rows; i++){
        if (pMask[i]>0){
            pImg[i*3+2] = int(pImg[i*3+2]*0.6 + 0.4*255);
        }
    }

    cv::imwrite("./result.jpg", imgM);
    //cv::imshow("result", imgM);
    //cv::waitKey(0);    
    delete handle;
    return 0;
}
