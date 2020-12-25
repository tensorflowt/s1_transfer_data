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

void draw_label(cv::Mat& image, cv::Point p, int color[3], std::string s) {
	//lable size:30 x slen*20+5
	int l_w = s.length() * 20 + 5;
	int l_h = 30;
	for (int h = p.x - l_h; h < p.x; h++)
	{
		uchar* current_row = image.ptr<uchar>(h);
		int x = p.y;
		while (x--!=0)
		{
			current_row++;
			current_row++;
			current_row++;
		}
		for (int w = p.y; w < p.y + l_w; w++)
		{
			*current_row++ = color[0];
			*current_row++ = color[1];
			*current_row++ = color[2];
		}
	}
	cv::putText(image, s, cv::Point(p.x + 5, p.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(0,0,0));
}


int main(int argc, char **argv){
    // command line parser
    const std::string keys =
    "{help h usage ? |      | print this message   }"
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
    
    ObjectDetectorTF_EFFIDET *detector = new ObjectDetectorTF_EFFIDET();

    if (0 != detector->init(param)){
        std::cout<<"Init Detector failed"<<std::endl;
        return -1;
    }
    time_end = time_stamp();
    std::cout<<"Init net time: "
             <<(time_end-time_beg)/1000<<"ms"<<std::endl;

    double time_total_s = time_stamp();
    cv::Mat imgM;
    // batch mode
    int max_batch_size = param.max_batch_size;
    int img_num = 0;
    std::vector<cv::Mat> batch_imgM;
    std::vector<std::string> batch_name;
    std::vector<std::string> img_path_list;
    while (std::getline(infile, line)){
        img_path_list.push_back(line.c_str());
        img_num ++;
    }

    // inference
    int cur_idx = 0;
    while (cur_idx < img_num){

        // 1. Load batch image
        batch_imgM.clear();
        batch_name.clear();
        time_beg = time_stamp();
        int buffer_size = std::min(max_batch_size,img_num - cur_idx);
        int cur_buffer_idx = 0;
        while (cur_buffer_idx < buffer_size)
        {
            imgM = cv::imread(img_path_list[cur_idx]);
            batch_imgM.push_back(imgM);
            batch_name.push_back(img_path_list[cur_idx]);

            cur_buffer_idx ++;
            cur_idx ++;
        }
        time_end = time_stamp();
        std::cout<<"Load image time: "
                <<(time_end-time_beg)/(1000*buffer_size)<<"ms"<<std::endl;

        // 2. Resize image
        /*
        time_beg = time_stamp();
        
        if (0 != detector->setBatchImage(batch_imgM,false)){
            std::cout<<"Set Image failed"<<std::endl;
            return -1;
        }
        time_end = time_stamp();
        std::cout<<"Resize image to target size time: "
                <<(time_end-time_beg)/(1000*buffer_size)<<"ms"<<std::endl;

        // 3. Forward
        time_beg = time_stamp();
        int img_rst_num = detector->predictBatch();
        time_end = time_stamp();
        std::cout<<"Average predict time in batch: " <<(time_end-time_beg)/(1000*buffer_size)<<"ms"<<std::endl;

        // 4. Draw image
        time_beg = time_stamp();
        for (int img_rst_idx=0; img_rst_idx<buffer_size; img_rst_idx++)
        {
            std::vector<ObjectDetection> obj_list;
            int status = detector->batchObject(img_rst_idx, obj_list);
            int obj_num = obj_list.size();
            cv::Mat rst_img = batch_imgM[img_rst_idx];
            for (int obj_idx=0; obj_idx<obj_num; obj_idx++)
            {
                ObjectDetection obj = obj_list[obj_idx];
                std::string obj_info = obj.getName() + ":" + to_string(obj.getScore());
                cv::Point top_left(obj.getRect().x,obj.getRect().y);
                cv::rectangle(rst_img,obj.getRect(),cv::Scalar(0,255,0),2);
                cv::putText(rst_img, obj_info, top_left, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
            }
            std::string save_path = outDir + '/' + batch_name[img_rst_idx].substr(batch_name[img_rst_idx].find_last_of("/\\") + 1);
            cv::imwrite(save_path,rst_img);

            //std::cout<<"Image idx:"<<img_rst_idx<<",object num:"<<obj_num<<std::endl;
        }
        time_end = time_stamp();
        std::cout<<"Average time for writing image: " <<(time_end-time_beg)/(1000*buffer_size)<<"ms"<<std::endl;
        */ 
    }
    double time_total_e=time_stamp();
    std::cout<<"Average time for each image [Read img + resize and copy to device + predict]: "
            <<(time_total_e - time_total_s)/(1000*img_num)<<"ms"<<std::endl;
    
    
    
    delete detector;
    return 0;
}
