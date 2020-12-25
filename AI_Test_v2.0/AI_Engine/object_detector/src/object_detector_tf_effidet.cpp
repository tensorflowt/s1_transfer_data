#ifdef ENABLE_TENSORFLOW
#include <string>
#include "tensorflow/core/graph/default_device.h"
#include "object_detector_tf_effidet.hpp"
#include "sak_utils.hpp"

using namespace std;
using tensorflow::Status;

ObjectDetectorTF_EFFIDET::ObjectDetectorTF_EFFIDET(){
}
    
ObjectDetectorTF_EFFIDET::~ObjectDetectorTF_EFFIDET(){
    m_objs.clear();
    // for batch mode
    m_batch_imgs.clear();
    m_origin_img_size.clear();
    m_resize_scale.clear();
    m_batch_objs.clear();
}


int ObjectDetectorTF_EFFIDET::init(const ODParam &param){
    if (0 != readLabelMap(param.label_map))
        return -1;

    m_net_w = param.net_w;
    m_net_h = param.net_h;
    // for batch mode
    m_max_batch_size = param.max_batch_size;
    
    // Load tensorflow model
    LOG_INFO("Loading graph: "<<param.model);
    tensorflow::GraphDef graph_def;
    Status status = ReadBinaryProto(tensorflow::Env::Default(), param.model, &graph_def);
    if (!status.ok()) {
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }

    auto options = tensorflow::SessionOptions();
    //auto* config = &options.config; // not work to restrict 1 gpu
    //(*config->mutable_device_count())["GPU"] = 1;
    if (param.gpu_id != -1)
        //options.config.mutable_gpu_options()->set_visible_device_list(std::to_string(param.gpu_id));
        tensorflow::graph::SetDefaultDevice("/device:GPU:" + std::to_string(param.gpu_id), &graph_def);
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(param.gpu_fraction);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    options.config.set_allow_soft_placement(true);
    
    m_session.reset(tensorflow::NewSession(options));
    status = m_session->Create(graph_def);
    if (!status.ok()) {
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }
    
    // do warmup
    LOG_INFO("Start warm up: ");
    //warmUp();
    return 0;
}

// set images for batch mode
int ObjectDetectorTF_EFFIDET::setBatchImage(const std::vector<cv::Mat> &batch_imgs, bool keep_aspect){
    // clear buffer image and info
    m_batch_imgs.clear();
    m_origin_img_size.clear();
    m_resize_scale.clear();
    m_objs.clear();
    m_batch_objs.clear();

    // set current batch size 
    m_cur_batch_size = batch_imgs.size();
    if (m_cur_batch_size > m_max_batch_size)
    {
        LOG_ERROR("Input image number exceed BATCH_SIZE set in config file!");
        return -1;
    }
    for (int idx = 0; idx < m_cur_batch_size; idx++)
    {
        cv::Mat img = batch_imgs[idx];

        // create resized image
        cv::Mat cur_img;
        if (true == keep_aspect){
            m_scale_x = float(img.cols)/m_net_w;
            m_scale_y = float(img.rows)/m_net_h;
            
            int scale_img_w = m_net_w;
            int scale_img_h = m_net_h;
            if (m_scale_x > m_scale_y){
                m_scale_y = m_scale_x;
                scale_img_h = int(img.rows/m_scale_y)/2*2;
            }
            else {
                m_scale_x = m_scale_y;
                scale_img_w = int(img.cols/m_scale_x)/2*2;
            }
            cur_img = cv::Mat(m_net_h, m_net_w, CV_8UC3, cv::Scalar(127, 127, 127));
            cv::Rect roi(0,0,scale_img_w, scale_img_h);
            cv::resize(img, cur_img(roi),
                    cv::Size(scale_img_w, scale_img_h), 0, 0,
                    cv::INTER_LINEAR);
            cv::cvtColor(cur_img, cur_img, cv::COLOR_BGR2RGB);
        }
        else {
            cv::resize(img, cur_img,
                    cv::Size(m_net_w, m_net_h), 0, 0,
                    cv::INTER_LINEAR);
            m_scale_x = float(img.cols)/m_net_w;
            m_scale_y = float(img.rows)/m_net_h;
            cv::cvtColor(cur_img, cur_img, cv::COLOR_BGR2RGB);
        }
        // save origin shape and resize scaling to m_origin_img_size and m_resize_scale
        m_origin_img_size.push_back(std::pair<int,int>(int(img.cols),int(img.rows)));
        m_resize_scale.push_back(std::pair<float,float>(m_scale_x,m_scale_y));
        m_batch_imgs.push_back(cur_img);
    }
    return 0;
}

int ObjectDetectorTF_EFFIDET::setTensor(tensorflow::Tensor& in_tensor){
    unsigned char* pData = m_img.data;
    auto outputTensorMapped = in_tensor.tensor<unsigned char, 4>();
    for (int h=0; h<m_img.rows; ++h){
        for (int w=0; w<m_img.cols; ++w){
            //outputTensorMapped(0, h, w, 0) = (float(pData[0])/255.0f - 0.485f)/0.229; // R
            //outputTensorMapped(0, h, w, 1) = (float(pData[1])/255.0f - 0.456f)/0.224; // G
            //outputTensorMapped(0, h, w, 2) = (float(pData[2])/255.0f - 0.406f)/0.225; // B
            outputTensorMapped(0, h, w, 0) = float(pData[0]); // R
            outputTensorMapped(0, h, w, 1) = float(pData[1]); // G
            outputTensorMapped(0, h, w, 2) = float(pData[2]); // B
            pData += 3;
        }
    }
    return 0;
}

int ObjectDetectorTF_EFFIDET::setBatchTensor(tensorflow::Tensor& in_tensor){
    for (int idx = 0; idx < m_cur_batch_size; idx++)
    {
        cv::Mat cur_img = m_batch_imgs[idx];
        unsigned char* pData = cur_img.data;
        auto outputTensorMapped = in_tensor.tensor<unsigned char, 4>();
        for (int h=0; h<cur_img.rows; ++h){
            for (int w=0; w<cur_img.cols; ++w){
                outputTensorMapped(idx, h, w, 0) = float(pData[0]); // R
                outputTensorMapped(idx, h, w, 1) = float(pData[1]); // G
                outputTensorMapped(idx, h, w, 2) = float(pData[2]); // B
                pData += 3;
            }
        }
    }
    return 0;
}



int ObjectDetectorTF_EFFIDET::predict(){
    m_objs.clear();
    m_batch_objs.clear();
    
    // set image to tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_UINT8,
                                    tensorflow::TensorShape({1, m_net_h, m_net_w, 3}));
    setTensor(input_tensor);

    // do predict
    std::vector<tensorflow::Tensor> outputs;
    std::string input_name = "image_arrays:0";
    std::vector<std::string> output_name = {"detections:0"};

    /*
    virtual Status tensorflow::Session::Run(
        const std::vector< std::pair< string, Tensor > > &inputs, 
        const std::vector< string > &output_tensor_names, 
        const std::vector< string > &target_node_names, 
        std::vector< Tensor > *outputs
        )=0
    */

    double time_beg = time_stamp();
    Status status = m_session->Run({{input_name, input_tensor}},
                                   output_name, {}, &outputs);
    if (!status.ok()){
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }
    double time_end = time_stamp();
    // Time
    std::cout<<"Session run time: "
             <<(time_end-time_beg)/(1000)<<"ms"<<std::endl;

 
    auto data = outputs[0].flat<float>();
    int det_num = data.size()/7;
    for (int i = 0; i < det_num; i++){
        float x1 = data(i*7 + 2)*m_scale_x;
        float y1 = data(i*7 + 1)*m_scale_y;
        float x2 = data(i*7 + 4)*m_scale_x;
        float y2 = data(i*7 + 3)*m_scale_y;
        x1 = std::min(std::max(x1, 0.f), m_origin_img_w-1.f);
        y1 = std::min(std::max(y1, 0.f), m_origin_img_h-1.f);
        x2 = std::min(std::max(x2, 0.f), m_origin_img_w-1.f);
        y2 = std::min(std::max(y2, 0.f), m_origin_img_h-1.f);
        if((x2 - x1 < 10) || (y2 - y1 < 10))
        {
            continue;
        }
        ObjectDetection obj(x1, y1, x2-x1, y2-y1,
                            data(i*7 + 5), int(data(i*7 + 6)), m_label_map[int(data(i*7 + 6))]);
        m_objs.push_back(obj);
    }
    
    /*
    int num = num_dets(0);
    for (int i=0; i<num; i++){
        float x1 = boxes(0, i, 1)*m_scale_x*m_net_w;
        float y1 = boxes(0, i, 0)*m_scale_y*m_net_h;
        float x2 = boxes(0, i, 3)*m_scale_x*m_net_w;
        float y2 = boxes(0, i, 2)*m_scale_y*m_net_h;
        x1 = std::min(std::max(x1, 0.f), m_origin_img_w-1.f);
        y1 = std::min(std::max(y1, 0.f), m_origin_img_h-1.f);
        x2 = std::min(std::max(x2, 0.f), m_origin_img_w-1.f);
        y2 = std::min(std::max(y2, 0.f), m_origin_img_h-1.f);
        ObjectDetection obj(x1, y1, x2-x1, y2-y1,
                            scores(i), int(classes(i)), m_label_map[int(classes(i))]);
        m_objs.push_back(obj);
    }
    */
    return m_objs.size();
}

int ObjectDetectorTF_EFFIDET::predictBatch(){

    // create tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_UINT8,
                                    tensorflow::TensorShape({m_max_batch_size, m_net_h, m_net_w, 3}));
    // copy image datum to tensor
    setBatchTensor(input_tensor);

    // do predict
    std::vector<tensorflow::Tensor> outputs;
    std::string input_name = "image_arrays:0";
    std::vector<std::string> output_name = {"detections:0"};

    double time_beg = time_stamp();
    Status status = m_session->Run({{input_name, input_tensor}},
                                   output_name, {}, &outputs);
    if (!status.ok()){
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }
    double time_end = time_stamp();
    // Time
    std::cout<<"Session run time: "
             <<(time_end-time_beg)/(1000*m_cur_batch_size)<<"ms"<<std::endl;

    // Save single image objects
    std::vector<ObjectDetection> cur_objs;


    auto data = outputs[0];//.flat<float>();
    auto result  = outputs[0].tensor<float, 3>();
    /*
    std::cout<<"outputs[0] dims 0:"<<data.shape().dim_size(0)<<std::endl;
    std::cout<<"outputs[0] dims 1:"<<data.shape().dim_size(1)<<std::endl;
    std::cout<<"outputs[0] dims 2:"<<data.shape().dim_size(2)<<std::endl;
    data.shape().dim_size(0) dims 0:8
    data.shape().dim_size(1) dims 1:100
    data.shape().dim_size(2) dims 2:7
    */

    for (int img_idx=0; img_idx<m_cur_batch_size; img_idx++)
    {   
        cur_objs.clear();
        int det_num = data.shape().dim_size(1);
        float scale_x = m_resize_scale[img_idx].first;
        float scale_y = m_resize_scale[img_idx].second;
        int orig_x = m_origin_img_size[img_idx].first;
        int orig_y = m_origin_img_size[img_idx].second;
        for (int i = 0; i < det_num; i++){
            float prob = result(img_idx,i,5);
            int category = result(img_idx,i,6);
            if (prob < 1e-2)
            {
                break;
            }
            float x1 = result(img_idx,i,2)*scale_x;
            float y1 = result(img_idx,i,1)*scale_y;
            float x2 = result(img_idx,i,4)*scale_x;
            float y2 = result(img_idx,i,3)*scale_y;
            x1 = std::min(std::max(x1, 0.f), orig_x-1.f);
            y1 = std::min(std::max(y1, 0.f), orig_y-1.f);
            x2 = std::min(std::max(x2, 0.f), orig_x-1.f);
            y2 = std::min(std::max(y2, 0.f), orig_y-1.f);
            
            if((x2 - x1 < 10) || (y2 - y1 < 10))
            {
                continue;
            }
            ObjectDetection obj(x1, y1, x2-x1, y2-y1,
                                prob, int(category), m_label_map[int(category)]);
            cur_objs.push_back(obj);
        }
        m_batch_objs.push_back(cur_objs);
    }

    /*
    int num = num_dets(0);
    for (int i=0; i<num; i++){
        float x1 = boxes(0, i, 1)*m_scale_x*m_net_w;
        float y1 = boxes(0, i, 0)*m_scale_y*m_net_h;
        float x2 = boxes(0, i, 3)*m_scale_x*m_net_w;
        float y2 = boxes(0, i, 2)*m_scale_y*m_net_h;
        x1 = std::min(std::max(x1, 0.f), m_origin_img_w-1.f);
        y1 = std::min(std::max(y1, 0.f), m_origin_img_h-1.f);
        x2 = std::min(std::max(x2, 0.f), m_origin_img_w-1.f);
        y2 = std::min(std::max(y2, 0.f), m_origin_img_h-1.f);
        ObjectDetection obj(x1, y1, x2-x1, y2-y1,
                            scores(i), int(classes(i)), m_label_map[int(classes(i))]);
        m_objs.push_back(obj);
    }
    */
    return m_batch_objs.size();
}


int ObjectDetectorTF_EFFIDET::batchObject(int idx, std::vector<ObjectDetection> &obj){
    if (idx >= m_batch_objs.size())
        return -1;
    obj = m_batch_objs[idx];
    return 0;
}

#endif
