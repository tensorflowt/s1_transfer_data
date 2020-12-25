#ifdef ENABLE_TENSORFLOW
#include "classifier_video_i3d.hpp"
#include "tensorflow/core/graph/default_device.h"

using tensorflow::Status;

ClassifierVidI3D::ClassifierVidI3D(){
}
    
ClassifierVidI3D::~ClassifierVidI3D(){
}

int ClassifierVidI3D::init(const VidCLSParam &param){
    if (0 != readLabelMap(param.label_map))
        return -1;

    m_output_node = param.output_node;
    m_input_node = param.input_node;
    m_net_w = param.net_w;
    m_net_h = param.net_h;
    m_net_t = param.net_t;
    
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
    
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(param.gpu_fraction);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    options.config.set_allow_soft_placement(true);
    if (param.gpu_id != -1)
        tensorflow::graph::SetDefaultDevice("/device:GPU:" + std::to_string(param.gpu_id), &graph_def);
    //options.config.mutable_gpu_options()->set_visible_device_list(std::to_string(param.gpu_id));// This is global setting
    
    m_session.reset(tensorflow::NewSession(options));
    status = m_session->Create(graph_def);
    if (!status.ok()) {
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }
    
    // do warmup
    warmUp();
    return 0;
}

int ClassifierVidI3D::setTensor(tensorflow::Tensor& in_tensor){

    auto outputTensorMapped = in_tensor.tensor<float, 5>();
    for (int t=0; t<m_net_t; t++){
        unsigned char* pData = m_imgV[t].data;
        for (int h=0; h<m_net_h; ++h){
            for (int w=0; w<m_net_w; ++w){
                outputTensorMapped(0, t, h, w, 0) = 2*(pData[3*w+0]/255.f)-1;
                outputTensorMapped(0, t, h, w, 1) = 2*(pData[3*w+1]/255.f)-1;
                outputTensorMapped(0, t, h, w, 2) = 2*(pData[3*w+2]/255.f)-1;
            }
            pData += m_imgV[t].step;
        }
    }
    return 0;
}

int ClassifierVidI3D::predict(float &score, int &id, std::string &name){
    score = 0;
    id = -1;
    name = "unknown";
    // set image to tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({1, m_net_t, m_net_h, m_net_w, 3}));
    setTensor(input_tensor);
    
    // do predict
    std::vector<tensorflow::Tensor> outputs;
    std::string input_name = m_input_node;
    std::vector<std::string> output_name = {m_output_node};
    Status status = m_session->Run({{input_name, input_tensor}},
                                   output_name, {}, &outputs);
    if (!status.ok()){
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }
    auto shape = outputs[0].shape();
    int class_num = shape.dim_size(1);
    
    tensorflow::TTypes<float>::Flat scores = outputs[0].flat<float>();
    for (int i=0; i<class_num; i++){
        if (score < scores(i)){
            score = scores(i);
            id = i;
        }
    }
    if (id != -1)
        name = m_label_map[id];
    return 0;
}

int ClassifierVidI3D::predict(std::map<std::string, float> &res){
    // set image to tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({1, m_net_t, m_net_h, m_net_w, 3}));
    setTensor(input_tensor);
    
    // do predict
    std::vector<tensorflow::Tensor> outputs;
    std::string input_name = m_input_node;
    std::vector<std::string> output_name = {m_output_node};
    Status status = m_session->Run({{input_name, input_tensor}},
                                   output_name, {}, &outputs);
    if (!status.ok()){
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }
    auto shape = outputs[0].shape();
    int class_num = shape.dim_size(1);
    
    tensorflow::TTypes<float>::Flat scores = outputs[0].flat<float>();
    for (int i=0; i<class_num; i++){
        res.insert(std::pair<std::string, float>(m_label_map[i], scores(i)));
    }
    return 0;
}
#endif
