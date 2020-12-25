#ifdef ENABLE_TENSORFLOW
#include "semantic_segmentation_tf.hpp"
#include "tensorflow/core/graph/default_device.h"

using tensorflow::Status;

SemanticSegTF::SemanticSegTF(){
}
    
SemanticSegTF::~SemanticSegTF(){
}

int SemanticSegTF::init(const SegParam &param){
    if (0 != readLabelMap(param.label_map))
        return -1;

    m_net_w = param.net_w;
    m_net_h = param.net_h;
    m_mask = cv::Mat( m_net_h, m_net_w, CV_8UC1, cv::Scalar(0));
    m_output_type = param.output_type;

    // Load tensorflow model
    tensorflow::GraphDef graph_def;
    Status status = ReadBinaryProto(tensorflow::Env::Default(), param.model, &graph_def);
    if (!status.ok()) {
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }

    auto options = tensorflow::SessionOptions();
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(param.gpu_fraction);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    options.config.set_allow_soft_placement(true);
    if (param.gpu_id != -1)
        tensorflow::graph::SetDefaultDevice("/device:GPU:" + std::to_string(param.gpu_id), &graph_def);
        
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

int SemanticSegTF::setTensor(tensorflow::Tensor& in_tensor){
    unsigned char* pData = m_img.data;
    auto outputTensorMapped = in_tensor.tensor<unsigned char, 4>();
    for (int h=0; h<m_img.rows; ++h){
        for (int w=0; w<m_img.cols; ++w){
            outputTensorMapped(0, h, w, 0) = pData[0];
            outputTensorMapped(0, h, w, 1) = pData[1];
            outputTensorMapped(0, h, w, 2) = pData[2];
            pData += 3;
        }
    }
    
    return 0;
}

int SemanticSegTF::predict(){
    // clear mask
    m_mask.setTo(cv::Scalar(0));

    // set image to tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_UINT8,
                                    tensorflow::TensorShape({1, m_net_h, m_net_w, 3}));
    setTensor(input_tensor);
    
    // do predict
    std::vector<tensorflow::Tensor> outputs;
    std::string input_name = "ImageTensor";
    std::vector<std::string> output_name = {"SemanticPredictions"};

    Status status = m_session->Run({{input_name, input_tensor}},
                                   output_name, {}, &outputs);
    if (!status.ok()){
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }
    
    // print output shape
    auto shape = outputs[0].shape();
    LOG_DEBUG("semantic seg output dim 0: "<<shape.dim_size(0));
    LOG_DEBUG("semantic seg output dim 1: "<<shape.dim_size(1));
    LOG_DEBUG("semantic seg output dim 2: "<<shape.dim_size(2));
   
    if ("INT32" == m_output_type){ 
        auto mask = outputs[0].flat_outer_dims<int,3>();
        unsigned char *pData = m_mask.data;
        for (int h=0; h<m_net_h; h++){
            for (int w=0; w<m_net_w; w++){
                *(pData++) = (unsigned char)(mask(0, h, w));
            }
        }
    }
    else if ("INT64" == m_output_type) {
        auto mask = outputs[0].flat_outer_dims<long long int, 3>();
        unsigned char *pData = m_mask.data;
        for (int h=0; h<m_net_h; h++){
            for (int w=0; w<m_net_w; w++){
                *(pData++) = (unsigned char)(mask(0, h, w));
            }
        }
    }
    else{
        LOG_ERROR("Unsupported output data type: "<<m_output_type);
        return -1;
    }

    return 0;
}
#endif
