#ifdef ENABLE_TORCH
#include "semantic_segmentation_torch.hpp"
#include "torch/torch.h"
#include "torch/script.h"


class TorchModuleWrapper
{
public:
    TorchModuleWrapper(){};
    ~TorchModuleWrapper(){};

public:
    torch::jit::script::Module m_module;
};

SemanticSegT7::SemanticSegT7(){
    m_module_wrapper = 0;
}
    
SemanticSegT7::~SemanticSegT7(){
    if (0 != m_module_wrapper){
        TorchModuleWrapper *module = (TorchModuleWrapper *)m_module_wrapper;
        delete module;
        m_module_wrapper = 0;
    }
}

int SemanticSegT7::init(const SegParam &param){
    if (0 != readLabelMap(param.label_map))
        return -1;

    m_net_w = param.net_w;
    m_net_h = param.net_h;
    m_mask = cv::Mat( m_net_h, m_net_w, CV_8UC1, cv::Scalar(0));
    m_device_id = param.gpu_id;
    TorchModuleWrapper *module = new TorchModuleWrapper();
    module->m_module = torch::jit::load(param.model);
    if (m_device_id < torch::cuda::device_count())
        module->m_module.to(torch::Device(torch::kCUDA, m_device_id));
    else 
        module->m_module.to(torch::kCPU);
    
    m_module_wrapper = module;
    // do warmup
    warmUp();
    return 0;
}

int SemanticSegT7::predict(){
    torch::Tensor img_ten = torch::from_blob(m_img.data, {1, m_img.rows, m_img.cols, 3}, torch::kByte);
    img_ten = img_ten.permute({0, 3, 1, 2});
    img_ten = img_ten.toType(torch::kFloat);
    img_ten = img_ten.div(255);
    
    if (m_device_id < torch::cuda::device_count())
        img_ten = img_ten.to(torch::Device(torch::kCUDA, m_device_id));
    else 
        img_ten = img_ten.to(torch::kCPU);

    TorchModuleWrapper *module = (TorchModuleWrapper *)m_module_wrapper;
    torch::NoGradGuard no_grad;
    torch::Tensor res = module->m_module.forward({img_ten}).toTensor();  
    // bg:0, water:1
    cv::Mat mask(res.size(0), res.size(1), CV_8UC1, (void*) res.data<uint8_t>()); 
    cv::resize(mask, m_mask, cv::Size(m_net_w, m_net_h));
    return 0;
}
#endif
