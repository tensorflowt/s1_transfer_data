#ifdef ENABLE_TORCH
#include "classifier_torch.hpp"


ClassifierT7::ClassifierT7(){
}
    
ClassifierT7::~ClassifierT7(){
}

int ClassifierT7::init(const CLSParam &param){
    if (0 != readLabelMap(param.label_map))
        return -1;

    m_net_w = param.net_w;
    m_net_h = param.net_h;
    m_device_id = param.gpu_id;
    m_img_float = cv::Mat(m_net_h, m_net_w, CV_32FC3);

    // Load tensorflow model
    m_module = torch::jit::load(param.model);
    if (param.gpu_id < torch::cuda::device_count())
        m_module.to(torch::Device(torch::kCUDA, param.gpu_id));
    else 
        m_module.to(torch::kCPU);

    // Do warmup
    warmUp();
    return 0;
}

int ClassifierT7::predict(float &score, int &id, std::string &name){
    score = 0;
    id = -1;
    name = "unknown";
    
    torch::Tensor img_ten = torch::from_blob(m_img.data, {1, m_img.rows, m_img.cols, 3}, torch::kByte);
    img_ten = img_ten.permute({0, 3, 1, 2});
    img_ten = img_ten.toType(torch::kFloat);
    
    if (m_device_id < torch::cuda::device_count())
        img_ten = img_ten.to(torch::Device(torch::kCUDA, m_device_id));
    else 
        img_ten = img_ten.to(torch::kCPU);
    img_ten = img_ten.div(255);
    img_ten[0][0] = img_ten[0][0].sub_(0.485).div_(0.229);  
    img_ten[0][1] = img_ten[0][1].sub_(0.456).div_(0.224); 
    img_ten[0][2] = img_ten[0][2].sub_(0.406).div_(0.225);
    
    torch::NoGradGuard no_grad;
    torch::Tensor res = m_module.forward({img_ten}).toTensor();  
    // bg:0, water:1
    int class_num = res.size(0);
    float *scores = res.data<float>();
    for (int i=0; i<class_num; i++){
        if (score < scores[i]){
            score = scores[i];
            id = i;
        }
    }
    if (id != -1)
        name = m_label_map[id];
    return 0;
}

int ClassifierT7::predict(std::map<std::string, float> &resM){
    resM.clear();
    
    torch::Tensor img_ten = torch::from_blob(m_img.data, {1, m_img.rows, m_img.cols, 3}, torch::kByte);
    img_ten = img_ten.permute({0, 3, 1, 2});
    img_ten = img_ten.toType(torch::kFloat);
    
    if (m_device_id < torch::cuda::device_count())
        img_ten = img_ten.to(torch::Device(torch::kCUDA, m_device_id));
    else 
        img_ten = img_ten.to(torch::kCPU);
    img_ten = img_ten.div(255);
    img_ten[0][0] = img_ten[0][0].sub_(0.485).div_(0.229);  
    img_ten[0][1] = img_ten[0][1].sub_(0.456).div_(0.224); 
    img_ten[0][2] = img_ten[0][2].sub_(0.406).div_(0.225);

    torch::NoGradGuard no_grad;
    torch::Tensor res = m_module.forward({img_ten}).toTensor();  
    // bg:0, water:1
    int class_num = res.size(0);
    float *scores = res.data<float>();
    for (int i=0; i<class_num; i++){
        resM.insert(std::pair<std::string, float>(m_label_map[i], scores[i]));
        // std::cout<<scores[i]<<std::endl;
    }

    
    return 0;
    
}
#endif
