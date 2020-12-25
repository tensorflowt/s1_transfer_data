#ifdef ENABLE_TORCH
#include <string>
#include "torch/torch.h"
#include "torch/script.h"
#include "object_detector_torch.hpp"
#include "sak_utils.hpp"

using namespace std;
/*
class TorchModuleWrapper
{
public:
    TorchModuleWrapper(){};
    ~TorchModuleWrapper(){};

public:
    torch::jit::script::Module m_module;
};
*/

ObjectDetectorTorch::ObjectDetectorTorch(){
}

ObjectDetectorTorch::~ObjectDetectorTorch(){
    m_objs.clear();
}

int ObjectDetectorTorch::init(const ODParam &param){
    if (0 != readLabelMap(param.label_map))
        return -1;

    m_net_w = param.net_w;
    m_net_h = param.net_h;
    
    m_device_id = param.gpu_id;
    
    // for post processing
    m_conf_thresh = param.conf_thresh;

    /*
    TorchModuleWrapper *module = new TorchModuleWrapper();
    module->m_module = torch::jit::load(param.model);
    if (m_device_id < torch::cuda::device_count())
        module->m_module.to(torch::Device(torch::kCUDA, m_device_id));
    else 
        module->m_module.to(torch::kCPU);
    
    m_module_wrapper = module;
    */
    m_module.reset(new torch::jit::script::Module(torch::jit::load(param.model)));
    assert(m_module != nullptr);

    if (m_device_id < torch::cuda::device_count())
        m_module->to(torch::Device(torch::kCUDA, m_device_id));
    else 
        m_module->to(torch::kCPU);
    
    //m_module->eval();
    // do warmup
    //LOG_INFO("Start warm up: ");
    //warmUp();
    return 0;
}


int ObjectDetectorTorch::postProcess(torch::Tensor &src_tensor){

    // src_tensor: batch_num,obj_num,5 + class_num
    // center x, center y, width, height, score, class_idx
    int item_attr_size = 5;
    int batch_size = src_tensor.size(0);
    int num_orig_number = src_tensor.size(1);
    int num_classes = src_tensor.size(2) - item_attr_size;
    std::cout<<"Class num:"<<num_classes<<std::endl;
    std::cout<<"Orig object num:"<<num_orig_number<<std::endl;

    // get candidates which object confidence > threshold
    // select(2,m) tensor[:,:,m]
    auto conf_mask = src_tensor.select(2, 4).ge(m_conf_thresh).unsqueeze(2);

    // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
    src_tensor.slice(2, item_attr_size, item_attr_size + num_classes) *= src_tensor.select(2, 4).unsqueeze(2); 

    // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
    torch::Tensor box = torch::zeros(src_tensor.sizes(), src_tensor.options());
    box.select(2, 0) = src_tensor.select(2, 0) - src_tensor.select(2, 2).div(2);
    box.select(2, 1) = src_tensor.select(2, 1) - src_tensor.select(2, 3).div(2);
    box.select(2, 2) = src_tensor.select(2, 0) + src_tensor.select(2, 2).div(2);
    box.select(2, 3) = src_tensor.select(2, 1) + src_tensor.select(2, 3).div(2);
    src_tensor.slice(2, 0, 4) = box.slice(2, 0, 4);

    // output tensor
    torch::Tensor output = torch::zeros({0,7});

    auto img_detection = torch::masked_select(src_tensor[0],conf_mask[0]).view({-1,src_tensor.size(2)});
    int num_obj = img_detection.size(0);
    std::cout<<"Filtered object num:"<<num_obj<<std::endl;

    /*
    for (int obj_idx = 0; obj_idx < num_obj; obj_idx ++)
    {

    }
    */
    return 0;
}


int ObjectDetectorTorch::predict(){
    m_objs.clear();
    
    // set input tensor and copy img data
    torch::Tensor img_ten = torch::from_blob(m_img.data, {1, m_img.rows, m_img.cols, 3}, torch::kByte);
    img_ten = img_ten.permute({0, 3, 1, 2});
    img_ten = img_ten.toType(torch::kFloat);
    img_ten = img_ten.div(255);
    
    if (m_device_id < torch::cuda::device_count())
        img_ten = img_ten.to(torch::Device(torch::kCUDA, m_device_id));
    else 
        img_ten = img_ten.to(torch::kCPU);


    torch::NoGradGuard no_grad;
    // output: nx6 (x1, y1, x2, y2, conf, cls)
    torch::Tensor output = m_module->forward({img_ten}).toTensor();
 
    int status = postProcess(output);

    // do predict
    // src_tensor: N*7
    /*
    tensorflow::TTypes<float, 2>::Tensor boxes = outputs[0].flat_outer_dims<float,2>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<int>::Flat classes = outputs[2].flat<int>();

    tensorflow::TensorShape class_shape = outputs[1].shape();
    int num = class_shape.dim_size(0);
    for (int i=0; i<num; i++){
        float x1 = boxes(i, 0)*m_scale_x;
        float y1 = boxes(i, 1)*m_scale_y;
        float x2 = boxes(i, 2)*m_scale_x;
        float y2 = boxes(i, 3)*m_scale_y;
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

#endif
