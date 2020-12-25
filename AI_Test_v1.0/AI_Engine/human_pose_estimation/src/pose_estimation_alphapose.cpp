#ifdef ENABLE_TORCH
#include "pose_estimation_alphapose.hpp"
#include "torch/torch.h"
#include "torch/script.h"
#include "sak_utils.hpp"

class TorchModuleWrapper
{
public:
    TorchModuleWrapper(){};
    ~TorchModuleWrapper(){};

public:
    torch::jit::script::Module m_module;
};

EstimAlphaPose::EstimAlphaPose(){
    m_module_wrapper = 0;
}
    
EstimAlphaPose::~EstimAlphaPose(){
    if (0 != m_module_wrapper){
        TorchModuleWrapper *module = (TorchModuleWrapper *)m_module_wrapper;
        delete module;
        m_module_wrapper = 0;
    }
}

int EstimAlphaPose::init(const HpesParam &param){
    m_net_w = param.net_w;
    m_net_h = param.net_h;
    m_device_id = param.gpu_id;
    TorchModuleWrapper *module = new TorchModuleWrapper();
    module->m_module = torch::jit::load(param.model);
    if (m_device_id < torch::cuda::device_count()){
        module->m_module.to(torch::Device(torch::kCUDA, m_device_id));
    }
    else{
        module->m_module.to(torch::kCPU);
    }
    
    m_module_wrapper = module;
    // do warmup
    warmUp();
    return 0;
}

Point2f  transpoints(Point2f pt, cv::Rect rect, int H, int W) {
	int resH = 64;
    int center_x = rect.width/2-1;
	int center_y = rect.height / 2-1;
	float new_width = rect.height * (float)W / H;
	Point2f pt2 = Point2f((int)(pt.x *rect.height/(float)resH), (int)(pt.y*rect.height/(float)resH));
	int transx = pt2.x - std::max(0, (int)((new_width-1)/2 - center_x));
	int transy = pt2.y - std::max(0, (int)((rect.height-1) / 2 - center_y));
	Point2f pt3 = Point2f(transx, transy);
	Point2f pt4 = Point2f(pt3.x + rect.x, pt3.y + rect.y);
	return pt4;
}

Valpoints getpred(torch::Tensor hms, cv::Rect rect, int H, int W) {
    torch::Tensor hm = hms[0];
	Valpoints valpoints;
	std::vector<Point2f> tripoints;
	std::vector<float> trivals;
	for (int i = 0; i < 17; i++) {
		torch::Tensor maxval = torch::max(hm[i]);
		torch::Tensor id = torch::argmax(hm[i]);
		int index = id.item().toInt();
		float val = maxval.item().toFloat(); 
		int row = (index + 1)*4 / W;
		int col = (index+1)%(W/4);
		Point2f hmp = Point2f(col, row);
		Point2f pt4 = transpoints(hmp, rect, H, W);
		tripoints.push_back(pt4);
		trivals.push_back(val);
	};
	valpoints.preds = tripoints;
	valpoints.scores = trivals;
	return valpoints;
}

int EstimAlphaPose::predict(){
    double model_begin = time_stamp();
    torch::Tensor img_ten = torch::from_blob(m_img.data, {1, m_img.rows, m_img.cols, 3}, torch::kByte);
    if (m_device_id < torch::cuda::device_count()){
        img_ten = img_ten.to(torch::Device(torch::kCUDA, m_device_id));
        img_ten = img_ten.permute({0, 3, 1, 2});
        img_ten = img_ten.toType(torch::kFloat);
        img_ten = img_ten.div(255);
        img_ten[0][0] = img_ten[0][0].sub_(0.485);
	    img_ten[0][1] = img_ten[0][1].sub_(0.456);
	    img_ten[0][2] = img_ten[0][2].sub_(0.406);
        img_ten = img_ten.to(torch::Device(torch::kCUDA, m_device_id));
        }
    else {
        img_ten = img_ten.to(torch::kCPU);
        img_ten = img_ten.permute({0, 3, 1, 2});
        img_ten = img_ten.toType(torch::kFloat);
        img_ten = img_ten.div(255);
        img_ten[0][0] = img_ten[0][0].sub_(0.485);
	    img_ten[0][1] = img_ten[0][1].sub_(0.456);
	    img_ten[0][2] = img_ten[0][2].sub_(0.406);
        img_ten = img_ten.to(torch::kCPU);
    }
    double model_end = time_stamp();
    TorchModuleWrapper *module = (TorchModuleWrapper *)m_module_wrapper;
    //double model_end = time_stamp();

    double time_start = time_stamp();
    torch::NoGradGuard no_grad;
    torch::Tensor res = module->m_module.forward({img_ten}).toTensor();  
    m_points = getpred(res, m_rect, m_net_h, m_net_w);
    double time_end = time_stamp(); 
    return 0;
}
#endif