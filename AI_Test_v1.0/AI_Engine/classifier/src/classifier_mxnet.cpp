#ifdef ENABLE_MXNET
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "classifier_mxnet.hpp"


ClassifierMXNET::ClassifierMXNET(){
    m_handle = nullptr;
}

ClassifierMXNET::~ClassifierMXNET(){
	if (m_handle != nullptr)
		MXPredFree(m_handle);
    m_handle = nullptr;
}

int ClassifierMXNET::loadFile(const std::string &fname, std::vector<char> &buf){
    std::ifstream fs(fname, std::ios::binary | std::ios::in);
	if (!fs.good()){
		LOG_ERROR(fname << " does not exist");
		return -1;
	}

	fs.seekg(0, std::ios::end);
	int fsize = fs.tellg();
	fs.seekg(0, std::ios::beg);
	buf.resize(fsize);
	fs.read(buf.data(), fsize);
	fs.close();
	return 0;
}

int ClassifierMXNET::init(const CLSParam &param){
    setenv("MXNET_CUDNN_AUTOTUNE_DEFAULT", "0", 1);
    m_net_w = param.net_w;
    m_net_h = param.net_h;
    std::vector<char> param_buffer;
	std::vector<char> json_buffer;
	if (loadFile(param.model, param_buffer) < 0){
		LOG_ERROR("Failed to load param file:"<<param.model);
		return -1;
	}
	if (loadFile(param.deploy, json_buffer) < 0){
		LOG_ERROR("Failed to load deploy file:"<<param.deploy);
		return -1;
	}

    mx_uint num_input_nodes = 1;
    const char * input_keys[1] = { "data" };
    const mx_uint input_shape_indptr[] = { 0, 4 };
    const mx_uint input_shape_data[] = {
        static_cast<mx_uint>(1),
        static_cast<mx_uint>(3),
        static_cast<mx_uint>(m_net_h),
        static_cast<mx_uint>(m_net_w)
    };

    if (0 != MXPredCreate(json_buffer.data(), param_buffer.data(), param_buffer.size(), 
                          2, param.gpu_id, 
                          num_input_nodes, input_keys, input_shape_indptr, input_shape_data, 
                          &m_handle)){
        LOG_ERROR("Create model failed");
        return -1;
    }
        
    // do warmup
    {
        m_img = cv::Mat(m_net_h, m_net_w, CV_8UC3);
        cv::randu(m_img, cv::Scalar::all(0), cv::Scalar::all(255));
        cv::Mat fea;
        extractFeature(fea);
    }
    return 0;
}

int ClassifierMXNET::setTensor(std::vector<float> &input){
    cv::Mat img_float;
    m_img.convertTo(img_float, CV_32FC3);   

    std::vector<cv::Mat> input_channels;
    float* input_data = input.data();
	for (int i = 0; i < 3; ++i) {
		cv::Mat channel(m_img.rows, m_img.cols, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += m_img.rows*m_img.cols;
	}
    cv::split(img_float, input_channels); 
    return 0;
}

int ClassifierMXNET::extractFeature(cv::Mat &fea){
    std::vector<float> input(3* m_img.rows * m_img.cols);
    setTensor(input);
    
    MXPredSetInput(m_handle, "data", input.data(), input.size());
    MXPredForward(m_handle);
    
    mx_uint *shape = nullptr;
    mx_uint shape_len = 0;
    MXPredGetOutputShape(m_handle, 0, &shape, &shape_len);

    int feature_size = 1;
    for (unsigned int i = 0; i<shape_len; i++)
        feature_size *= shape[i];
    std::vector<float> feature(feature_size);
    MXPredGetOutput(m_handle, 0, feature.data(), feature_size);
    
    fea = cv::Mat(feature).reshape(1, 1).clone();
    cv::normalize(fea, fea);
    return 0;
}


#endif
