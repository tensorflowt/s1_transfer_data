#ifdef ENABLE_TENSORFLOW
#include "object_detector_mrcnn_tf.hpp"
#include "tensorflow/core/graph/default_device.h"

using tensorflow::Status;

ObjectDetectorMrcnnTF::ObjectDetectorMrcnnTF(){
}
    
ObjectDetectorMrcnnTF::~ObjectDetectorMrcnnTF(){
    m_objs.clear();
}

//image in bgr order
int ObjectDetectorMrcnnTF::setImage(const cv::Mat &img, bool keep_aspect){
    m_origin_img_w = img.cols;
    m_origin_img_h = img.rows;
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
        m_img = cv::Mat(m_net_h, m_net_w, CV_8UC3, cv::Scalar(103.53, 116.28, 123.675)); //bgr
        cv::Rect roi(0,0,scale_img_w, scale_img_h);
        cv::resize(img, m_img(roi),
                   cv::Size(scale_img_w, scale_img_h), 0, 0,
                   cv::INTER_LINEAR);
        //cv::cvtColor(m_img, m_img, cv::COLOR_BGR2RGB);
    }
    else {
        cv::resize(img, m_img,
                   cv::Size(m_net_w, m_net_h), 0, 0,
                   cv::INTER_LINEAR);
        m_scale_x = float(img.cols)/m_net_w;
        m_scale_y = float(img.rows)/m_net_h;
        //cv::cvtColor(m_img, m_img, cv::COLOR_BGR2RGB);
    }
    return 0;
}


int ObjectDetectorMrcnnTF::setTensor(tensorflow::Tensor& in_tensor){
    unsigned char* pData = m_img.data;
    auto outputTensorMapped = in_tensor.tensor<float, 3>();
    for (int h=0; h<m_img.rows; ++h){
        for (int w=0; w<m_img.cols; ++w){
            outputTensorMapped(h, w, 0) = float(pData[0]);
            outputTensorMapped(h, w, 1) = float(pData[1]);
            outputTensorMapped(h, w, 2) = float(pData[2]);
            pData += 3;
        }
    }
    return 0;
}

int ObjectDetectorMrcnnTF::predict(){
    m_objs.clear();
    
    // set image to tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({m_net_h, m_net_w, 3}));
    setTensor(input_tensor);

    // do predict
    std::vector<tensorflow::Tensor> outputs;
    std::string input_name = "image:0";
    std::vector<std::string> output_name = {"output/boxes:0",
                                            "output/scores:0",
                                            "output/labels:0"};

    Status status = m_session->Run({{input_name, input_tensor}},
                                   output_name, {}, &outputs);
    if (!status.ok()){
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }
    
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
    return m_objs.size();
}

#endif
