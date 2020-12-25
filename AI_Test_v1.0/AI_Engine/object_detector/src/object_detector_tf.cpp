#ifdef ENABLE_TENSORFLOW
#include "object_detector_tf.hpp"
#include "tensorflow/core/graph/default_device.h"

using tensorflow::Status;

ObjectDetectorTF::ObjectDetectorTF(){
}
    
ObjectDetectorTF::~ObjectDetectorTF(){
    m_objs.clear();
}

int ObjectDetectorTF::init(const ODParam &param){
    if (0 != readLabelMap(param.label_map))
        return -1;

    m_net_w = param.net_w;
    m_net_h = param.net_h;
    
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
    warmUp();
    return 0;
}

int ObjectDetectorTF::setTensor(tensorflow::Tensor& in_tensor){
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

int ObjectDetectorTF::predict(){
    m_objs.clear();
    
    // set image to tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_UINT8,
                                    tensorflow::TensorShape({1, m_net_h, m_net_w, 3}));
    setTensor(input_tensor);

    // do predict
    std::vector<tensorflow::Tensor> outputs;
    std::string input_name = "image_tensor:0";
    std::vector<std::string> output_name = {"detection_boxes:0",
                                            "detection_scores:0",
                                            "detection_classes:0",
                                            "num_detections:0"};

    Status status = m_session->Run({{input_name, input_tensor}},
                                   output_name, {}, &outputs);
    if (!status.ok()){
        LOG_ERROR(status.ToString().c_str());
        return -1;
    }
    
    tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float,3>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_dets = outputs[3].flat<float>();

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
    return m_objs.size();
}
#endif
