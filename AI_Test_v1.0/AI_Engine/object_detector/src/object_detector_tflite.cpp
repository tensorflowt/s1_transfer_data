#ifdef ENABLE_TFLITE

#include "object_detector_tflite.hpp"
#include "tensorflow/lite/optional_debug_tools.h"
//#include "tensorflow/lite/profiling/profiler.h"
//#include "tensorflow/lite/string_util.h"

ObjectDetectorTflite::ObjectDetectorTflite(){
}
    
ObjectDetectorTflite::~ObjectDetectorTflite(){
    m_objs.clear();
}

int ObjectDetectorTflite::init(const ODParam &param){
    if (0 != readLabelMap(param.label_map)){
        LOG_ERROR("readLabelMap failed: "<<param.label_map);
        return -1;
    }
    
    m_net_w = param.net_w;
    m_net_h = param.net_h;

    // Load tflite model
    tflite::ops::builtin::BuiltinOpResolver resolver;
    m_model = tflite::FlatBufferModel::BuildFromFile(param.model.c_str());
    if (!m_model) {
        LOG_ERROR("Failed to mmap model " <<param.model);
        return -1;
    }
    
    LOG_INFO("Loaded model " << param.model);
    m_model->error_reporter();
    LOG_INFO("resolved reporter");
    
    tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter);
    if (!m_interpreter) {
        LOG_ERROR("Failed to construct interpreter");
        return -1;
    }

    m_interpreter->UseNNAPI(false);
    m_interpreter->SetAllowFp16PrecisionForFp32(false);
    m_interpreter->SetNumThreads(1);
    
    if (m_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cout << "Failed to allocate tensors!";
        return -1;
    }
    
    if (0) {
        PrintInterpreterState(m_interpreter.get());
        LOG_DEBUG("tensors size: " << m_interpreter->tensors_size());
        LOG_DEBUG("nodes size: " << m_interpreter->nodes_size());
        LOG_DEBUG("inputs: " << m_interpreter->inputs().size());
        LOG_DEBUG("input(0) name: " << m_interpreter->GetInputName(0));
        int t_size = m_interpreter->tensors_size();
        for (int i = 0; i < t_size; i++) {
            if (m_interpreter->tensor(i)->name)
                std::cout << i << ": " << m_interpreter->tensor(i)->name << ", "
                          << m_interpreter->tensor(i)->bytes << ", "
                          << m_interpreter->tensor(i)->type << ", "
                          << m_interpreter->tensor(i)->params.scale << ", "
                          << m_interpreter->tensor(i)->params.zero_point << "\n";
        }
        const std::vector<int> inputs = m_interpreter->inputs();
        const std::vector<int> outputs = m_interpreter->outputs();
        LOG_DEBUG("number of inputs: " << inputs.size());
        LOG_DEBUG("number of outputs: " << outputs.size());
    }
    
    // do warmup
    warmUp();
    return 0;
}

int ObjectDetectorTflite::setTensor(){
    uint8_t *pSrc = m_img.data;
    int input = m_interpreter->inputs()[0];
    int size = m_net_w*m_net_h*3;

    int type = m_interpreter->tensor(input)->type;
    if (kTfLiteFloat32 == type){
        LOG_INFO("Float input tensor");
        float *pDst = m_interpreter->typed_tensor<float>(input);
        for (int i=0; i<size; i++)
            pDst[i] = (pSrc[i]-127.5f)/127.5f;
    }
    else if (kTfLiteUInt8 == type){
        LOG_INFO("Uint8 input tensor");
        uint8_t *pDst = m_interpreter->typed_tensor<uint8_t>(input);
        for (int i=0; i<size; i++)
            pDst[i] = pSrc[i];
    }
    else
        return -1;
        
    return 0;
}


int ObjectDetectorTflite::predict(){
    m_objs.clear();
    if (0 != setTensor())
        return -1;

    if (m_interpreter->Invoke() != kTfLiteOk)
       return -1;

    float *pBoxes = m_interpreter->typed_output_tensor<float>(0);
    float *pScores = m_interpreter->typed_output_tensor<float>(1);
    float *pClasses = m_interpreter->typed_output_tensor<float>(2);
    float *pDetNum = m_interpreter->typed_output_tensor<float>(3);
    int num = pDetNum[0];
    for (int i=0; i<num; i++){
        
        float x1 = pBoxes[i*4+1]*m_scale_x*m_net_w;
        float y1 = pBoxes[i*4+0]*m_scale_y*m_net_h;
        float x2 = pBoxes[i*4+3]*m_scale_x*m_net_w;
        float y2 = pBoxes[i*4+2]*m_scale_y*m_net_h;
        ObjectDetection obj(x1, y1, x2-x1+1, y2-y1+1,
                            pScores[i], int(pClasses[i]),
                            m_label_map[int(pClasses[i])]);
        m_objs.push_back(obj);
        
    }
    
    return m_objs.size();
}
#endif
