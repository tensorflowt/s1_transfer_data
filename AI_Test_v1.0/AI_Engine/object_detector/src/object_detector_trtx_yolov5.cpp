#ifdef ENABLE_TENSORRTX 
#include "sak_utils.hpp"
#include "object_detector_trtx_yolov5.hpp"
#include "yololayer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <chrono>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;

ObjectDetectorTRTX_YOLOV5::ObjectDetectorTRTX_YOLOV5(){
}
    
ObjectDetectorTRTX_YOLOV5::~ObjectDetectorTRTX_YOLOV5(){
    m_objs.clear();
}

int ObjectDetectorTRTX_YOLOV5::init(const ODParam &param){
    //if (0 != readLabelMap(param.label_map))
    //    return -1;
	std::cout << "ObjectDetectorTRTX_YOLOV5 init..." << std::endl;														

    m_net_w = param.net_w;
    m_net_h = param.net_h;
    m_output_size = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
    conf_thresh = param.conf_thresh;  //add
    nms_thresh = param.nms_thresh;    //add

    cudaSetDevice(param.gpu_id);

    // Load tensorrt model
    LOG_INFO("Loading model: "<<param.model);

    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(param.model, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    } 
    else {

        LOG_ERROR("Cannot open yolov5 model file!!!");;
		std::cerr << "Cannot open yolov5 model file!!!" << std::endl;															 
        return -1;
    }

    m_trt_runtime = createInferRuntime(m_trt_logger);
    assert(m_trt_runtime != nullptr);
    m_trt_engine = m_trt_runtime->deserializeCudaEngine(trtModelStream, size);
    assert(m_trt_engine != nullptr);
    m_trt_context = m_trt_engine->createExecutionContext();
    assert(m_trt_context != nullptr);
    delete[] trtModelStream;

    assert(m_trt_engine->getNbBindings() == 2);
    const int inputIndex = m_trt_engine->getBindingIndex("data");
    assert(inputIndex == 0);
    const int outputIndex = m_trt_engine->getBindingIndex("prob");
    assert(outputIndex == 1);
    CHECK(cudaMalloc(&m_gpu_buffers[0], 3 * m_net_h * m_net_w * sizeof(float)));
    CHECK(cudaMalloc(&m_gpu_buffers[1], m_output_size * sizeof(float)));
    CHECK(cudaStreamCreate(&m_cudastream));

    // do warmup
    warmUp();
    return 0;
}

//image in bgr order
int ObjectDetectorTRTX_YOLOV5::setImage(const cv::Mat &img, bool keep_aspect){
    m_origin_img_w = img.cols;
    m_origin_img_h = img.rows;
    int w, h, x, y;
    float r_w = m_net_w / (img.cols*1.0);
    float r_h = m_net_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = m_net_w;
        h = r_w * img.rows;
        x = 0;
        y = (m_net_h - h) / 2;
    } else {
        w = r_h* img.cols;
        h = m_net_h;
        x = (m_net_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    m_img = cv::Mat(m_net_h, m_net_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(m_img(cv::Rect(x, y, re.cols, re.rows)));
    return 0;
}

int ObjectDetectorTRTX_YOLOV5::setTensor(float *data){
   
    cv::Mat _tmp(m_net_h, m_net_w, CV_8UC3);
    cv::Mat _dst(m_net_h, m_net_w, CV_32FC3, (void*)data);

    if (m_img.empty() || _tmp.empty() || _dst.empty())
    {
        return -1;
    }

    cv::Mat imgr = cv::Mat(m_net_h, m_net_w, CV_8UC1, _tmp.data);
    cv::Mat imgg = cv::Mat(m_net_h, m_net_w, CV_8UC1, _tmp.data + m_net_h * m_net_w);
    cv::Mat imgb = cv::Mat(m_net_h, m_net_w, CV_8UC1, _tmp.data + 2 * m_net_h * m_net_w);

    cv::Mat imgs[3] = { imgb, imgg, imgr };
    cv::split(m_img, imgs);

    _tmp.convertTo(_dst, CV_32FC3, 1.0/255.0);
    
    return 0;
}

static void doInference(IExecutionContext& context, float* input, int input_h, int input_w, float* output, int output_size, int batchSize, void** buffers, cudaStream_t& stream) {

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()

    // Create GPU buffers on device

    // Create stream

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    //CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //context.enqueue(batchSize, buffers, stream, nullptr);
    //CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    //cudaStreamSynchronize(stream);

    CHECK(cudaMemcpy(buffers[0], input, batchSize * 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice));
    context.execute(batchSize, buffers);
    CHECK(cudaMemcpy(output, buffers[1], batchSize * output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Release stream and buffers
}

cv::Rect get_rect(int src_h, int src_w, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (src_w * 1.0);
    float r_h = Yolo::INPUT_H / (src_h * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (Yolo::INPUT_H - r_w * src_h) / 2;
        b = bbox[1] + bbox[3]/2.f - (Yolo::INPUT_H - r_w * src_h) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (Yolo::INPUT_W - r_h * src_w) / 2;
        r = bbox[0] + bbox[2]/2.f - (Yolo::INPUT_W - r_h * src_w) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

static bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}

static void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

int ObjectDetectorTRTX_YOLOV5::predict(){
    m_objs.clear();

    // set image to tensor
    float *input_tensor = new float[m_net_h * m_net_w * 3];
    //auto time_beg1 = std::chrono::system_clock::now();
    setTensor(input_tensor);
    //auto time_end1 = std::chrono::system_clock::now();
    //std::cout << "predict->set image to tensor time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end1 - time_beg1).count() << "ms" << std::endl;

    // do predict
    //auto time_beg2 = std::chrono::system_clock::now();
    float *prob = new float[m_output_size];
    doInference(*m_trt_context, input_tensor, m_net_h, m_net_w, prob, m_output_size, 1, m_gpu_buffers, m_cudastream);
    //auto time_end2 = std::chrono::system_clock::now();
    //std::cout << "predict->do Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end2 - time_beg2).count() << "ms" << std::endl;

    // nms
    //auto time_beg3 = std::chrono::system_clock::now();
    std::vector<Yolo::Detection> res;
    nms(res, prob, conf_thresh, nms_thresh);
    //auto time_end3 = std::chrono::system_clock::now();
    //std::cout << "predict->nms time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end3 - time_beg3).count() << "ms" << std::endl;
    delete[] input_tensor;
    delete[] prob;

    // coordinate transformation
    //auto time_beg4 = std::chrono::system_clock::now();
    for (size_t j = 0; j < res.size(); j++) {
        cv::Rect r = get_rect(m_origin_img_h, m_origin_img_w, res[j].bbox);
        //ObjectDetection obj(r.x, r.y, r.widht, r.height, res[j].conf, int(res[j].class_id), m_label_map[int(res[j].class_id)]);
        ObjectDetection obj(r.x, r.y, r.width, r.height, res[j].conf, int(res[j].class_id), std::to_string(int(res[j].class_id)));
        m_objs.push_back(obj);
    }
    //auto time_end4 = std::chrono::system_clock::now();
    //std::cout << "predict->coordinate transformation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end4 - time_beg4).count() << "ms" << std::endl;
    return m_objs.size();
}

#endif