#ifdef ENABLE_TENSORRTX 
#include "sak_utils.hpp"
#include "object_detector_trtx_yolov5.hpp"
#include "yololayer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <chrono>

#define USE_FP16  // Ä¬ÈÏFP32

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

// ----------------- functions for building tensorrt model -----------------
static std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

static IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

static ILayer* convBnLeaky(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);
    return lr;
}

static ILayer* focus(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname, int net_w, int net_h) {
    ISliceLayer* s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, net_h / 2, net_w / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, net_h / 2, net_w / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, net_h / 2, net_w / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, net_h / 2, net_w / 2 }, Dims3{ 1, 2, 2 });
    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBnLeaky(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

static ILayer* bottleneck(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBnLeaky(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBnLeaky(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

static ILayer* bottleneckCSP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBnLeaky(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor* y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor* inputTensors[] = { cv3->getOutput(0), cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBnLeaky(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}

static ILayer* SPP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBnLeaky(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k1, k1 });
    pool1->setPaddingNd(DimsHW{ k1 / 2, k1 / 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k2, k2 });
    pool2->setPaddingNd(DimsHW{ k2 / 2, k2 / 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k3, k3 });
    pool3->setPaddingNd(DimsHW{ k3 / 2, k3 / 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });

    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBnLeaky(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

static std::vector<float> getAnchors(std::map<std::string, Weights>& weightMap) {
    std::vector<float> anchors_yolo;
    Weights Yolo_Anchors = weightMap["model.27.anchor_grid"];
    assert(Yolo_Anchors.count == 18);
    int each_yololayer_anchorsnum = Yolo_Anchors.count / 3;
    const float* tempAnchors = (const float*)(Yolo_Anchors.values);
    for (int i = 0; i < Yolo_Anchors.count; i++)
    {
        if (i < each_yololayer_anchorsnum)
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
        if ((i >= each_yololayer_anchorsnum) && (i < (2 * each_yololayer_anchorsnum)))
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
        if (i >= (2 * each_yololayer_anchorsnum))
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
    }
    return anchors_yolo;
}

static IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, IConvolutionLayer* det0, IConvolutionLayer* det1, IConvolutionLayer* det2, int net_w, int net_h, int class_num) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    std::vector<float> anchors_yolo = getAnchors(weightMap);
    for (int i = 0; i < anchors_yolo.size(); i++) {
        std::cout << anchors_yolo[i] << " ";
    }
    std::cout << std::endl;
    PluginField pluginMultidata[4];
    int NetData[4];
    NetData[0] = class_num;
    NetData[1] = net_w;
    NetData[2] = net_h;
    NetData[3] = Yolo::MAX_OUTPUT_BBOX_COUNT;
    pluginMultidata[0].data = NetData;
    pluginMultidata[0].length = 3;
    pluginMultidata[0].name = "netdata";
    pluginMultidata[0].type = PluginFieldType::kFLOAT32;
    int scale[3] = { 32, 16, 8 };
    int plugindata[3][8];
    std::string names[3];
    for (int k = 1; k < 4; k++)
    {
        plugindata[k - 1][0] = net_w / scale[k - 1];
        plugindata[k - 1][1] = net_h / scale[k - 1];
        for (int i = 2; i < 8; i++)
        {
            plugindata[k - 1][i] = int(anchors_yolo[(k - 1) * 6 + i - 2]);
        }
        pluginMultidata[k].data = plugindata[k - 1];
        pluginMultidata[k].length = 8;
        names[k - 1] = "yolodata" + std::to_string(k);
        pluginMultidata[k].name = names[k - 1].c_str();
        pluginMultidata[k].type = PluginFieldType::kFLOAT32;
    }
    PluginFieldCollection pluginData;
    pluginData.nbFields = 4;
    pluginData.fields = pluginMultidata;
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", &pluginData);
    ITensor* inputTensors_yolo[] = { det0->getOutput(0), det1->getOutput(0), det2->getOutput(0) };
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    return yolo;
}

static ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string wts_file, int net_w, int net_h, int class_num) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, net_h, net_w} with name "data"
    ITensor* data = network->addInput("data", dt, Dims3{ 3, net_h, net_w });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_file);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0", net_w, net_h);
    auto conv1 = convBnLeaky(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBnLeaky(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBnLeaky(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBnLeaky(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

    // yolov5 head
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
    auto conv10 = convBnLeaky(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 256 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(256);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBnLeaky(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 128 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(128);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
    IConvolutionLayer* conv18 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (class_num + 5), DimsHW{ 1, 1 }, weightMap["model.18.weight"], weightMap["model.18.bias"]);

    auto conv19 = convBnLeaky(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.19");
    ITensor* inputTensors20[] = { conv19->getOutput(0), conv14->getOutput(0) };
    auto cat20 = network->addConcatenation(inputTensors20, 2);
    auto bottleneck_csp21 = bottleneckCSP(network, weightMap, *cat20->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.21");
    IConvolutionLayer* conv22 = network->addConvolutionNd(*bottleneck_csp21->getOutput(0), 3 * (class_num + 5), DimsHW{ 1, 1 }, weightMap["model.22.weight"], weightMap["model.22.bias"]);

    auto conv23 = convBnLeaky(network, weightMap, *bottleneck_csp21->getOutput(0), 256, 3, 2, 1, "model.23");
    ITensor* inputTensors24[] = { conv23->getOutput(0), conv10->getOutput(0) };
    auto cat24 = network->addConcatenation(inputTensors24, 2);
    auto bottleneck_csp25 = bottleneckCSP(network, weightMap, *cat24->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.25");
    IConvolutionLayer* conv26 = network->addConvolutionNd(*bottleneck_csp25->getOutput(0), 3 * (class_num + 5), DimsHW{ 1, 1 }, weightMap["model.26.weight"], weightMap["model.26.bias"]);

    //auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    //const PluginFieldCollection* pluginData = creator->getFieldNames();
    //IPluginV2* pluginObj = creator->createPlugin("yololayer", pluginData);
    //ITensor* inputTensors_yolo[] = { conv26->getOutput(0), conv22->getOutput(0), conv18->getOutput(0) };
    //auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    auto yolo = addYoLoLayer(network, weightMap, conv18, conv22, conv26, net_w, net_h, class_num);

    yolo->getOutput(0)->setName("prob");
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

static void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string wts_file, Logger& logger, int net_w, int net_h, int class_num) {
    // Create builder
    IBuilder* builder = createInferBuilder(logger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wts_file, net_w, net_h, class_num);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

ObjectDetectorTRTX_YOLOV5::ObjectDetectorTRTX_YOLOV5(){
}

ObjectDetectorTRTX_YOLOV5::~ObjectDetectorTRTX_YOLOV5(){
    m_objs.clear();
}

int ObjectDetectorTRTX_YOLOV5::init(const ODParam &param){
    if (0 != readLabelMap(param.label_map))
        return -1;
    std::cout << "ObjectDetectorTRTX_YOLOV5 init..." << std::endl;

    m_net_w = param.net_w;
    m_net_h = param.net_h;
    m_class_num = param.class_num;
    m_output_size = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
    conf_thresh = param.conf_thresh;  //add
    nms_thresh = param.nms_thresh;    //add

    cudaSetDevice(param.gpu_id);

    // Load tensorrt model
    LOG_INFO("Loading model: "<<param.model);
    std::cout << "Loading model: " << param.model << std::endl;

    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(param.model, std::ios::binary);
    if (!file.good()) {
        LOG_INFO("Loading wts model for exchange model: ");
        IHostMemory* modelStream{ nullptr };
        APIToModel(1, &modelStream, param.wts_file, m_trt_logger, m_net_w, m_net_h, m_class_num);
        assert(modelStream != nullptr);
        std::ofstream p(param.model, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        p.close();
        file.open(param.model, std::ios::binary);
    }
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

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

    CHECK(cudaMemcpy(buffers[0], input, batchSize * 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice));
    context.execute(batchSize, buffers);
    CHECK(cudaMemcpy(output, buffers[1], batchSize * output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Release stream and buffers
}

static cv::Rect get_rect(int src_h, int src_w, int net_h, int net_w, float bbox[4]) {
    int l, r, t, b;
    float r_w = net_w / (src_w * 1.0);
    float r_h = net_h / (src_h * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (net_h - r_w * src_h) / 2;
        b = bbox[1] + bbox[3]/2.f - (net_h - r_w * src_h) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (net_w - r_h * src_w) / 2;
        r = bbox[0] + bbox[2]/2.f - (net_w - r_h * src_w) / 2;
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
        cv::Rect r = get_rect(m_origin_img_h, m_origin_img_w, m_net_h, m_net_w, res[j].bbox);
        //ObjectDetection obj(r.x, r.y, r.width, r.height, res[j].conf, int(res[j].class_id), m_label_map[int(res[j].class_id)]);
        ObjectDetection obj(r.x, r.y, r.width, r.height, res[j].conf, int(res[j].class_id), std::to_string(int(res[j].class_id)));
        m_objs.push_back(obj);
    }
    //auto time_end4 = std::chrono::system_clock::now();
    //std::cout << "predict->coordinate transformation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end4 - time_beg4).count() << "ms" << std::endl;
    return m_objs.size();
}

#endif
