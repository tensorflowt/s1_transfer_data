#ifdef ENABLE_DARKNET
#include "object_detector_darknet.hpp"
#include "cuda_runtime.h"
#include "darknet.h"

static image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

static image mat_to_image(cv::Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}



ObjectDetectorDarknet::ObjectDetectorDarknet(){
    m_network = 0;
}
    
ObjectDetectorDarknet::~ObjectDetectorDarknet(){
    m_objs.clear();
    if (0 != m_tensor.data)
        free_image(m_tensor);
}

int ObjectDetectorDarknet::init(const ODParam &param){
    if (0 != readLabelMap(param.label_map)){
        LOG_ERROR("readLabelMap failed: "<<param.label_map);
        return -1;
    }

    m_net_w = param.net_w;
    m_net_h = param.net_h;
    if (param.nms_thresh < 1e-3)
        m_nms_thresh = 0.45;
    else
        m_nms_thresh = param.nms_thresh;
    
    m_tensor = make_image(m_net_w, m_net_h, 3);
    
    if (param.gpu_id != -1)
        cuda_set_device(param.gpu_id);
    
    LOG_INFO(param.deploy);
    LOG_INFO(param.model);
    
    m_network = load_network((char*)param.deploy.c_str(), (char *)param.model.c_str(), 0);
    set_batch_network(m_network, 1);
    
    // do warmup
    warmUp();
    return 0;
}

int ObjectDetectorDarknet::setImage(const cv::Mat &img, bool keep_aspect){
    m_origin_img_w = img.cols;
    m_origin_img_h = img.rows;
    // calculate the scale
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
    m_offset_x = (m_net_w-scale_img_w)/2;
    m_offset_y = (m_net_h-scale_img_h)/2;
    
    // resize the image
    fill_image(m_tensor, .5);
    image tmp= mat_to_image(img);
    letterbox_image_into(tmp, m_net_w, m_net_h, m_tensor);
    free_image(tmp);
    return 0;
}


int ObjectDetectorDarknet::predict(){
    m_objs.clear();

    // cant't get class_num with l.classes, todo check
    layer l = m_network->layers[m_network->n-1];
    int class_num = m_label_map.size();
    //std::cout<<l.classes<<" "<<class_num<<std::endl;

    network_predict(m_network, m_tensor.data);
    int nboxes = 0;
    float thresh = 0.1;//5;
    detection *dets = get_network_boxes(m_network, m_net_w, m_net_h, thresh, 0.5, 0, 1, &nboxes);
    if (m_nms_thresh) 
        do_nms_sort(dets, nboxes, class_num, m_nms_thresh);
    
    // Get detection results
    for (int i=0; i<nboxes; i++){
        // get the max probility class label
        int best_id = -1;
        float max_score = 0;
        
        for(int j = 0; j < class_num; ++j){
            if (dets[i].prob[j] > max_score){
                best_id = j;
                max_score = dets[i].prob[j];
            }
        }
        //printf("%d, %f, %d\n", best_id, max_score, class_num);
        if (max_score < thresh)
            continue;
        
        box b = dets[i].bbox;
        float x = ((b.x-b.w/2.)*m_net_w-m_offset_x)*m_scale_x;
        float y = ((b.y-b.h/2.)*m_net_h-m_offset_y)*m_scale_y;
        float w = b.w*m_net_w*m_scale_x;
        float h = b.h*m_net_h*m_scale_y;
        x = std::min(std::max(x, 0.f), m_origin_img_w-1.f);
        y = std::min(std::max(y, 0.f), m_origin_img_h-1.f);
        w = std::min(std::max(w, 0.f), m_origin_img_w-1.f-x);
        h = std::min(std::max(h, 0.f), m_origin_img_h-1.f-y);
        ObjectDetection obj(x, y, w, h, max_score, best_id, m_label_map[best_id]);
        m_objs.push_back(obj);
    }

    free_detections(dets,  nboxes);
    return m_objs.size();
}
#endif
