#define GPU
#include "darknet.h"
#include "opencv2/opencv.hpp"

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

int main(int argc, char **argv){
    // ./demo  cfg weight img
    float thresh=0.5;
    float hier_thresh = 0.5;
    cuda_set_device(2);
    network *net = load_network(argv[1], argv[2], 0);
    set_batch_network(net, 1);
    
    float nms=.45;
#if 0
    image im = load_image_color(argv[3],0,0);
    printf("%d, %d\n", im.w, im.h);
    image sized = letterbox_image(im, net->w, net->h);
    printf("%d, %d\n", net->w, net->h);
#else
    // resize the image
    cv::Mat img = cv::imread(argv[3]);
    image im = mat_to_image(img);
    image sized = make_image(net->w, net->h, 3);
    letterbox_image_into(im, net->w, net->h, sized);
    free_image(im);

#endif

    layer l = net->layers[net->n-1];
    //int nCls = 3;
    printf("class num: %d\n", l.classes);
    
    float *X = sized.data;
        
    //network_predict(net, X);
    network_predict(net, X);
    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
    //printf("%d\n", nboxes);
    //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("class_num: %d\n", l.classes);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    
    int i = 0;
    for (;i<nboxes; i++){
        // get the max probility class label
        int best_id = -1;
        float max_score = 0;
        
        int j;
        for(j = 0; j <3; ++j){
            if (dets[i].prob[j] > max_score){
                best_id = j;
                max_score = dets[i].prob[j];
            }
        }
        printf("%d, %f, %d\n", best_id, max_score, 3);
    }
    printf("predict num: %d\n",nboxes);
    
    free_image(im);
    free_image(sized);
    
}
