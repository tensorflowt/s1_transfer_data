# 基于TensorRTX的YOLOv5s模型加速实现说明

1.模型转换(pt2wts)
'''
git clone git@192.168.1.31:SanY_CV_Research/yolov5.git
cd yolov5
python gen_wts.py
'''

2.模型推理引擎输出(xxx.engine)
'''
cd ../AI_Engine/object_detector/yolov5
mkdir build
cd build
cmake ..
make
./yolov5s -s
'''
3.编译demo示例可执行文件
'''
cd /AI_Test
make build
cd build
cmake ..
make
'''

4.执行demo示例
'''
cd /AI_Test/build/bin/ai_engine
./demo_trtx_od_image_yolov5 -g=0 -m=6
'''

5.输出结果如下：
Load image time: 28.151ms
Resize image to target size time: 5.952ms
predict->set image to tensor time: 1.502ms
predict->do Inference time: 5.523ms
predict->nms time: 0.022ms
predict->coordinate transformation time: 0.006ms
Predict time: 7.14ms
Writing image time: 35.974ms
Average time for each image [Read img + resize and copy to device + predict]: 78.8475ms
