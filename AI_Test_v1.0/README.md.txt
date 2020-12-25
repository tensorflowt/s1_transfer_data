# Win10系统下YOLOv5（TensorRT FP16）前向推理工程使用说明

## 一、工程环境：

```
VS版本：Visual Studio 2019

cmake版本：3.17.5

显卡驱动版本：452.06-notebook-win10-64bit-international-dch-whql

CUDA版本：cuda_10.2.89_441.22_win10（说明：Visual Studio 2019目前只适配cuda10.2）

CUDNN版本：cudnn-10.0-windows10-x64-v7.6.5.32

TensorRT版本：TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6

OPENCV版本：opencv-4.3.0-vc14_vc15

Anaconda3: Anaconda3-2020.07-Windows-x86_64
```

## 二、工程文件组织结构：

```
|---AI Test
    |---AI_3rdparty
    |---AI_Engine
    |---AI_data
    |---config
    |---build
```

## 三、工程使用教程：

### Step1：cmake.

![image-20201116145806386](C:\Users\ZHANGWENTAO\AppData\Roaming\Typora\typora-user-images\image-20201116145806386.png)

### Step2：编译.

#### a.修改相关文件参数信息：

##### **i.yololayer.h文件修改模型参数：**

文件路径：

```
D:\Project\AI_Test\AI_Engine\object_detector\yolov5
```

修改内容：

```
#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H
#include <vector>
#include <string>
#include "NvInfer.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 1;    //根据自定义模型进行修改
    static constexpr int INPUT_H = 640;    //根据自定义模型进行修改
    static constexpr int INPUT_W = 640;    //根据自定义模型进行修改
```

##### **ii.yolov5.cpp文件修改模型路径：**

文件路径：

```
D:\Project\AI_Test\AI_Engine\object_detector\yolov5
```

修改内容：

![image-20201116151427193](C:\Users\ZHANGWENTAO\AppData\Roaming\Typora\typora-user-images\image-20201116151427193.png)

#### b.编译：

![image-20201116151616070](C:\Users\ZHANGWENTAO\AppData\Roaming\Typora\typora-user-images\image-20201116151616070.png)

选择项目ALL_BUILD右键进行生成,生成的可执行文件存放在如下路径：

```
D:\Project\AI_Test\build\bin\ai_engine\Debug
```

![image-20201116152245814](C:\Users\ZHANGWENTAO\AppData\Roaming\Typora\typora-user-images\image-20201116152245814.png)

#### c.模型转换：

将pytorch训练生成的wts模型（pt模型可通过gen_wts.py脚本生成）转为TensorRT进行前向推理的engine模型文件.

注意：因为TensorRT不支持跨平台/系统/显卡，因此模型转换步骤需要和模型推理步骤在同一个环境下进行，否则生成的engine无效!!!

实现方式：

```
yolov5 -s
```

生成的模型文件存放地址：

```
D:\Project\AI_Test\build\AI_Engine\object_detector\yolov5
```

### Step3：实现yolov5（TensorRT FP16）前向推理

##### a.数据准备：

###### i.将生成的模型文件存放到指定路径：

```
D:\Project\AI_Test\config
```

###### ii.将测试图集存放到指定路径：

```
D:\Project\AI_Test\AI_data\testset_20201116
```

###### iii.指定测试结果信息存储路径：

```
D:\Project\AI_Test\AI_data\result
```

###### iv.修改config.yml文件信息：

```
%YAML:1.0

### Global model param
NET_DEPLOY: "None"
NET_MODEL: "yolov5s.engine"   //指定模型名称
NET_IN_W: 640                 //指定模型w
NET_IN_H: 640                 //指定模型h
LABEL_MAP: "label.txt"        //定义模型标签信息
NMS_THRESH: 0.3               //设置模型NMS置信度
CONF_THRESH: 0.72             //设置模型conf置信度信息
```

##### b.TensorRT FP16前向推理:

```
D:\Project\AI_Test\config\config.yml D:\Project\AI_Test\config\testset.txt D:\Project\AI_Test\AI_data\result -m=6 -g=0
```

完成！

## 四、注意事项：

1.在执行文件时，可能会遇到找不到object_detector.dll文件的信息，进行如下操作即可。

![image-20201225103436102](C:\Users\ZHANGWENTAO\AppData\Roaming\Typora\typora-user-images\image-20201225103436102.png)





