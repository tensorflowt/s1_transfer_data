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
|---AI_Test
    |---AI_3rdparty
    |---AI_Engine
    |---AI_data
    |---config
    |---build
    |---CmakeLists
    |---README.md
```

## 三、工程使用教程：

### Step1：cmake.

![image-20201225082125506](C:\Users\ZHANGWENTAO\AppData\Roaming\Typora\typora-user-images\image-20201225082125506.png)

### Step2：编译.

![image-20201118171503404](C:\Users\ZHANGWENTAO\AppData\Roaming\Typora\typora-user-images\image-20201118171503404.png)

选择项目ALL_BUILD右键进行生成,生成的可执行文件存放在如下路径：

```
D:\Project\AI_Test\build\bin\ai_engine\Debug
```

![image-20201225082301736](C:\Users\ZHANGWENTAO\AppData\Roaming\Typora\typora-user-images\image-20201225082301736.png)

### Step3：实现yolov5（TensorRT FP16）前向推理

##### a.数据准备：

###### i.将生成的模型文件(wts)存放到指定路径：

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
NET_MODEL: "best_x_sunqt_1116_640x640x1.engine"    # 生成的引擎文件
NET_IN_W: 640                                      # 网络模型的输入宽度
NET_IN_H: 640                                      # 网络模型的输入高度
CLASS_NUM: 1                                       # 网络模型的检测类别数
LABEL_MAP: "label.txt"                             # 检测模型的实际预测类别名
WTS_FILE: "best_x_sunqt_1116_640x640x1.wts"        # 原始wts模型
NMS_THRESH: 0.3                                    # NMS置信度
CONF_THRESH: 0.72								   # score置信
DET_INFO: "det_info.txt"                           # 检测结果存储文件（可不设置）
```

##### b.TensorRT FP16前向推理:

```
D:\Project\AI_Test\config\config.yml D:\Project\AI_Test\config\testset.txt D:\Project\AI_Test\AI_data\result -m=6 -g=0
```

**说明：**

1）当config文件夹中无yolov5s.engine文件时，会去寻找wts文件模型进行模型转换（wts2engine）

2）当config文件中有yolov5s.engine文件时，会直接调用进行前向推理

3）只需编译一次工程，后期模型更换后只需：①替换模型wts文件②更改config.yml配置文件

完成！

