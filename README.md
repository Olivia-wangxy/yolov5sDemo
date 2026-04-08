本Demo是在该博客提供Demo的基础上进行修改使用的https://blog.csdn.net/hanshiying007/article/details/137332023

修改点：
1. 在官方github中找到支持RK3576开发板的RKNN模型，添加到assets中（yolov5s-640-640.rknn）
2. 为确保算法是在NPU上运行的，对输入图片做了INT8量化处理，修改位置为yolov5.cc/inference_yolov5_model

运行效果：
1. 原Demo中的yolov5s-int8.rknn可在RK3568中的NPU运行，用时65ms左右（rknn_run done, Elapse = 64.13 ms, FPS = 15.59）
2. 使用yolov5s-640-640.rknn在RK3576上对同一张图多次识别，用时约18ms~60ms，平均在40ms左右（normal Elapse Time = 41.07 ms, FPS = 24.35）

# 1. 背景
需要调研RK3576的开发板的NPU运算能力是否满足团队需求：
1. 初期调研内容：开发板的NPU性能是否能满足运行主流的图像识别的算法（比如YOLO）

# 2. Android运行Yolov5s算法的Demo
## 2.1 查阅资料
官方airockchip的github中提供机器学习相关资料，该链接中对模型等内容做了详细介绍，可供参考https://github.com/airockchip/rknn_model_zoo/blob/main/README_CN.md
为了在Android开发板上运行图像识别算法，翻阅网上开源资料做了很多尝试，官方提供的demo在该链接中https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2，其中能给Android使用的demo只有一个，在尝试中存在各种问题，于是在博客中寻找其他demo，最终在该博客提供的demo的基础上修改实现了yolov5s图像识别算法https://blog.csdn.net/hanshiying007/article/details/137332023。
## 2.2 运行Demo
从博客中下载的Demo中带的rknn模型是适用于RK3568开发板的，其他开发板运行会报错，并且Demo原本的逻辑只支持FP模型，未支持INT8模型，由于调研需求是判断开发板NPU性能是否满足，所以需要使用INT8量化模型来确保在NPU上运行yolov5s算法。
因此在Demo基础上的两个需求点为：① 找支持RK3576开发板的RKNN（INT8）模型替换，② 对输入的图片进行INT8量化后再通过RKNN推理。
1. 首先解决第一个问题，在翻看官方资料时，rknpu2/examples/rknn_yolov5_demo/model/RK3576中发现了支持RK3576的rknn模型yolov5s-640-640.rknn，先导入Demo中测试下（结果证明找对了）。
2. 运行yolov5s-640-640.rknn过程中有很多报错问题需要处理，其实是在处理问题的过程中才搞清楚需要转成INT8模型才能用，于是借助AI的力量修改出了可运行的yolov5sDemo。
本地修改后的Demo已上传至github的main分支：https://github.com/Olivia-wangxy/yolov5sDemo，在不同开发板上运行的话需要在MainActivity中修改yolov5Detect.init传入的模型，否则开发板和模型不匹配的话会报错，无法正常运行。
## 2.3 注意点
在Demo中选择好图片点击识别会执行JNI的detect方法：
1. 将图片裁剪成rknn要求的尺寸（640×640）
2. Android 侧图像通常是 RGBA8888，但 YOLO / RKNN 模型输入通常是 RGB888，需要注意scr和dst的图片格式，否则与rknn要求的输入不匹配会报错
3. 如果rknn是INT8模型的话，输入rknn的图片数据需要量化处理为INT8（NPU运行模型），FP32模型的话只能使用CPU运行
4. 运行的开发板需要跟rknn模型的类型匹配，否则初始化失败
5. NPU跑完算法后CPU处理图片，处理时可设置识别类型数量等内容，可在post_process接口中修改

# 3. 训练模型运行
官方提供的RKNN模型支持80类常见物体识别，团队未来使用大概率是菜品识别方向的应用，所以需要将识别类型训练为菜品相关的类别。先查阅资料实现一个简单的训练模型，识别自定义的物品类别。
从训练模型到在Android开发板上运行模型需要以下阶段：训练模型生成.pt文件 ——> 转为跨平台的.onnx模型（CPU可运行） ——> 转为NPU可运行的.rknn模型
## 3.1 搭建环境
训练模型需要Linux环境，可在Windows系统中安装WSL，因为参考资料普遍使用的是Ubuntu-20.04和python38，使用Ubuntu-22.04会遇到各种问题，所以这里以下载Ubuntu-20.04为例，步骤如下：
#### 下载安装WSL：
1. 查看发行版本：wsl --list --online
2. 安装WSL：wsl --install -d Ubuntu-20.04（默认C盘，下载好后重启PC）
3. 设置账号密码
4. 查看版本：
lsb_release -a
python3 --version
5. 执行python3.8：
sudo apt update
sudo apt install -y python3-pip python3-venv
6. 创建/进入虚拟环境：
python3 -m venv rknn_env
source rknn_env/bin/activate
#### 迁移WSL到D盘（避免占用C盘空间）：
1. 关闭并导出WSL：
wsl --shutdown
wsl --export Ubuntu-20.04 D:\WSL\Ubuntu-20.04.tar
2. 注销C盘WSL导入D盘：
wsl --unregister Ubuntu-20.04
wsl --import Ubuntu-20.04 D:\WSL\Ubuntu-20.04 D:\WSL\Ubuntu-20.04.tar
wsl -l -v
rm D:\WSL\Ubuntu-20.04.tar
#### WSL操作：
1. 查看用户（root/home）：cat /etc/passwd | grep home
2. 指定wsl实例/用户：
wsl（进入默认wsl实例）
wsl -d Ubuntu-20.04
wsl -d Ubuntu-22.04 -u wangxinyu33
3. 切换用户：su - wangxinyu33
修改默认wsl：wsl --set-default Ubuntu-20.04
4. 修改wsl默认用户：ubuntu2204.exe config --default-user wangxinyu33
## 3.2 准备数据集
### 3.2.1 准备制作数据集的工具
1. 进入虚拟环境：source rknn_env/bin/activate
2. 安装LableImg：
sudo apt update
sudo apt install -y python3-pyqt5 pyqt5-dev-tools
pip install labelImg
pip show labelImg
3. 打开LableImg制作数据集：labelImg
### 3.2.2 制作数据集
Yolov5训练用的数据集格式是yolo格式的，Yolov3训练用的数据集格式是VOC格式，如果希望数据集复用，可以制作VOC格式数据集转为yolo格式数据集。可在网上下载COCO128图片集（链接：https://pan.baidu.com/s/1MYSzPkPVUxpE1wt7zUxO0g 提取码：8r4i），使用LabelImg工具给图片打标签制作数据集，详情可参考博客https://blog.csdn.net/qq_40280673/article/details/125158582?spm=1001.2014.3001.5502。
## 3.3 训练模型
训练yolov5模型需要先下载对应的代码，将数据集转为yolov5需要的格式，然后进行模型训练，训练后会生成.pt格式的训练模型，该模型可以使用CPU直接运行出自定义的图像识别结果。
### 3.3.1 下载训练代码
训练模型的代码下载自博客https://developer.aliyun.com/article/1626014，同时参考了另一篇博客训练模型的内容https://blog.csdn.net/qq_40280673/article/details/125168930?spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-125168930-blog-143647392.235%5Ev43%5Epc_blog_bottom_relevance_base6&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-125168930-blog-143647392.235%5Ev43%5Epc_blog_bottom_relevance_base6&utm_relevant_index=12。
训练代码要放 Ubuntu-20.04 的用户目录下，先进入虚拟环境再下载python、openCV等工具，这样方便管理。代码和工具准备好后可以输入命令先测试代码链路是否正常：python detect.py --source data/images/bus.jpg --weights pretrained/yolov5s.pt，运行后会在输出路径中看到图像识别后的输出结果。代码链路没问题接下来就开始训练自己想要的模型。
<img width="2227" height="571" alt="image" src="https://github.com/user-attachments/assets/1fbe3465-4674-4ab3-a71c-e72e07f6f30a" />

### 3.3.2 转换数据集格式
根据第二篇博客的知道，将原图和打标签的文件放在yolov5-mask\VOCdevkit\VOC2007，然后运行AI生成的格式转换脚本voc2yolo.py即可将VOC格式的数据集转为yolo格式。
在WSL中运行脚本：python voc2yolo.py
```
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile

classes = ["westernfood","chinesefood"]        ##这里要写好标签对应的类
# classes=["ball"]

TRAIN_RATIO = 80     #表示将数据集划分为训练集和验证集，按照2:8比例来的


def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml' % image_id)
    out_file = open('VOCdevkit/VOC2007/YOLOLabels/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


wd = os.getcwd()
wd = os.getcwd()
data_base_dir = os.path.join(wd, "VOCdevkit/")
if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)
work_sapce_dir = os.path.join(data_base_dir, "VOC2007/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)
annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
if not os.path.isdir(annotation_dir):
    os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)
image_dir = os.path.join(work_sapce_dir, "JPEGImages/")
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
clear_hidden_files(image_dir)
yolo_labels_dir = os.path.join(work_sapce_dir, "YOLOLabels/")
if not os.path.isdir(yolo_labels_dir):
    os.mkdir(yolo_labels_dir)
clear_hidden_files(yolo_labels_dir)
yolov5_images_dir = os.path.join(data_base_dir, "images/")
if not os.path.isdir(yolov5_images_dir):
    os.mkdir(yolov5_images_dir)
clear_hidden_files(yolov5_images_dir)
yolov5_labels_dir = os.path.join(data_base_dir, "labels/")
if not os.path.isdir(yolov5_labels_dir):
    os.mkdir(yolov5_labels_dir)
clear_hidden_files(yolov5_labels_dir)
yolov5_images_train_dir = os.path.join(yolov5_images_dir, "train/")
if not os.path.isdir(yolov5_images_train_dir):
    os.mkdir(yolov5_images_train_dir)
clear_hidden_files(yolov5_images_train_dir)
yolov5_images_test_dir = os.path.join(yolov5_images_dir, "val/")
if not os.path.isdir(yolov5_images_test_dir):
    os.mkdir(yolov5_images_test_dir)
clear_hidden_files(yolov5_images_test_dir)
yolov5_labels_train_dir = os.path.join(yolov5_labels_dir, "train/")
if not os.path.isdir(yolov5_labels_train_dir):
    os.mkdir(yolov5_labels_train_dir)
clear_hidden_files(yolov5_labels_train_dir)
yolov5_labels_test_dir = os.path.join(yolov5_labels_dir, "val/")
if not os.path.isdir(yolov5_labels_test_dir):
    os.mkdir(yolov5_labels_test_dir)
clear_hidden_files(yolov5_labels_test_dir)

train_file = open(os.path.join(wd, "yolov5_train.txt"), 'w')
test_file = open(os.path.join(wd, "yolov5_val.txt"), 'w')
train_file.close()
test_file.close()
train_file = open(os.path.join(wd, "yolov5_train.txt"), 'a')
test_file = open(os.path.join(wd, "yolov5_val.txt"), 'a')
list_imgs = os.listdir(image_dir)  # list image files
prob = random.randint(1, 100)
print("Probability: %d" % prob)
for i in range(0, len(list_imgs)):
    path = os.path.join(image_dir, list_imgs[i])
    if os.path.isfile(path):
        image_path = image_dir + list_imgs[i]
        voc_path = list_imgs[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        label_name = nameWithoutExtention + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)
    prob = random.randint(1, 100)
    print("Probability: %d" % prob)
    if (prob < TRAIN_RATIO):  # train dataset
        if os.path.exists(annotation_path):
            train_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov5_images_train_dir + voc_path)
            copyfile(label_path, yolov5_labels_train_dir + label_name)
    else:  # test dataset
        if os.path.exists(annotation_path):
            test_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov5_images_test_dir + voc_path)
            copyfile(label_path, yolov5_labels_test_dir + label_name)
train_file.close()
test_file.close()
```

### 3.3.3 训练模型
在yolov5-mask\data中参考mask_data.yaml新建一个food_data.yaml配置文件
```
# Custom data for safety helmet


# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]

train: /home/wangxinyu33/yolov5-mask/VOCdevkit/images/train
val: /home/wangxinyu33/yolov5-mask/VOCdevkit/images/val

# number of classes
nc: 2

# class names
names: ['westernfood', 'chinesefood']
```
运行命令开始训练模型（需要较长时间）：python train.py --data food_data.yaml --cfg mask_yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 100 --batch-size 4 --device cpu
训练完成后会生成一个best.pt模型，使用命令测试可以看到训练模型按我们指定的物品种类进行图像识别：python detect.py  --weights runs/train/exp4/weights/best.pt --source data/images/3.jpg
<img width="2217" height="496" alt="image" src="https://github.com/user-attachments/assets/b0e09943-4fa5-424d-83e1-93591b21bcd5" />

到此，简单的训练模型已完成，接下来需要转换训练格式为NPU支持的格式，在Android开发板上运行。
## 3.4 转onnx模型
通过export.py脚本将pt模型转为onnx模型：python export.py --weights runs/train/exp4/weights/best.pt --imgsz 640 --batch-size 1 --device cpu --include onnx 
<img width="2143" height="1253" alt="image" src="https://github.com/user-attachments/assets/764e6e54-1182-4aa3-a7bb-28c7c4c1b5f7" />

在PC上先测试转换的onnx模型是否正常，借助AI生成脚本 yolov5_food.py（官方提供脚本为rknn_model_zoo\examples\yolov5\python\yolov5.py）
```
import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort

# ===================== 配置 =====================
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

CLASSES = ("westernfood", "chinesefood")

# ===================== 工具函数 =====================

def letterbox(im, new_shape=640, color=(114,114,114)):
    shape = im.shape[:2]
    ratio = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))

    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def nms_boxes(boxes, scores):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        ovr = inter/(areas[i]+areas[order[1:]]-inter)

        inds = np.where(ovr<=NMS_THRESH)[0]
        order = order[inds+1]

    return keep

def post_process(outputs, ratio, dwdh):
    pred = outputs[0][0]  # (25200, 7)

    boxes = pred[:, :4]
    obj_conf = pred[:, 4:5]
    class_conf = pred[:, 5:]

    scores = obj_conf * class_conf
    classes = np.argmax(scores, axis=1)
    scores = np.max(scores, axis=1)

    mask = scores >= OBJ_THRESH
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    if len(boxes) == 0:
        return None, None, None

    # xywh → xyxy
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:,0] = boxes[:,0] - boxes[:,2]/2
    boxes_xyxy[:,1] = boxes[:,1] - boxes[:,3]/2
    boxes_xyxy[:,2] = boxes[:,0] + boxes[:,2]/2
    boxes_xyxy[:,3] = boxes[:,1] + boxes[:,3]/2

    keep = nms_boxes(boxes_xyxy, scores)

    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    classes = classes[keep]

    # 还原到原图尺寸
    boxes_xyxy[:,[0,2]] -= dwdh[0]
    boxes_xyxy[:,[1,3]] -= dwdh[1]
    boxes_xyxy /= ratio

    return boxes_xyxy, classes, scores

# ===================== 主程序 =====================

def main(args):
    session = ort.InferenceSession(args.model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    image_files = [f for f in os.listdir(args.img_folder)
                   if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))]

    for img_name in image_files:
        img_path = os.path.join(args.img_folder, img_name)
        img0 = cv2.imread(img_path)
        if img0 is None:
            continue

        img, ratio, dwdh = letterbox(img0, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2,0,1)
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0

        outputs = session.run(None, {input_name: img})
        boxes, classes, scores = post_process(outputs, ratio, dwdh)

        print(f"\nIMG: {img_name}")
        if boxes is not None:
            for box, cls, score in zip(boxes, classes, scores):
                x1,y1,x2,y2 = box.astype(int)
                label = f"{CLASSES[cls]} {score:.2f}"
                print(label)

                cv2.rectangle(img0,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(img0,label,(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        if args.img_show:
            cv2.imshow("result", img0)
            cv2.waitKey(0)

        if args.img_save:
            os.makedirs("result", exist_ok=True)
            cv2.imwrite(os.path.join("result", img_name), img0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--img_show', action='store_true')
    parser.add_argument('--img_save', action='store_true')
    args = parser.parse_args()

    main(args)
```
用onnx模型进行图像识别：python yolov5_food.py --model_path runs/train/exp4/best.onnx --img_folder data/images/onnx_image --img_show
<img width="2228" height="652" alt="image" src="https://github.com/user-attachments/assets/101fbd41-e352-4875-924e-f49c0ea22eee" />


检查onnx推理结果
/rknn_model_zoo/examples/yolov5/python$ $ 
python yolov5.py --model_path ../model/yolov5s.onnx --img_folder ../model/ --img_show
python yolov5.py --model_path ../model/yolov5s_relu.onnx --img_folder ../model/ --img_show
<img width="2218" height="900" alt="image" src="https://github.com/user-attachments/assets/e6ddc392-b879-4e45-8d8e-5a4b2d8ad61e" />


onnx转rknn
/rknn_model_zoo/examples/yolov5/python$ $ 
python convert.py yolov5s.onnx rk3568 i8 yolov5s.rknn
python convert.py yolov5s.onnx rk3568 fp yolov5s_fp.rknn
<img width="2236" height="686" alt="image" src="https://github.com/user-attachments/assets/3fc47ab2-a5c0-4365-ba51-2de74268ec88" />

<img width="2218" height="964" alt="image" src="https://github.com/user-attachments/assets/a34114a7-ef72-4039-b233-8a508923b571" />


检查rknn推理结果
/rknn_model_zoo/examples/yolov5/python$ $ 
python rknn_output.py
<img width="1557" height="1256" alt="image" src="https://github.com/user-attachments/assets/3e19c435-a3ad-4e89-9dfe-e7c563874c48" />


查看onnx有几个输出，输出格式是否与官方一致
```
import onnx
model = onnx.load("best.onnx")
print(len(model.graph.output))

for output in model.graph.output:
    shape = []
    for dim in output.type.tensor_type.shape.dim:
        shape.append(dim.dim_value)
    print(output.name, shape)
```

<img width="1225" height="614" alt="image" src="https://github.com/user-attachments/assets/a7d10938-9583-4372-b21a-f9e8a77c1930" />


<img width="785" height="1255" alt="image" src="https://github.com/user-attachments/assets/bb72daa2-5cf4-4316-aaff-8361efb6e46d" />


检查激活函数算子
```
import onnx

model = onnx.load("yolov5s.onnx")

ops = set(node.op_type for node in model.graph.node)

print(ops)
```
<img width="2222" height="564" alt="image" src="https://github.com/user-attachments/assets/f94987fd-f6ac-4f8e-8360-ed21604b196c" />




## 3.5 转rknn模型
将.onnx格式的训练模型转换为.onnx模型需要下载Rockchip官方提供的模型转换和量化工具RKNN-Toolkit2。官方还提供了示例模型仓库RKNN Model Zoo，可以一起下载下来参考，其中包含了多种算法模型。
### 3.5.1 下载工具
1. 下载 RKNN-Toolkit2 仓库
git clone https://github.com/airockchip/rknn-toolkit2.git --depth 1
2. 下载 RKNN Model Zoo 仓库 
git clone https://github.com/airockchip/rknn_model_zoo.git --depth 1
3. 下载依赖
cd ~/rknn-toolkit2/rknn-toolkit2/packages/x86_64
pip install -r requirements_cp38-2.3.2.txt
4. 安装rknn-toolkit2
pip install rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
5. 进入Python交互环境检查安装结果 (是否报错)
python -c "from rknn.api import RKNN; print('RKNN OK')"
<img width="2226" height="524" alt="image" src="https://github.com/user-attachments/assets/e1938481-754f-4b46-bf06-b3e3f54f8a73" />

### 3.5.2 模型转换
工具准备好进入RKNN Model Zoo目录可参考其中的convert.py修改出符合自定义训练模型的格式转换脚本。
```
import sys
import os
from rknn.api import RKNN

# ==============================
# 固定参数（适配 food 模型）
# ==============================

TARGET_PLATFORM = 'rk3568'
DATASET_PATH = '../../../datasets/COCO/food_subset_53.txt'
OUTPUT_PATH = '../model/best_food_int8.rknn'

IMG_SIZE = 640

def check_file(path):
    if not os.path.exists(path):
        print(f'❌ File not found: {path}')
        exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_food.py best.onnx")
        exit(1)

    onnx_path = sys.argv[1]
    check_file(onnx_path)
    check_file(DATASET_PATH)

    print("====================================")
    print("  RK3568 YOLOv5 FOOD INT8 Converter ")
    print("====================================")

    rknn = RKNN(verbose=True)

    # 1️⃣ 配置模型
    print('--> Config model')
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=TARGET_PLATFORM,
        quantized_dtype='asymmetric_quantized-8'
    )
    print('done')

    # 2️⃣ 加载 ONNX
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=['images'],
        input_size_list=[[1, 3, IMG_SIZE, IMG_SIZE]]
    )

    if ret != 0:
        print('❌ Load ONNX failed')
        exit(ret)
    print('done')

    # 3️⃣ 构建 RKNN（INT8）
    print('--> Building RKNN model (INT8 quantization)')
    ret = rknn.build(
        do_quantization=True,
        dataset=DATASET_PATH
    )

    if ret != 0:
        print('❌ Build failed')
        exit(ret)
    print('done')

    # 4️⃣ 导出 RKNN
    print('--> Export RKNN model')
    ret = rknn.export_rknn(OUTPUT_PATH)

    if ret != 0:
        print('❌ Export failed')
        exit(ret)

    print('done')
    print(f'\n🎉 RKNN model saved to: {OUTPUT_PATH}')

    rknn.release()

if __name__ == '__main__':
    main()
```
其中用于量化的数据集至少需要50张图片，否则量化精度不够，因此在\rknn_model_zoo\datasets\COCO\中创建自定义模型对应的food_subset文件夹和food_subset_53.txt文件，将路径替换到convert_food.py脚本中，运行命令开始模型转换：python convert_food.py ../model/best.onnx
<img width="2225" height="746" alt="image" src="https://github.com/user-attachments/assets/011ef2d3-a2e6-48e3-a907-ebae6553bbd1" />

...
<img width="2233" height="673" alt="image" src="https://github.com/user-attachments/assets/2a84bb50-170e-4f2d-8d66-bce7f132169a" />

## 3.6 Demo适配训练模型


将OBJ_CLASS_NUM改为2，coco_80_labels_list.txt改为food_2_labels_list.txt











