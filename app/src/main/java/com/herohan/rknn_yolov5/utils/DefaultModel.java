package com.herohan.rknn_yolov5.utils;

public enum DefaultModel {
    YOLOV5S_FP("yolov5s-fp.rknn"),  // RK3568
    YOLOV5S_640_INT8("yolov5s-640-640.rknn"),   // RK3576
    YOLOV5S_RELU_INT8("yolov5s_relu.rknn"),  // 测试
    YOLOV5S_APK_INT8("yolov5s_apk.rknn"),  // 测试
    YOLOV5S_INT8("yolov5s-int8.rknn");  // RK3568

    public final String name;

    DefaultModel(String name) {
        this.name = name;
    }
}
