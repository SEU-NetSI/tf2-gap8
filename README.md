## 环境配置
https://seunetsi.feishu.cn/wiki/wikcnlM1ntLRl1IwgwQJkqaCZJi

搭建GAPSDK环境，本项目的测试环境基于gap_sdk v4.12.0

## 训练模型
https://github.com/aqqz/tf2.git

基于该项目，使用TensorFlow 2.0训练并使用TensorFlow Lite 8bit量化得到tflite模型

## 移植
修改Makefile中的`TRAINED_MODEL`变量为对应的tflite模型

## 编译运行
```bash
make clean all run
```

## 烧录至aideck
```bash
make clean all image flash
```