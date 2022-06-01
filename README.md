## 环境配置
查看https://seunetsi.feishu.cn/wiki/wikcnlM1ntLRl1IwgwQJkqaCZJi搭建GAPSDK环境，本项目的测试环境基于gap_sdk v4.12.0

## 训练模型
在https://github.com/aqqz/tf2.git项目，使用python训练得到预量化的tflite模型

## 移植
修改Makefile中的`TRAINED_MODEL`变量为对应的tflite模型

## 运行
在项目Makefile目录下
```bash
make clean all run
```