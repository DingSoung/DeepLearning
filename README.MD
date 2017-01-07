---
title: 深度学习整理
date:	2017.01.03
tags: [C,C++]
---

# 环境
## Anaconda(包涵Ipython NoteBook，)
```bash
curl -O https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
sh Anaconda2-4.2.0-Linux-x86_64.sh
```
## Ipython NoteBook
```bash
ipython notebook
```
## python, centOS 7,MAC自带
```bash
sudo easy_install pip
```
## TensFlow
* 安装
```bash
 pip install tensorflow
 pip install tensorflow-gpu
 ```
* 使用Anaconda操作Tensorflow
```bash
# 创建名为tensflow的环境，python
conda create -n tensorflow python=2.7
# 激活
source activate tensorflow
# 执行文件
# 关闭
source deactivate
```

* TensBoard可视化  
 <http://127.0.0.1:6006>