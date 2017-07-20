## 环境
* python & pip
  go to [python](https://www.python.org) to install Python, use follow command install pip  
  ```bash
  sudo easy_install pip
  ```

* virtualenv
    ```shell
    pip install --upgrade virtualenv
    ```

* tensorflow

  ```shell
  virtualenv --system-site-packages ~/tensorflow
  source ~/tensorflow/bin/activate
  pip install --upgrade tensorflow
  // or pip install --upgrade tensorflow-gpu
  ```

## 基本  

* 执行代码
    ```bash
    python helloworld.py
    ```

* 中文兼容
    ```python
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    ```
  
## 分类
* Session执行Graph
    ```
    Graph(图)
    |———— optimizer(op 节点)
          |———— Variable(变量), constant(常量)
          |———— 运算
          |———— train(训练)
    Session(会话)
    ```

## 用法
1. 基本结构
    ```python
    # 构造图，计算之类的
    # 使用Session.run() 执行
    # 释放
    ```
    见例子[helloworld](helloworld.py)

2. 使用 `with` 代码块在Session使用完后自动关闭以释放资源
    ```python
    with tf.Session() as sess:
      result = sess.run([product])
      print result
    ```

3. 指定CPU，GPU计算
    ```python
    with tf.device("/gpu:1"):
      matrix1 = tf.constant([[3., 3.]])
      matrix2 = tf.constant([[2.],[2.]])
      product = tf.matmul(matrix1, matrix2)
    ```

4. [`InteractiveSession`交互式](interactive_session.py)  
    *其实就是一个公共变量保存状态（很奇怪怎么保证线程安全)*

## Other
- Anaconda 、Ipython notebook etc.~~
  go to [continuum](https://www.continuum.io) to install Anaconda
- active ipython:
  ```shell
  ipython notebook
  ```
- create environment named tersorflow, active and deactive it
  ```shell
  conda create -n tensorflow python=2.7
  source activate tensorflow
  source deactivate
  ```
- active tensorboard 
  ```shell
  python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory`
  ```