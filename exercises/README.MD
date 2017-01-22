
## 代码
1. [helloworld](helloworld.py)
2. [basic_usage](basic_usage.py)
3. [variables_state](variables_state.py)
3. [feed](feed.py)
3. [interactive_session](interactive_session.py)
3. [example](example.py)
4. [softmax](softmax.py)
5. [softmax_advanced](softmax_advanced.py)
6. [softmax_convolution](softmax_convolution.py)
7. [tensorflow_mechnics](tensorflow_mechnics.py)

## 基本
1. 执行代码

    ```bash
    python helloworld.py
    ```

2. 中文兼容

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

2. 使用 "with" 代码块在Session使用完后自动关闭以释放资源

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