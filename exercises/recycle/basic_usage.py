#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])  # 常量op 1x2 矩阵
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)  # 矩阵乘法

'''
# 启动默认图.
sess = tf.Session()

# 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op)的执行
result = sess.run(product)
print result

# 任务完成, 关闭会话.
sess.close()
'''

# 使用 "with" 代码块 来自动完成关闭动作
with tf.Session() as sess:
    result = sess.run([product])
    print result
