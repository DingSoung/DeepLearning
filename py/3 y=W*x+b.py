#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# 随机数据
x_data = np.random.rand(500).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# y = W * x + b模型
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = x_data * W + b

# 配置优化器
loss = tf.reduce_mean(tf.square(y_data - y))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量 并循环训练
sess = tf.Session() 
sess.run(tf.global_variables_initializer())
for step in range(10000):
    sess.run(train)
    if step % 50 == 0:
        print(step, sess.run(W), sess.run(b))