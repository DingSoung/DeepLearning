#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(500).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 图
# 变量
Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = x_data * Weight + biases

# op 节点
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# sessions
sess = tf.Session()
sess.run(init)


for step in range(1001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weight), sess.run(biases))