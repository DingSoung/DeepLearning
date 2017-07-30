#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

# 原始数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Download Done!")

# y = softmax(W.*x + b)模型
W = tf.Variable(tf.zeros([784, 10]))
x = tf.placeholder(tf.float32, [None, 784])
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 优化器
y_ = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(y_*tf.log(y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化变量 并循环训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))