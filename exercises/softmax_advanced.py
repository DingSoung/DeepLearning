#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 训练模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# InteractiveSession 允许使用traub_step.run
tf.InteractiveSession().run(tf.global_variables_initializer())  # 如果你没有使用InteractiveSession，那么你需要在启动session之前构建整个计算图，然后启动该计算图

# 开始训练
for i in range(1000):
      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# 评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
