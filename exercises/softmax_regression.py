#softmax回归

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# Softmax回归模型 y = softmax(W .* x + b)
W = tf.Variable(tf.zeros([784,10])) # 操作符号变量
x = tf.placeholder("float", [None, 784])  # 占位符 后面从mnist替换
b = tf.Variable(tf.zeros([10])) # 操作符号变量
y = tf.nn.softmax(tf.matmul(x,W) + b)  # 预测的值

# 评估指标
y_ = tf.placeholder("float", [None,10])  # 实际确值 后面从mnist替换
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  #交叉熵 成本函数

# 训练模型
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # op 梯度下降算法
init = tf.global_variables_initializer()   # op  初始化我们创建的变量

# 启动模型 初始化变量
sess = tf.Session()
sess.run(init)  # 执行初始化操作

# 数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in range(1000):  #循环训练1000次
      batch_xs, batch_ys = mnist.train.next_batch(100)  # 随机抓取训练数据中的100个批处理数据点
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 用这些数据点作为参数替换之前的占位符来运行

# 评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
