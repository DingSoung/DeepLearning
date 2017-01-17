#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import math

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
batch_size = 16

data_sets = input_data.read_data_sets('MNIST_data/', one_hot=True)

images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

# Inference
hidden1_units = 1
with tf.name_scope('hidden1') as scope:
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')

hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
logits = tf.matmul(hidden2, weights) + biases

# Loss
batch_size = tf.size(labels)
labels = tf.expand_dims(labels, 1)
indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
concated = tf.concat(1, [indices, labels])
onehot_labels = tf.sparse_to_dense(
    concated, 
    tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits,
    onehot_labels,
    name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

# Train
tf.scalar_summary(loss.op.name, loss)
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

# Graph
#with tf.Graph().as_default():

# sessions
sess = tf.Session()  #with tf.Session() as sess:  # 限制作用域
init = tf.initialize_all_variables()
sess.run(init)
for step in xrange(max_steps):
    sess.run(train_op)

images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)

feed_dict = {
    images_placeholder: images_feed,
    labels_placeholder: labels_feed,
}

for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(
        data_sets.train,images_placeholder,
        labels_placeholder)
    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
    if step % 100 == 0:
        print 'Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration)

# for TensorBoard
summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
summary_str = sess.run(summary_op, feed_dict=feed_dict)
summary_writer.add_summary(summary_str, step)

# save checkpoint
saver = tf.train.Saver()
saver.save(sess, FLAGS.train_dir, global_step=step)
saver.restore(sess, FLAGS.train_dir)

# evalute
print 'Training Data Eval:'
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.train)
print 'Validation Data Eval:'
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.validation)
print 'Test Data Eval:'
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.test)

#eva Graph
test_all_images, test_all_labels = get_data(train=False)
eval_correct = mnist.evaluation(logits, labels_placeholder)
eval_correct = tf.nn.in_top_k(logits, labels, 1)

# eva output
for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / float(num_examples)
    print 'Num examples: %d  Num correct: %d  Precision @ 1: %0.02f' % (num_examples, true_count, precision)