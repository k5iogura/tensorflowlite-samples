# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from mnist import MNIST

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#SESS
sess = tf.InteractiveSession()

net = MNIST()
y_  = tf.placeholder("float", shape=[None, 10])

#Saver
saver = tf.train.Saver()

#Train
cross_entropy = -tf.reduce_sum(y_ * tf.log(net.y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(net.y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

train_count=20000
train_count=200
for i in range(train_count):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            net.x: batch[0], y_: batch[1], net.keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={net.x: batch[0], y_: batch[1], net.keep_prob: 0.5})

print "test accuracy %g" % accuracy.eval(
    feed_dict={
        net.x: mnist.test.images, y_: mnist.test.labels, net.keep_prob: 1.0
    }
)

# Saving Model
print("Saving Model Stage")
saver.save(sess,"./ckpt/mnist.ckpt",global_step=100)
frzdef = tf.compat.v1.graph_util.convert_variables_to_constants(
    sess,
    sess.graph_def,
    ['outputX']
)

prefix = 'mnist_'

with open(prefix+'frozen.pb', 'wb') as f:
    f.write(frzdef.SerializeToString())

tf.train.write_graph(
    sess.graph_def, '.',
    prefix+'frozen.pbtxt',
    as_text=True
)

print("Fin")
