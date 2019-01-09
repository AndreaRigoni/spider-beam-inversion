#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from strike_imgutils import *


# PROVA GENERAZIONE PARALLELA CON MAP E PY_FUNC
# def gen(x):
#     img = np.random.normal(size=1000000)
#     img = np.random.normal(size=1000000)
#     img = np.random.normal(size=1000000)
#     img = np.random.normal(size=1000000)
#     img = np.random.normal(size=1000000)
#     img = np.random.normal(size=1000000)
#     img = np.reshape(img, (1000, 1000))
#     return img
# ds = tf.data.Dataset.range(100).map(lambda x: tf.py_func(gen, [x], (tf.float64)) ,num_parallel_calls=8).batch(100)
# it = ds.make_one_shot_iterator()
# with tf.Session() as sess:
#     x = sess.run(it.get_next())
#     # print('i:',i)
#     print('done')
#     Strike_PlotUtils.show_images(x)
# plt.show()
# exit()



# PROVA GENERATOR DIVERSI 
#
#
# myg = MyGen(dataset_size=1000, n_gauss=1)
myg = EinsumGen(dataset_size=1000, n_gauss=1)
myg.read_from_generator()
# myg.read_from_generator2()
# myg.read_from_generator3()
# myg.write_to_files('gsk1p')

ds = myg.dataset.batch(4)
it = ds.make_one_shot_iterator()
st = time.time()
with tf.Session() as sess:
    l,x = sess.run(it.get_next())
    Strike_PlotUtils.show_images(x, xys2=l)
et = time.time()
print('elapsed: ',et-st)
plt.show()
exit()


# WITE FILES 1 GAUSS
#
# myg = MyGen(dataset_size=20000, n_gauss=1)
# myg.write_to_files('gsk1')
# exit()




# RUN DATASET #
ds = myg.read_from_files('gsk2')

# Train Parameters
batch_size    = 4
learning_rate = 0.0005
num_steps     = 60
display_step  = 1
GAIN          = 2
drop_out      = 0
n_gauss       = myg.cfg.n_gauss
n_classes     = 5 * n_gauss # FIX



# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------
# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # MNIST data input is a 1-D vector of  784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        imgX, imgY = myg.cfg.img_size
        
        x = tf.reshape(x, shape=[-1, imgY, imgX, 1])
        x = tf.image.convert_image_dtype(x,tf.float32)
        x = x / tf.reduce_max(x)

        # Convolution Layer 
        conv = tf.layers.conv2d(x, 32, 10, activation=tf.nn.relu, strides=2)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv = tf.layers.max_pooling2d(conv, 2, 2)

        # Convolution Layer 
        conv = tf.layers.conv2d(conv, 32, 10, activation=tf.nn.relu, strides=2)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv = tf.layers.max_pooling2d(conv, 2, 2)        

        # Convolution Layer 
        conv = tf.layers.conv2d(conv, 32, 10, activation=tf.nn.relu, strides=2)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv = tf.layers.max_pooling2d(conv, 2, 2)        

        # Flatten the data to a 1-D vector for the fully connected layer
        fc = tf.contrib.layers.flatten(conv)

        # Fully connected layer
        fc = tf.layers.dense(fc, 1024)
        fc = tf.layers.dropout(fc, rate=dropout, training=is_training)        

        # Fully connected layer
        fc = tf.layers.dense(fc, 1024)
        fc = tf.layers.dropout(fc, rate=dropout, training=is_training)        

        # Fully connected layer
        fc = tf.layers.dense(fc, 500)
        fc = tf.layers.dropout(fc, rate=dropout, training=is_training)        

        # Output layer, class prediction
        out = tf.layers.dense(fc, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        # out = tf.nn.softmax(out) if not is_training else out
    return out



# Automatically refill the data queue when empty
ds = ds.repeat()
# Create batches of data
ds = ds.batch(batch_size)
# Prefetch data for faster consumption
ds = ds.prefetch(batch_size)

# iterator
it = ds.make_initializable_iterator()

# variables
xys, X = it.get_next()

# Create a graph for training
logits_train = conv_net(X, n_classes, drop_out, reuse=False, is_training=True)

# Create another graph for testing that reuse the same weights, but has
# different behavior for 'dropout' (not applied).
logits_test = conv_net(X, n_classes, drop_out, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = GAIN * tf.reduce_prod(tf.losses.mean_squared_error(xys,logits_train))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op  = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
# correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(xy, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# accuracy = tf.reduce_mean(tf.losses.mean_squared_error(xy,logits_test))
accuracy = GAIN * tf.reduce_prod(tf.losses.mean_squared_error(xys,logits_test)) 

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    sess.run(it.initializer)
    
    # Training cycle
    for step in range(1, num_steps + 1):

        # Run optimization
        sess.run(train_op)
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            # (note that this consume a new batch of data)
            loss, acc = sess.run([loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.6f}".format(acc))



###
### TEST trained network
###
with tf.Session() as sess:
    ds = myg.read_from_generator()
    ds = ds.batch(36)
    it = ds.make_one_shot_iterator()
    xys, X = sess.run(it.get_next())
    sample_out = sess.run(conv_net(X, n_classes, drop_out, reuse=True, is_training=False))
    Strike_PlotUtils.show_images(X,sample_out).suptitle("TEST")

plt.show()






