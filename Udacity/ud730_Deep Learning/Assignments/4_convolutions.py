# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:02:58 2018

@author: Arc
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range




pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
#Reformat into a TensorFlow-friendly shape:
# convolutions need the image data formatted as a cube (width by height by #channels)
# labels as float 1-hot encodings.
  
image_size = 28
num_labels = 10
num_channels = 1 # grayscale


def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  
  # labels[:,None]将 labels的shape由(n,)变成(n,1), 
  # np.arange(num_labels) = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  # np.arange(num_labels) == labels[:,None] 得到一个 m x 10 的二维array，每一行都是一个one-hot编码
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32) 
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def maxpool2d(data, k = 2, s = 2, padding = 'SAME'):
  """
  kernel size: k, stride: s
  """
  return tf.nn.max_pool(data,[1,k,k,1], [1,s, s, 1], padding)
      
# Let's build a small network with two convolutional layers, 
# followed by one fully connected layer. Convolutional networks are more expensive computationally, 
# so we'll limit its depth and number of fully connected nodes.

#batch_size = 16
#patch_size = 5
#depth = 16   #output channel
#num_hidden = 64 # 隐藏层节点数
#
#graph = tf.Graph()
#
#with graph.as_default():
#
#  # Input data.
#  tf_train_dataset = tf.placeholder(
#    tf.float32, shape=(batch_size, image_size, image_size, num_channels)) # 默认输入数据格式为：batch, 图形高度，图形宽度，通道数量
#  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#  tf_valid_dataset = tf.constant(valid_dataset)
#  tf_test_dataset = tf.constant(test_dataset)
#  
#  # Variables.
#  layer1_weights = tf.Variable(tf.truncated_normal(   # 定义第一层的卷积核
#      [patch_size, patch_size, num_channels, depth], stddev=0.1))
#  layer1_biases = tf.Variable(tf.zeros([depth]))  # biases尺寸必须与weights的depth相同
#  layer2_weights = tf.Variable(tf.truncated_normal(  # 第二层卷积核
#      [patch_size, patch_size, depth, depth], stddev=0.1))
#  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
#  layer3_weights = tf.Variable(tf.truncated_normal( # shape: (7*7*depth, 64) ，全连接层
#      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
#  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
#  layer4_weights = tf.Variable(tf.truncated_normal(
#      [num_hidden, num_labels], stddev=0.1))
#  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
#  
#  # Model. 每一层由 WX+b -> activation 变成了 conv2d(X)+b -> activation
#  def model(data):
#     #conv2d 数据默认格式为"NHWC", 数据按这样的顺序存储：[batch, in_height, in_width, in_channels]
#     # strides = [1,2,2,1], 在input = data上采样时在height和width上采样的步长都为2.
#     # output: feature map 
#     # shape: [batch, out_height, out_width, output_channel] = [16, 14, 14, 16]
#    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
#    hidden = tf.nn.relu(conv + layer1_biases)
#    # 第二层
#    # input: hidden shape (16, 14, 14, 16) ; output shape: (16, 7, 7, 16)
#    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
#    hidden = tf.nn.relu(conv + layer2_biases)
#    shape = hidden.get_shape().as_list() # [16, 7, 7, 16], 经过两个 stride = 2 的 CNN，图像尺寸变成 image_size // 4 * image_size // 4 
#    # shape[0] = batch, 对应一个batch中的样本数量，因为第三层是全连接层，每个样本的数据要由三维降成一维即：shape[1] * shape[2] * shape[3]
#    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases) # 这一层是一个NN， ReLu(wx+b)
#    return tf.matmul(hidden, layer4_weights) + layer4_biases #输出层 y = Wx + b
#  
#  # Training computation.
#  logits = model(tf_train_dataset)
#  loss = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))
#    
#  # Optimizer.
#  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#  
#  # Predictions for the training, validation, and test data.
#  train_prediction = tf.nn.softmax(logits)
#  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
#  test_prediction = tf.nn.softmax(model(tf_test_dataset))
#
#  
#num_steps = 1001
#
#with tf.Session(graph=graph) as session:
#  tf.global_variables_initializer().run()
#  print('Initialized')
#  for step in range(num_steps):
#    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#    batch_labels = train_labels[offset:(offset + batch_size), :]
#    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
#    _, l, predictions = session.run(
#      [optimizer, loss, train_prediction], feed_dict=feed_dict)
#    if (step % 50 == 0):
#      print('Minibatch loss at step %d: %f' % (step, l))
#      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#      print('Validation accuracy: %.1f%%' % accuracy(
#        valid_prediction.eval(), valid_labels))
#  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  
###################################################################################
# Problem 1

# The convolutional model above uses convolutions with stride 2 to reduce the dimensionality. 
# Replace the strides by a max pooling operation (`nn.max_pool()`) of stride 2 and kernel size 2.

batch_size = 16
patch_size = 5
depth = 16   #output channel
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

 # Input data.
 tf_train_dataset = tf.placeholder(
   tf.float32, shape=(batch_size, image_size, image_size, num_channels)) # 默认输入数据格式为：batch, 图形高度，图形宽度，通道数量
 tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
 tf_valid_dataset = tf.constant(valid_dataset)
 tf_test_dataset = tf.constant(test_dataset)

 # Variables.
 layer1_weights = tf.Variable(tf.truncated_normal(   # 定义第一层的卷积核, shape= (5,5,1,16)
     [patch_size, patch_size, num_channels, depth], stddev=0.1))
 layer1_biases = tf.Variable(tf.zeros([depth]))
 layer2_weights = tf.Variable(tf.truncated_normal(  # 第二层卷积核
     [patch_size, patch_size, depth, depth], stddev=0.1))
 layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
 layer3_weights = tf.Variable(tf.truncated_normal( # shape: (7*7*depth, 64) ，全连接层
     [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
 layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
 layer4_weights = tf.Variable(tf.truncated_normal(
     [num_hidden, num_labels], stddev=0.1))
 layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))




 # Model. 每一层由 WX+b -> activation 变成了 conv2d(X)+b -> activation
 def model(data):
    # output: feature map
    # shape: [batch, out_height, out_width, output_channel] = [16, 28, 28, 16]
   conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
   # maxpool shape: (16, 14, 14, 16)
   conv = maxpool2d(conv)
   hidden = tf.nn.relu(conv + layer1_biases)
   # 第二层
   # input: hidden shape (16, 14, 14, 16) ; output shape: (16, 14, 14, 16)
   conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
   # maxpool shape: (16,7, 7, 16)
   conv = maxpool2d(conv)
   hidden = tf.nn.relu(conv + layer2_biases)
   shape = hidden.get_shape().as_list() # [16, 7, 7, 16], 经过两个 stride = 2 的 CNN，图像尺寸变成 image_size // 4 * image_size // 4
   # shape[0] = batch, 对应一个batch中的样本数量，因为第三层是全连接层，每个样本的数据要由三维降成一维即：shape[1] * shape[2] * shape[3]
   reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]]) # shape(16, 784)
   hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases) # shape (16, 64)
   return tf.matmul(hidden, layer4_weights) + layer4_biases #输出层 y = Wx + b (16, 10)

 # Training computation.
 logits = model(tf_train_dataset)
 loss = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

 # Optimizer.
 optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

 # Predictions for the training, validation, and test data.
 train_prediction = tf.nn.softmax(logits)
 valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
 test_prediction = tf.nn.softmax(model(tf_test_dataset))


num_steps = 1001

with tf.Session(graph=graph) as session:
 tf.global_variables_initializer().run()
 print('Initialized')
 for step in range(num_steps):
   offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
   batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
   batch_labels = train_labels[offset:(offset + batch_size), :]
   feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
   _, l, predictions = session.run(
     [optimizer, loss, train_prediction], feed_dict=feed_dict)
   if (step % 50 == 0):
     print('Minibatch loss at step %d: %f' % (step, l))
     print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
     print('Validation accuracy: %.1f%%' % accuracy(
       valid_prediction.eval(), valid_labels))
 print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  
##################################################################################
  
# Problem 2
# Try to get the best performance you can using a convolutional net. 
# Look for example at the classic LeNet5 architecture, adding Dropout, 
# and/or adding learning rate decay.
  
batch_size = 16
patch_size = 5
depth = 16   #output channel
num_hidden = 64
beta = 0.0007


graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels)) # 默认输入数据格式为：batch, 图形高度，图形宽度，通道数量
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(   # 定义第一层的卷积核, shape= (5,5,1,16)
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth])) 
  layer2_weights = tf.Variable(tf.truncated_normal(  # 第二层卷积核
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal( # shape: (7*7*depth, 64) ，全连接层
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  

  
  # Model. 每一层由 WX+b -> activation 变成了 conv2d(X)+b -> activation
  keep_prob = tf.placeholder(tf.float32)
  def model(data, dropout = False, keep_prob = keep_prob):
     # output: feature map 
     # shape: [batch, out_height, out_width, output_channel] = [16, 28, 28, 16]
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    # maxpool shape: (16, 14, 14, 16)
    conv = maxpool2d(conv)
    hidden = tf.nn.relu(conv + layer1_biases)
    # 第二层
    # input: hidden shape (16, 14, 14, 16) ; output shape: (16, 14, 14, 16)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
    # maxpool shape: (16,7, 7, 16)
    conv = maxpool2d(conv)
    hidden = tf.nn.relu(conv + layer2_biases)
    
    # dropout
    if dropout:
        hidden = tf.nn.dropout(hidden, keep_prob)
    
    shape = hidden.get_shape().as_list() # [16, 7, 7, 16], 经过两个 stride = 2 的 CNN，图像尺寸变成 image_size // 4 * image_size // 4 
    # shape[0] = batch, 对应一个batch中的样本数量，因为第三层是全连接层，每个样本的数据要由三维降成一维即：shape[1] * shape[2] * shape[3]
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]]) # shape(16, 784)
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases) # shape (16, 64)
        
    return tf.matmul(hidden, layer4_weights) + layer4_biases #输出层 y = Wx + b (16, 10)
  
  # Training computation.
  logits = model(tf_train_dataset, dropout = True)
  logits_predict = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))
  l2_loss = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer4_weights)
  loss = loss + beta*l2_loss 
    
    
  # Optimizer.
  learning_rate = 0.05
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.92, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits_predict)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, dropout = False))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, dropout = False))

  
num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))  
  
