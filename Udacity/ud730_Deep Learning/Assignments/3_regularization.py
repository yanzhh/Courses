# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:18:20 2018

@author: Arc
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

#Load data

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
  
# Reformat into a shape that's more adapted to the models we're going to train:

# data as a flat matrix,
# labels as float 1-hot encodings.
  
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


################################################################################
# Problem 1
# Introduce and tune L2 regularization for both logistic and neural network models. 
# Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. 
# In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t). 
# The right amount of regularization should improve your validation / test accuracy.

# Logistic model

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.

beta = 0.008

train_subset = 10000

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))
  l2_loss = tf.nn.l2_loss(weights)
  loss = loss + beta * l2_loss
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

# Now Let's run this computation and iterate:

num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# NN model
  
beta = 0.018
batch_size = 128

graph = tf.Graph()
with graph.as_default():
    
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables.
    hidden_nodes = 1024
    hidden_weights = tf.Variable( 
            tf.truncated_normal([image_size * image_size, hidden_nodes])) # 从截断的正态分布中输出随机值。x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    hidden_biases = tf.Variable(tf.zeros([hidden_nodes])) # size: (10,)
    
    weights = tf.Variable( # size: (28^2, 10)
            tf.truncated_normal([hidden_nodes, num_labels])) # 从截断的正态分布中输出随机值。x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    biases = tf.Variable(tf.zeros([num_labels])) # size: (10,)
    
    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    hidden_layer = tf.add(tf.matmul(tf_train_dataset, hidden_weights), hidden_biases)
    hidden_layer = tf.nn.relu(hidden_layer)
    logits = tf.matmul(hidden_layer, weights) + biases # Y = XW + b, size: (10000,10), 每个行向量对应一个sample的score，下一步进行softmax后对应于各个labels的概率
    loss = tf.reduce_mean( # 未指定第二个变量reduction_indices，因此在tensor的所有元素上求平均值
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))
    l2_loss = tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(weights)
    loss = loss + beta*l2_loss
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_valid_dataset, hidden_weights), hidden_biases))
    valid_prediction = tf.nn.softmax(
        tf.matmul(valid_hidden_layer, weights) + biases)
    test_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_test_dataset, hidden_weights), hidden_biases))
    test_prediction = tf.nn.softmax(tf.matmul(test_hidden_layer, weights) + biases) 



num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


###################################################################################
# Problem 2
# Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches.
# What happens?

beta = 0.018
batch_size = 128

graph = tf.Graph()
with graph.as_default():
    
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables.
    hidden_nodes = 1024
    hidden_weights = tf.Variable( 
            tf.truncated_normal([image_size * image_size, hidden_nodes])) # 从截断的正态分布中输出随机值。x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    hidden_biases = tf.Variable(tf.zeros([hidden_nodes])) # size: (10,)
    
    weights = tf.Variable( # size: (28^2, 10)
            tf.truncated_normal([hidden_nodes, num_labels])) # 从截断的正态分布中输出随机值。x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    biases = tf.Variable(tf.zeros([num_labels])) # size: (10,)
    
    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    hidden_layer = tf.add(tf.matmul(tf_train_dataset, hidden_weights), hidden_biases)
    hidden_layer = tf.nn.relu(hidden_layer)
    logits = tf.matmul(hidden_layer, weights) + biases # Y = XW + b, size: (10000,10), 每个行向量对应一个sample的score，下一步进行softmax后对应于各个labels的概率
    loss = tf.reduce_mean( # 未指定第二个变量reduction_indices，因此在tensor的所有元素上求平均值
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))
    l2_loss = tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(weights)
    loss = loss + beta*l2_loss
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_valid_dataset, hidden_weights), hidden_biases))
    valid_prediction = tf.nn.softmax(
        tf.matmul(valid_hidden_layer, weights) + biases)
    test_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_test_dataset, hidden_weights), hidden_biases))
    test_prediction = tf.nn.softmax(tf.matmul(test_hidden_layer, weights) + biases) 



num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset_limit = 300  #限制offset的范围
    offset = (step * batch_size) % offset_limit
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))



#################################################################################
# Problem 3
# Introduce Dropout on the hidden layer of the neural network. 
# Remember: Dropout should only be introduced during training, not evaluation, 
# otherwise your evaluation results would be stochastic as well. TensorFlow provides 
# nn.dropout() for that, but you have to make sure it's only inserted during training.

# What happens to our extreme overfitting case?

beta = 0.001
batch_size = 128

graph = tf.Graph()
with graph.as_default():
    
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables.
    hidden_nodes = 1024
    hidden_weights = tf.Variable( 
            tf.truncated_normal([image_size * image_size, hidden_nodes])) # 从截断的正态分布中输出随机值。x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    hidden_biases = tf.Variable(tf.zeros([hidden_nodes])) # size: (10,)
    
    weights = tf.Variable( # size: (28^2, 10)
            tf.truncated_normal([hidden_nodes, num_labels])) # 从截断的正态分布中输出随机值。x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    biases = tf.Variable(tf.zeros([num_labels])) # size: (10,)
    
    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    hidden_layer = tf.add(tf.matmul(tf_train_dataset, hidden_weights), hidden_biases)
    hidden_layer = tf.nn.relu(hidden_layer)
    
    # dropout
    keep_prob = tf.placeholder(tf.float32)
    hidden_layer_dropout = tf.nn.dropout(hidden_layer, keep_prob)
    
    # only insert dropout at training, so we need two logits equation 
    logits_training = tf.matmul(hidden_layer_dropout, weights) + biases
    logits = tf.matmul(hidden_layer, weights) + biases # Y = XW + b, size: (10000,10), 每个行向量对应一个sample的score，下一步进行softmax后对应于各个labels的概率
    loss = tf.reduce_mean( # 未指定第二个变量reduction_indices，因此在tensor的所有元素上求平均值
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits_training))
    l2_loss = tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(weights)
    loss = loss + beta*l2_loss 
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_valid_dataset, hidden_weights), hidden_biases))
    valid_prediction = tf.nn.softmax(
        tf.matmul(valid_hidden_layer, weights) + biases)
    test_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_test_dataset, hidden_weights), hidden_biases))
    test_prediction = tf.nn.softmax(tf.matmul(test_hidden_layer, weights) + biases) 
    
num_steps = 14001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 2000 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))



########################################################################################

# Problem 4
# Try to get the best performance you can using a multi-layer model! 
# The best reported test accuracy using a deep network is 97.1%.

# One avenue you can explore is to add multiple layers.

# Another one is to use learning rate decay:

# global_step = tf.Variable(0)  # count the number of steps taken.
# learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# Learning rate decay:

beta = 0.001
batch_size = 128

graph = tf.Graph()
with graph.as_default():
    
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables.
    hidden_nodes = 1024
    hidden_weights = tf.Variable( 
            tf.truncated_normal([image_size * image_size, hidden_nodes])) # 从截断的正态分布中输出随机值。x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    hidden_biases = tf.Variable(tf.zeros([hidden_nodes])) # size: (10,)
    
    # Output layer
    weights = tf.Variable( # size: (28^2, 10)
            tf.truncated_normal([hidden_nodes, num_labels])) # 从截断的正态分布中输出随机值。x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    biases = tf.Variable(tf.zeros([num_labels])) # size: (10,)
    
    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    hidden_layer = tf.add(tf.matmul(tf_train_dataset, hidden_weights), hidden_biases)
    hidden_layer = tf.nn.relu(hidden_layer)
    
    # dropout
    keep_prob = tf.placeholder(tf.float32)
    hidden_layer_training = tf.nn.dropout(hidden_layer, keep_prob)
    
    # only insert dropout at training, so we need two logits equation 
    logits_training = tf.matmul(hidden_layer_training, weights) + biases
    logits = tf.matmul(hidden_layer, weights) + biases # Y = XW + b, size: (10000,10), 每个行向量对应一个sample的score，下一步进行softmax后对应于各个labels的概率
    loss = tf.reduce_mean( # 未指定第二个变量reduction_indices，因此在tensor的所有元素上求平均值
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits_training))
    l2_loss = tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(weights)
    loss = loss + beta*l2_loss 
    
    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.3, global_step, 1000, 0.9, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
      
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_valid_dataset, hidden_weights), hidden_biases))
    valid_prediction = tf.nn.softmax(
        tf.matmul(valid_hidden_layer, weights) + biases)
    test_hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_test_dataset, hidden_weights), hidden_biases))
    test_prediction = tf.nn.softmax(tf.matmul(test_hidden_layer, weights) + biases) 
    
num_steps = 20001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  

# Multilayer + learning rate decaying
  
beta = 0.001
batch_size = 128
layer_depth = 5


graph = tf.Graph()
with graph.as_default():
    
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables. Input layer
    hidden_nodes = 1024
    stddev = np.sqrt(2/hidden_nodes)
    weights0 = tf.Variable( 
            tf.truncated_normal([image_size * image_size, hidden_nodes], stddev = stddev))
    biases0 = tf.Variable(tf.zeros([hidden_nodes]))
    
    # Varibles. Middle layers
    hidden_nodes_pre = hidden_nodes #上一层的节点数
    hidden_weights = []
    hidden_biases = []
    for i in range(layer_depth - 2):
        if hidden_nodes_pre > 2: #上一层节点数 > 2
            hidden_nodes_cur= int(hidden_nodes_pre//2) #当前层节点数为上一层节点数的一半
        else: #上一层的节点数为1或2
            hidden_nodes_cur == hidden_nodes_pre
        stddev = np.sqrt(2/hidden_nodes_cur)
        hidden_weights.append(tf.Variable( 
            tf.truncated_normal([hidden_nodes_pre, hidden_nodes_cur], stddev = stddev)))  
        hidden_biases.append(tf.Variable(tf.zeros([hidden_nodes_cur])))
        hidden_nodes_pre = hidden_nodes_cur  #对于下一层，pre层的节点数就是当前层的节点数
    
    # Output layer
    weights = tf.Variable( # size: (28^2, 10)
            tf.truncated_normal([hidden_nodes_cur, num_labels])) # 从截断的正态分布中输出随机值。x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    biases = tf.Variable(tf.zeros([num_labels])) # size: (10,)    
    
    
    
    # Training computation
    # Input layer   
    y0 = tf.add(tf.matmul(tf_train_dataset, weights0), biases0)
    y0 = tf.nn.relu(y0)
    y0_training = tf.nn.dropout(y0, 0.5) # output for training with dropout
    
    valid_y0 = tf.matmul(tf_valid_dataset, weights0) + biases0
    valid_y0 = tf.nn.relu(valid_y0)
    
    test_y0 = tf.matmul(tf_test_dataset, weights0) + biases0
    test_y0 = tf.nn.relu(test_y0)
    
    # Middle layers
    yi_training = y0_training
    hidden_layer = y0
    
    valid_hidden_layer = valid_y0
    test_hidden_layer = test_y0
    
    for i in range(layer_depth - 2):
        yi_training = tf.nn.relu(tf.matmul(yi_training, hidden_weights[i]) + hidden_biases[i]) # the computation for the ith middle layer without dropout(depth = i + 1)
        keep_prob = 0.5*(1+ (i+1)/(layer_depth-1))
        hidden_layer_training = tf.nn.dropout(yi_training, keep_prob) # the computation for the ith middle layer with dropout(depth = i + 1)
        
        #for prediction with no dropout
        hidden_layer = tf.nn.relu(tf.matmul(hidden_layer, hidden_weights[i]) + hidden_biases[i])
        
        valid_hidden_layer = tf.nn.relu(tf.matmul(valid_hidden_layer, hidden_weights[i]) + hidden_biases[i])
        test_hidden_layer = tf.nn.relu(tf.matmul(test_hidden_layer, hidden_weights[i]) + hidden_biases[i])
    
    # Output layer
    logits_training = tf.matmul(hidden_layer_training, weights) + biases # with dropout, for training

    logits = tf.matmul(hidden_layer, weights) + biases #without dropout, for prediction
    valid_logits = tf.matmul(valid_hidden_layer, weights) + biases
    test_logits = tf.matmul(test_hidden_layer, weights) + biases
    
    # L2 Regularization
    loss = tf.reduce_mean( # 未指定第二个变量reduction_indices，因此在tensor的所有元素上求平均值
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits_training))
    l2_loss = tf.nn.l2_loss(weights0) + tf.nn.l2_loss(weights)
    for i in range(layer_depth - 2):
        l2_loss += tf.nn.l2_loss(hidden_weights[i]) 
    loss = loss + beta*l2_loss 

    

    
    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.8, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
      
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits) 
    valid_prediction = tf.nn.softmax(valid_logits)
    test_prediction = tf.nn.softmax(test_logits) 
    
num_steps = 18001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))