import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  def softmax(x):
      """x: a vector"""
      m = np.max(x)
      den = np.sum(np.exp(x-m))
      nom = np.exp(x-m)
      return nom/den



  batch_size = X.shape[0]
  num_classies = W.shape[1]
  s = X @ W
  # scores = softmax(s)

  for i in range(batch_size):
      score = softmax(s[i])  # score vector of ith sample
      s_y = score[y[i]]      # correct_label_score, y[i] is the right idx of classification
      loss += - np.log(s_y)  # loss for ith sample
      for j in range(num_classies):
          if j == y[i]:
              ds = s_y - 1
              dW[:, j] += ds*X[i]
          else:
              ds = score[j]
              dW[:, j] += ds*X[i]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= batch_size
  loss += reg * np.sum(W * W)
  dW /= batch_size
  dW += 2 * reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  def softmax(x):

      orig_shape = x.shape
      if len(x.shape) > 1:
          # Matrix
          ### YOUR CODE HERE
          m = np.max(x, axis=1).reshape(-1, 1)
          denominator = np.sum(np.exp(x - m), axis=1, keepdims=True)
          numerator = np.exp(x - m)
          x = numerator / denominator
          # raise NotImplementedError
          ### END YOUR CODE

      else:
          # Vector
          ### YOUR CODE HERE
          m = np.max(x)
          denominator = np.sum(np.exp(x - m), keepdims=True)
          numerator = np.exp(x - m)
          x = numerator / denominator
          # raise NotImplementedError
          ### END YOUR CODE
      assert x.shape == orig_shape
      return x

  def onehot(y, num_class):
      """y: int denote # class"""
      batch_size = y.shape[0]
      code = np.zeros((batch_size,num_class))
      code[range(batch_size),y] = 1
      return code

  batch_size = X.shape[0]
  num_classies = W.shape[1]
  s = X @ W
  scores = softmax(s)
  correct_label_scores = scores[range(batch_size), y]
  loss = sum(-np.log(correct_label_scores))/batch_size
  loss += reg * np.sum(W * W)
  dW = X.T@(scores - onehot(y,num_classies))/batch_size
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

