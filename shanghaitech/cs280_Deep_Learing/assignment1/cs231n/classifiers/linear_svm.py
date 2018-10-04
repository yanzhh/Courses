import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW_list = []
  for i in range(num_train):
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    np_loss = np.maximum(0, scores - correct_class_score + 1)
    np_loss[y[i]] = 0
    np_loss[np_loss > 0] = 1

    for j in range(num_classes):
      if j == y[i]:
        dW[:,j] = -sum(np_loss)*X[i]
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] = np_loss[j] * X[i]
    dW_list.append(dW)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW =  sum(dW_list)/num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  batch_size = X.shape[0]
  np_score_mat = X@W  # (N,C)
  np_score_label = np_score_mat[[range(batch_size)],y].reshape(-1,1) # (N,1)
  np_score_mat = np_score_mat - np_score_label + 1 # (N,C)
  np_score_mat[np_score_mat <= 0] = 0
  np_score_mat[[range(batch_size)],y] = 0

  loss = np.sum(np_score_mat)/batch_size
  loss += reg * np.sum(W * W)

  # np_loss_mat = np.zeros(shape = np_score_mat.shape)
  # np_loss_mat[np_score_mat>0] = 1

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  np_loss_mat = np_score_mat
  np_loss_mat[np_loss_mat > 0] = 1
  np_loss_mat[[range(batch_size)],y] = -np.sum(np_loss_mat, axis=1)
  dW = X.T @ np_loss_mat
  dW /= batch_size
  dW += 2*reg*W



  # np_loss_mat = np.zeros(shape = np_score_mat.shape)
  # np_loss_mat[np_score_mat>0] = 1
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
