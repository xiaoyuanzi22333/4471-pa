from audioop import mul
import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    scores_exsum = np.sum(np.exp(scores))
    scores_exp = np.exp(scores)
    correct_class_score = scores[y[i]]
    loss += - (correct_class_score - np.log(scores_exsum))
    for j in range(num_classes):
      dW[:,j] += scores_exp[j]/scores_exsum*X[i]
    dW[:,y[i]] += -X[i]

  loss /= num_train
  loss += reg*np.sum(W * W)
  dW /= num_train
  dW += 2*reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  mult = X.dot(W)
  exp = np.exp(mult)
  exp_log = np.log(np.sum(exp,axis = 1))
  sub_sum = np.sum(mult[range(X.shape[0]),y])
  loss = (np.sum(exp_log) - sub_sum)/X.shape[0]
  loss += reg*np.sum(W*W)

  suum = np.sum(exp,axis = 1)
  cof = np.array(exp/np.matrix(suum).T)
  cof[range(X.shape[0]),y] -= 1
  dW = np.dot(X.T, cof)

  dW /= X.shape[0]
  dW += 2*reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

