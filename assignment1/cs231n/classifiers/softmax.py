from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

#     N = X.shape[0]
#     D = X.shape[1]
#     C = W.shape[1]

#     # cross entropy loss and gradient
#     for i in range(N):
#         f = X[i].dot(W)  # (D) * (D, C) -> (C)
#         # adjusted_f = f - np.max(f)
#         unnormalized_probs = np.exp(f)  # (C)
#         prob = unnormalized_probs[y[i]] / np.sum(unnormalized_probs)
#         loss += -np.log(prob)

#         for k in range(C):
#             grad = -1 / (prob * np.log(prob))  # (1)
#             grad *= unnormalized_probs[k]  # (1)
#             grad *= X[i]  # (D)
#             if k == y[i]:
#                 dW[:, k] += 1 * grad
#             else:
#                 dW[:, k] += (-1 / unnormalized_probs[k] ** 2) * grad

#     # normalize loss and gradient by number of trainin examples
#     loss /= N
#     dW /= N

#     # regularization loss and gradient
#     for j in range(D):
#         for k in range(C):
#             loss += reg * W[j][k] ** 2
#             dW += 2 * reg * W[j][k]


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    for i in range(N):
        scores = np.matmul(X[i], W)  # (N, D) * (D, C) -> (N, C)
        adjusted_scores = scores + -np.max(scores)  # for numerical stability
        unnormalized_probs = np.exp(adjusted_scores)

        loss += -np.log(unnormalized_probs[y[i]] / np.sum(unnormalized_probs))

    loss /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
