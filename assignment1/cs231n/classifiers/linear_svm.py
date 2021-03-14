from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

                # derivative with respect to a given weight w_ij is 0 if margin
                # is 0, else it is x_ij if i is not the correct class or -x_ij
                # if it is the right class
                # Note: in our equations W = (C, D), here W = (D, C) so the
                # columns represent the template/weight vector for each class
                dW[:, j] += X[i]
                dW[:, y[i]] += -X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # divide gradients by number of examples seen
    dW /= num_train

    # add regularization gradient, we could've added this term for each
    # iteration in the training loop which would've been corrected by
    # dW /= num_train, however it is smarter to just add it once at the end
    # rather than adding it N times and then dividing by N
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # modified above code

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    scores = np.matmul(X, W)  # (N, D) * (D, C) -> (N, C)

    # for each score in the matrix, subtract the score of the correct class for
    # that row
    # reshape is needed for correct numpy broadcasting (ie repeat the score of
    # the correct class in each column).
    #
    # notice that this means one entry per row in margin will be 0, since the
    # one entry per row will be subtracting the score of the correct class
    # with itself
    margins = scores - scores[np.arange(N), y].reshape(-1, 1)  # (N, C)

    # add delta
    margins += 1

    # max function (make negative entires 0)
    margins[margins < 0] = 0

    # since we added delta to each element, including the entry where we did
    # score of correct class - score of correct class, we need to fix this by
    # making the margin of the correct class to 0 (in the formula, we say
    # j != y_i when we take the loss L_i, so we are staying true to that here)
    margins[np.arange(N), y] = 0

    # loss is the sum of all the margins, divided by the number of examples
    loss = np.sum(margins) / N

    # regularization loss
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # number of times X[i] is in the gradient for class j in a given row
    # 0 if margin <= 0, 1 if margin > 0 and j != y_i, and -n for j = y_i where
    # n is the  number of times the margin was > 0 for the row
    margins[margins > 0] = 1
    margins[np.arange(N), y] = -np.sum(margins, axis=1)

    # repeat X[i] for the number of classes, and multiply by margins to
    # get an array of gradients of shape (N, C, D)
    grad = X.reshape(N, 1, D)
    grad = np.repeat(grad, C, axis=1)

    # element wise multiply vectors by margin scalars
    grad = grad * margins.reshape(-1, C, 1)

    # sum and average gradients across all examples
    dW = np.sum(grad, axis=0).T / N  # (N, C, D) --sum--> (C, D) --transpose--> (D, C)

    # add regularization gradient
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
