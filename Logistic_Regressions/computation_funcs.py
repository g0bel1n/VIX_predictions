import numpy as np


def exp_fact(beta, x_array):
    s = np.array([x_array @ beta])
    return (np.transpose(np.exp(s) / (1 + np.exp(s))))[0]


def penalty(beta, alpha, r):
    assert r <= 1, "r must be inferior to 1"
    return (r * np.sum(np.abs(beta))
            + (1 - r) * (sum(np.square(beta)) / 2)) * alpha


def gradient_penalty(beta, alpha, r):
    assert r <= 1, "r must be inferior to 1"
    return (r * np.sign(beta)
            + (1 - r) * beta) * alpha


def log_likelihood(beta, x_array: np.array, y_array: np.array, penalization=None, r=None, alpha=None):
    a = np.transpose(y_array) @ (x_array @ beta)
    b = np.sum(np.log(1 + np.exp(exp_fact(beta, x_array))))

    if penalization == 'l1':
        p = penalty(beta, alpha, r=1)
    elif penalization == 'l2':
        p = penalty(beta, alpha, r=0)
    elif penalization == 'Elastic Net':
        p = penalty(beta, alpha, r)
    else:
        p = 0

    return a - b + p


def log_likelihood_hessian(beta, x, y, penalization=None, r=None, alpha=None):

    """

    wrong
    :param beta:
    :param x:
    :param y:
    :param penalization:
    :param r:
    :param alpha:
    :return:
    """
    d = np.diag(exp_fact(beta, x).ravel())
    a = np.dot(-np.transpose(x), d)
    print("Hessian shape  ", np.dot(a,x).shape)
    if penalization == "l2":
        return np.dot(a, x) + alpha * np.eye(len(x[0]))
    elif penalization == 'Elastic Net':
        return np.dot(a, x) + alpha * (1 - r) * np.eye(len(x[0]))
    else:
        return np.dot(a, x)


def log_likelihood_gradient(beta, x_array, y_vector, penalization=None, r=None, alpha=None):
    a = y_vector - exp_fact(beta, x_array)

    print(a.shape)
    print("gradient shape ", (np.transpose(x_array) @ a).shape)
    if penalization == 'l1':
        p = gradient_penalty(beta, alpha, r=1)
    elif penalization == 'l2':
        p = gradient_penalty(beta, alpha, r=0)
    elif penalization == 'Elastic Net':
        p = gradient_penalty(beta, alpha, r)
    else:
        p = 0
    return np.transpose(x_array) @ a + p


def get_roc(y, probas):
    """

    :param y: true values of y vector
    :param probas: probabilities of positive predicted output
    :return: true positive and false positive rates for Receiver operating characteristic
    """
    roc_values = []
    for thresh in np.linspace(0, 1, 100):
        preds = [1 if prob > thresh else 0 for prob in probas]

        tn, fp, fn, tp = get_confusion_matrix_coef(y, preds)
        tpr = float(tp) / float(tp + fn)
        fpr = float(fp) / float(fp + tn)
        roc_values.append([tpr, fpr])
    tpr_values, fpr_values = zip(*roc_values)
    return tpr_values, fpr_values


def get_confusion_matrix_coef(y, y_pred):
    """

    :param y: true value of vector y
    :param y_pred: predicted values
    :return: the 4 coefficients of positive/negative confusion matrix
    """
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            if y_pred[i] == 0:
                tn += 1
            else:
                tp += 1
        else:
            if y_pred[i] == 0:
                fn += 1
            else:
                fp += 1

    return tn, fp, fn, tp


def get_auc(x, y):
    """
    :param x: x axis values
    :param y: y axis values
    :return: area under the (x,y) curve using the triangle method
    """
    assert len(x) == len(y), "x and y must have the same size"
    auc = 0
    for i in range(len(x) - 1):
        auc += (x[i] - x[i + 1]) * (y[i + 1] + y[i]) / 2
    return auc
