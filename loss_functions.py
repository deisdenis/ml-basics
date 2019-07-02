import numpy as np


def hinge_loss(z):
    if z >= 1:
        return 0
    return 1 - z


def squared_error_loss(z):
    return z**2/2


def empirical_risk(feature_vector, labels, theta, theta_0=0, loss_function=hinge_loss):
    n = feature_vector.shape[0]
    risk_sum = 0
    for i in range(n):
        loss_parameter = labels[i] - (np.dot(theta, feature_vector[i, :]) + theta_0)
        risk_sum += loss_function(loss_parameter)
    return risk_sum/n
