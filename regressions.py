import numpy as np

def ridge_regression_step_update(feature_vector, label, theta, theta_0, eta, l):
    theta = (1 - eta * l) * theta + eta * (label - np.dot(theta, feature_vector))
    theta_0 = (1 - eta * l) * theta_0 + eta (label - theta_0)
    return theta, theta_0




def ridge_regression(feature_matrix, labels, N=1):
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0

    for n in range(N):
        for i in range(feature_matrix.shape[0]):
            theta, theta_0 = ridge_regression_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
    return theta, theta_0
