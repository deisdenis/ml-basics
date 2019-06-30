import numpy as np


def perceptron_single_step_update(
        feature_vector,
        label,
        theta,
        theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        theta - The current theta being used by the perceptron
            algorithm before this update.
        theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    need_update = label * (np.dot(theta, feature_vector) + theta_0) > 0
    if need_update:
        return theta + feature_vector * label, theta_0 + label
    return theta, theta_0


def perceptron(feature_matrix, labels, N):
    """
    Runs the  perceptron algorithm on a given set of data. Runs N
    iterations through the data set.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        N - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after N iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after N iterations through
    the feature matrix.
    """
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0

    for n in range(N):
        for i in range(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
    return theta, theta_0
