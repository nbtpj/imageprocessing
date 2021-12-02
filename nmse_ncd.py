import numpy as np
import cv2


def nmse(origin, estimator):
    numerator = np.sum(np.linalg.norm(origin - estimator, axis=2) ** 2)
    denominator = np.sum(np.linalg.norm(origin, axis=2) ** 2)
    return numerator / denominator


def ncd(origin, estimator):
    origin_lab = cv2.cvtColor(origin, cv2.COLOR_BGR2Lab)
    estimator_lab = cv2.cvtColor(estimator, cv2.COLOR_BGR2Lab)
    numerator = np.sum(np.linalg.norm(origin_lab - estimator_lab, axis=2) ** 2)
    denominator = np.sum(np.linalg.norm(origin_lab, axis=2) ** 2)
    return numerator / denominator


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
