from typing import Tuple

import numpy as np


def pad(I: np.array, type: str = "edge") -> np.array:
    """
    Mở rộng các biên của ảnh. Được sử dụng trong phép convd.
    :param I: Ảnh màu shape=(x,y,3)
    :param type: cách độn thêm: edge -> Mở rộng giống điểm gần nhất, khác -> Mở rộng bằng không
    :return: I' một ảnh đã qua mở rộng biên
    """
    if type == "edge":
        return np.pad(I, ((1, 1), (1, 1), (0, 0)), 'edge')
    return np.pad(I, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)


def conv2d(I: np.array, kernel, kernel_shape: Tuple = None) -> np.array:
    """
    Phép nhân chập 2D.
    :param kernel_shape: hình dạng áp dụng hàm nhân chập
    :param I: Ảnh ở dạng ma trận 2D
    :param kernel: hàm nhân hoặc ma trận nhân
    :return: Ảnh sau phép nhân chập ở dạng ma trận 2D
    """
    if isinstance(kernel, (np.ndarray, np.generic)):
        kernel_shape = kernel.shape
    rs = []
    for y in range(I.shape[0] - kernel_shape[0] + 1):
        a = []
        for x in range(I.shape[1] - kernel_shape[1] + 1):
            g = I[y:y + kernel_shape[0], x:x + kernel_shape[1]]
            if isinstance(kernel, (np.ndarray, np.generic)):
                dot = g * kernel
                dot = np.sum(dot)
            else:
                dot = kernel(g)
            a.append(dot)
        rs.append(a)
    return np.array(rs)


KERNEL_SHAPE = (3, 3)


def DDF(i: np.array) -> np.array:
    def DDF_estimator(window: np.array):
        points = window.reshape(-1, window.shape[-1])
        l2 = window / np.linalg.norm(window, axis=-1)[..., None]
        flatten = l2.reshape(-1, l2.shape[-1])
        cos = np.matmul(flatten, flatten.T)
        cos[cos > 1] = 1
        angles = np.arccos(cos)  # shape n x n
        distances = np.array([[np.linalg.norm(fi - fj) for fi in points] for fj in points])  # shape n x n
        dffs = [np.matmul(a, d.T) for a, d in zip(angles, distances)]
        min_i = np.argmin(dffs)
        return points[min_i]

    return conv2d(i, kernel=DDF_estimator, kernel_shape=KERNEL_SHAPE).astype(int)


def GVDF(i: np.array) -> np.array:
    def BVDF(window: np.array):
        points = window.reshape(-1, window.shape[-1])
        l2 = window / np.linalg.norm(window, axis=-1)[..., None]
        flatten = l2.reshape(-1, l2.shape[-1])
        cos = np.matmul(flatten, flatten.T)
        cos[cos > 1] = 1
        angles = np.arccos(cos)
        anglesum = np.sum(angles, axis=-1)
        pair = [(point, a) for point, a in zip(points, anglesum)]
        pair.sort(key=lambda p: p[-1])
        pair = pair[:int(points.shape[0] / 2)]
        points = np.array([p[0] for p in pair])
        grayscale = np.dot(points, [0.299, 0.587, 0.114])
        pair = [(point, a) for point, a in zip(points, grayscale)]
        pair.sort(key=lambda p: p[-1])
        return pair[int(points.shape[0] / 2)][0]

    return conv2d(i, kernel=BVDF, kernel_shape=KERNEL_SHAPE).astype(int)


def ANMF(i: np.array) -> np.array:
    def ANMF_estimator(window: np.array):
        points = window.reshape(-1, window.shape[-1])
        mid_i = int(points.shape[0] / 2)
        y = points[mid_i]
        ys = np.delete(points, mid_i, axis=0)
        n = ys.shape[0]
        k = 1
        M = 3

        def hl(n, k, ys, yl, M):
            return n ** (-k / M) * np.sum(np.linalg.norm(ys - yl))

        def K(z):
            return np.exp(-1 * np.linalg.norm(z))

        ws = [hl(n, k, ys, yl, M) ** (-M) * K((y - yl) / hl(n, k, ys, yl, M)) for yl in ys]
        total = np.sum(ws)
        samples = [yl * tl / total for yl, tl in zip(ys, ws)]
        return np.sum(samples, axis=0)

    return conv2d(i, kernel=ANMF_estimator, kernel_shape=KERNEL_SHAPE).astype(int)


def VMF(i: np.array) -> np.array:
    def VMF_estimator(window: np.array):
        points = window.reshape(-1, window.shape[-1])
        return np.median(points, axis=0)

    return conv2d(i, kernel=VMF_estimator, kernel_shape=KERNEL_SHAPE).astype(int)


def AVMF(i: np.array) -> np.array:
    def AVMF_estimator(window: np.array):
        points = window.reshape(-1, window.shape[-1])
        return np.mean(points, axis=0)

    return conv2d(i, kernel=AVMF_estimator, kernel_shape=KERNEL_SHAPE).astype(int)


if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    import cv2

    i = cv2.imread("lenna.png")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(i)
    plt.show()


    # %% md
    ## thêm nhiễu
    ### 1. Gauss
    # %%
    def add_gauss_noise(i: np.array, s: float) -> np.array:
        gauss = np.random.normal(loc=0, scale=s, size=i.shape) + i
        gauss = (gauss - np.amin(gauss.reshape(-1, 3), axis=0)) / (
                np.amax(gauss.reshape(-1, 3), axis=0) - np.amin(gauss.reshape(-1, 3), axis=0)) * 255
        gauss = gauss.astype(int)
        return gauss


    gauss = add_gauss_noise(i, 30)
    plt.axis('off')
    plt.imshow(gauss)
    plt.show()


    # %% md
    ### 2. Impulsive
    # %%
    def add_impulsive_noise(i: np.array, ratio: float) -> np.array:
        import copy
        # random position
        i_p = copy.deepcopy(i).reshape(-1, 3)
        total_pos = i_p.shape[0]
        pos = total_pos * np.random.rand(int(total_pos * ratio))
        pos = pos.astype(int)
        for p in pos:
            is_salt = np.random.ranf() > 0.5
            if is_salt:
                i_p[p] = [0, 0, 0]
            else:
                i_p[p] = [255, 255, 255]
        return i_p.reshape(i.shape)


    impulsive = add_impulsive_noise(i, 0.02)
    plt.axis('off')
    plt.imshow(impulsive)
    plt.show()

    # %%
    from filter import *


    def filter_plot(i: np.array, filter) -> np.array:
        filtered = filter(i)
        plt.axis('off')
        plt.imshow(filtered)
        plt.show()
        return filtered


    filters = [
        ANMF,
        AVMF,
        DDF,
        GVDF
    ]
    for filter in filters:
        print(filter.__name__)
        a = filter_plot(gauss, filter)
        b = filter_plot(impulsive, filter)


