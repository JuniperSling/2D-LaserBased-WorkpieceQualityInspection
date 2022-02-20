import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  # numpy的卷积函数


def standardization(y):
    mu = np.mean(y, axis=0)
    sigma = np.std(y, axis=0)
    return (y - mu) / sigma


def kick_bad(x, y):  # 剔除坏点
    left = y.mean() - 2 * y.std()
    delete = []
    for i in range(0, len(y)):
        if y[i] < left:
            delete.append(i)
    x = np.delete(x, delete)
    y = np.delete(y, delete)
    return x, y


if __name__ == '__main__':
    path = "/Users/albertmilagro/Desktop/LS/ld/data"
    os.chdir(path)
    for dirName, dirs, files in os.walk("."):
        for f in files:
            if f.split('.')[1] != 'txt':
                continue
            x = np.loadtxt(f, usecols=0, encoding='unicode_escape').ravel()
            y = np.loadtxt(f, usecols=1, encoding='unicode_escape').ravel()
            # y = moving_average(interval=y, window_size=2)
            print(f)
            x, y = kick_bad(x, y)
            y = standardization(y)

            plt.plot(x, y)
            plt.scatter(x, y, marker='x', c='r')
            plt.savefig(path + "/" + f.split(".")[0] + ".png")
            plt.clf()
