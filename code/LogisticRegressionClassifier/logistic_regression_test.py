import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.linear_model import LogisticRegression
from LogisticRegressionClassifier import LogisticRegressionClassifier
import os


def load_data(filename:str):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            splited_line = [float(i) for i in line.strip().split('\t')]
            data, label = [1.0] + splited_line[: -1], splited_line[-1]
            dataset.append(data)
            labels.append(label)
    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels


def snapshot(w, dataset, labels, pic_name):
    if not os.path.exists("./test/snapshot"):
        os.mkdir("./test/snapshot")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pts = {}
    for data, label in zip(dataset.tolist(), labels.tolist()):
        pts.setdefault(label, [data]).append(data)
    for label, data in pts.items():
        data = np.array(data)
        plt.scatter(data[:, 1], data[:, 2], label=label, alpha=0.5)

    # 分割线绘制
    def get_y(x, w):
        w0, w1, w2 = w
        return (-w0 - w1*x)/w2
    x = [-4.0, 3.0]
    y = [get_y(i, w) for i in x]
    plt.plot(x, y, linewidth=2, color='#FB4A42')
    pic_name = './test/snapshot/{}'.format(pic_name)
    fig.savefig(pic_name)
    plt.close(fig)


if __name__ == '__main__':
    clf = LogisticRegressionClassifier()
    dataset, labels = load_data("./test/test_set.txt")
    # print(dataset, labels)
    # w, ws = clf.gradient_ascent(dataset, labels, max_iter=50000)
    w, ws = clf.stoch_gradient_ascent(dataset, labels, max_iter=500)
    print(w, ws, ws.shape)
    m, n, l = ws.shape
    # print(m, n)
    # 绘制分割线
    for i, w in enumerate(ws):
        if i % (m // 10) == 0:
            snapshot(w, dataset, labels, '{}.png'.format(clf.stoch_gradient_ascent.__name__ + "_" + str(i)))
            print('{}.png saved'.format(i))
    fig = plt.figure()
    for i in range(n):
        label = 'w{}'.format(i)
        ax = fig.add_subplot(n, 1, i+1)
        ax.plot(ws[:, i], label=label)
        ax.legend()
    fig.savefig('w_traj.png')
