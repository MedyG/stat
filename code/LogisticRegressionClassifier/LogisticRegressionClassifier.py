import numpy as np
import random

class LogisticRegressionClassifier(object):
    def __init__(self):
        print("new logistic regression classifier")

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def gradient_ascent(self, dataset, labels, max_iter=10000):
        dataset = np.matrix(dataset)
        vlabels = np.matrix(labels).reshape(-1, 1)
        m, n = dataset.shape
        w = np.ones((n, 1))
        alpha = 0.001
        ws = []
        for i in range(max_iter):
            error = vlabels - self.sigmoid(dataset*w)
            w += alpha * dataset.T * error
            ws.append(w.reshape(-1, 1).tolist())
        self.w = w
        return w, np.array(ws)

    def stoch_gradient_ascent(self, dataset, labels, max_iter=150):
        """
        随机梯度上升
        :param dataset: 训练数据
        :param labels: 标记数据
        :param max_iter: 最大迭代次数
        :return:
        """
        dataset = np.matrix(dataset)
        vlabels = np.matrix(labels).reshape(-1, 1)
        m, n = dataset.shape
        w = np.matrix(np.ones((n, 1)))
        ws = []
        for i in range(max_iter):
            indices = list(range(m))
            random.shuffle(indices)
            for j, indice in enumerate(indices):
                data, label = dataset[indice], labels[indice]
                error = label - self.sigmoid((data*w).tolist()[0][0])
                alpha = 4/(i + j + 1) + 0.01
                d = alpha * data.T * error
                w += alpha * data.T * error
                ws.append(w.tolist())
        self.w = w
        return w, np.array(ws)
