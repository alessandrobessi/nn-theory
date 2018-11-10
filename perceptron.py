from typing import Optional

import numpy as np


def train_perceptron(x: np.array,
                     y: np.array,
                     w: Optional[np.array] = None) -> np.array:
    if w is None:
        w = np.random.rand(x.shape[1]).transpose()

    convergence = False
    it = 0
    while not convergence:
        it += 1
        print("iteration", it)
        num_correct = 0
        for i in range(x.shape[0]):
            dp = np.dot(w, x[i])
            old_w = w
            if y[i] == 1 and dp <= 0:
                w = w + x[i]
            if y[i] == 0 and dp > 0:
                w = w - x[i]

            if y[i] == 1 and dp > 0:
                num_correct += 1
            if y[i] == 0 and dp <= 0:
                num_correct += 1

            print("l2norm", np.linalg.norm(w - old_w))

        if num_correct == x.shape[0]:
            print("num_correct", num_correct)
            convergence = True
    return w


if __name__ == '__main__':
    x = np.array([[1, 1, 0],
                  [1, 1, 1],
                  [1, 0, 1]])
    y = np.array([1, 1, 0]).transpose()

    w = train_perceptron(x, y)
    print("w", w)
    eps = 1e-12

    for i in range(x.shape[0]):
        cos_alpha = np.dot(w, x[i]) / (np.linalg.norm(w) * np.linalg.norm(x[i]) + eps)
        print("cos(alpha) {}, arccos {}".format(cos_alpha, np.degrees(np.arccos(cos_alpha))))
