import numpy as np
import numpy.linalg as nlg


class DictionaryLearning:
    def __init__(self, D, x, lambda_=None, verbose=True):
        """
        solving the following the problem:

        arg min {x} 1/2*||y-D*x||_F^2 + lambda_*||x||_0

        D: Dictionary
        x: Sparse Representation
        lambda_: regulation parameter
        """
        self.D = D
        self.x = x
        self.lambda_ = lambda_
        self.verbose = verbose

    def _proximal_D(self, dictionary):
        # normalize each column(atom)
        for i in range(dictionary.shape[1]):
            dictionary[:, i] = dictionary[:, i] / np.linalg.norm(dictionary[:, i])
        return dictionary

    def _greedy(self, x, non_zero):
        for c in range(x.shape[1]):
            y = np.argsort(np.abs(x[:, c]))[::-1]
            x_new = np.zeros(x.shape[0])
            for i in y[0:non_zero]:
                x_new[i] = x[i, c]
            x[:, c] = x_new
        return x

    def train(self, y, x=None, learning_rate=2e-4, num_iter=1000, non_zero=3):
        if x is None:
            x = np.random.randn(self.D.shape[1], y.shape[1])

        for step in range(1, num_iter + 1):
            # """calculate gradient"""
            # partial_D = -(y - self.D @ x) @ self.x.T
            # partial_x = -self.D.T @ (y - self.D @ self.x)
            #
            # """gradient step"""
            # self.D -= learning_rate*partial_D
            # self.x -= learning_rate*partial_x
            #
            # """proximal step"""
            # self.D = self._proximal_D(self.D)
            # self.x = self._greedy(self.x, non_zero=non_zero)
            """D upadte"""
            partial_D = -(y - self.D @ x) @ self.x.T
            self.D -= learning_rate * partial_D
            self.D = self._proximal_D(self.D)

            """x update"""
            partial_x = -self.D.T @ (y - self.D @ self.x)
            self.x -= learning_rate * partial_x
            self.x = self._greedy(self.x, non_zero=non_zero)

            if self.verbose is True and (step % 100) == 0:
                print("step ", "%5i" % step,
                      " ||y-D*x|| =", "%4.4f" % nlg.norm(y - self.D @ self.x))

    def get_D_and_x(self):
        return self.D, self.x

