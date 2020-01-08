import numpy as np
import numpy.linalg as nlg


class SparseCoding2D1A:
    def __init__(self, D1, D2, A, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, verbose=True):
        """
        solving the following problem:

        arg min {x1, x2} 1/2||y-D1*x1-D2*x2||_F^2 + lambda_1*||x1||_0 + lambda_2*||x2||_0
                         + lambda_3*1/2||A*D1*x1-z1||_F^2 + lambda_4*1/2||A*D1*x1-z1||_F^2
                         + lambda_5*||z1||_0 + lambda_6*||z2||_0

        D1: dictionary 1
        D2: dictionary 2
        A : analysis dictionary
        x1: sparse representation of y1 by dictionary 1
        x2: sparse representation of y2 by dictionary 2
        z1: auxiliary variable 1
        z2: auxiliary variable 2
        lambda_{1, 2, 3, 4, 5, 6}: regulation parameters
        """
        self.D1 = D1
        self.D2 = D2
        self.A = A
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.lambda_5 = lambda_5
        self.lambda_6 = lambda_6
        self.verbose = verbose

    def _proximal_D(self, dictionary):
        # normalize each column(atom)
        for i in range(dictionary.shape[1]):
            dictionary[:, i] = dictionary[:, i] / np.linalg.norm(dictionary[:, i])
        return dictionary

    def _greedy(self, x, n_nonzero):
        for c in range(x.shape[1]):
            y = np.argsort(np.abs(x[:, c]))[::-1]
            x_new = np.zeros(x.shape[0])
            for i in y[0:n_nonzero]:
                x_new[i] = x[i, c]
            x[:, c] = x_new
        return x

    def _soft_threshold(self, x, lambda_threshold, coef=0.5):
        for i in range(x.shape[1]):
            x[:, i] = np.multiply(np.sign(x[:, i]), np.maximum(0, np.abs(x[:, i]) - coef * lambda_threshold[i]))
        return x

    def _greedy_z(self, z, n_nonzero=None):
        n = z.shape[0]
        m = z.shape[1]
        if n_nonzero is None:
            n_nonzero = int(n*m/10)
        z = np.reshape(z, n * m)
        y = np.argsort(np.abs(z))[::-1]
        z_new = np.zeros_like(z)
        for i in y[0:n_nonzero]:
            z_new[i] = z[i]
        z_new = np.reshape(z_new, (n, m))
        return z_new

    def _soft_threshold_z(self, z, lambda_threshold):
        return np.multiply(np.sign(z), np.maximum(0, np.abs(z) - lambda_threshold))

    def get_x(self, y, x1=None, x2=None, proximal_use=None, learning_rate=2e-4, num_iter=1000, soft_coef=0.3,
              l_thres_x1=None, l_thres_x2=None, n_nonzero=6):
        """get x1 and x2 via sparse coding"""

        """initial x1 and x2"""
        np.random.seed(1)
        if x1 is None:
            x1 = np.random.randn(self.D1.shape[1], y.shape[1])
        if x2 is None:
            x2 = np.random.randn(self.D2.shape[1], y.shape[1])

        """intiial z1 and z2"""
        z1 = np.random.randn(self.A.shape[0], y.shape[1])
        z2 = np.random.randn(self.A.shape[0], y.shape[1])

        for step in range(1, num_iter + 1):
            """step 1 update x1 and x2"""
            """calculate x1 and x2 gradient"""
            partial_x1 = -self.D1.T @ (y - self.D1 @ x1 - self.D2 @ x2) \
                + self.lambda_3 * (self.A @ self.D1).T @ (self.A @ self.D1 @ x1 - z1)
            partial_x2 = -self.D2.T @ (y - self.D1 @ x1 - self.D2 @ x2) \
                + self.lambda_4 * (self.A @ self.D2).T @ (self.A @ self.D2 @ x2 - z2)

            """gradient decent step"""
            x1 -= learning_rate * partial_x1
            x2 -= learning_rate * partial_x2

            """proximal mapping step"""
            if proximal_use is "soft":
                x1 = self._soft_threshold(x1, lambda_threshold=np.mean(abs(x1), axis=0), coef=soft_coef)
                x2 = self._soft_threshold(x2, lambda_threshold=np.mean(abs(x2), axis=0), coef=soft_coef)
            elif proximal_use is "greedy":
                x1 = self._greedy(x1, n_nonzero=n_nonzero)
                x2 = self._greedy(x2, n_nonzero=n_nonzero)

            """step 2 update z1 and z2"""
            """calculate z1 and z2 gradient"""
            partial_z1 = -self.lambda_3 * (self.A @ self.D1 @ x1 - z1)
            partial_z2 = -self.lambda_4 * (self.A @ self.D2 @ x2 - z2)
            """gradient decent step"""
            z1 -= learning_rate * partial_z1
            z2 -= learning_rate * partial_z2
            """proximal mapping step"""
            if proximal_use is "soft":
                z1 = self._soft_threshold_z(z1, lambda_threshold=np.mean(np.abs(z1)))
                z2 = self._soft_threshold_z(z2, lambda_threshold=np.mean(np.abs(z2)))
            elif proximal_use is "greedy":
                z1 = self._greedy_z(z1)
                z2 = self._greedy_z(z2)
            else:
                print("error: didn't choose proximal mapping method")
                break

            """print information (optional)"""
            if (step % 100) == 0 and self.verbose is True:
                print("step ", "%5i" % step,
                      " ||y-D1*x1-D2*x2|| =", "%4.4f" % nlg.norm(y - self.D1 @ x1 - self.D2 @ x2), "\n",
                      " ||x1||_0 =", "%d" % np.sum(x1 != 0), " ||x2||_0 =", "%d" % np.sum(x2 != 0), "\n",
                      " ||x1||_1 =", "%4.4f" % nlg.norm(x1, ord=1), " ||x2||_1 =", "%4.4f" % nlg.norm(x2, ord=1), "\n",
                      " ||A*D1*x1-z1||_F =", "%4.4f" % nlg.norm(self.A @ self.D1 @ x1 - z1),  "\n",
                      " ||A*D2*x2-z2||_F =", "%4.4f" % nlg.norm(self.A @ self.D2 @ x2 - z2), "\n",
                      " ||z1||_0 =", "%d" % np.sum(z1 != 0), " ||z2||_0 =", "%d" % np.sum(z2 != 0), "\n",
                      " ||z1||_1 =", "%4.4f" % nlg.norm(z1, ord=1), " ||z2||_1 =", "%4.4f" % nlg.norm(z2, ord=1), "\n")

        return x1, x2


class SparseCoding2D:
    def __init__(self, D1, D2, lambda_1=None, lambda_2=None, verbose=True):
        """
        solving the following problem:

        arg min {x1, x2} 1/2||y-D1*x1-D2*x2||_F^2 + lambda_1*||x1||_0 + lambda_2*||x2||_0

        D1: dictionary 1
        D2: dictionary 2
        x1: sparse representation of y1 by dictionary 1
        x2: sparse representation of y2 by dictionary 2
        lambda_{1, 2}: regulation parameters
        """
        self.D1 = D1
        self.D2 = D2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.verbose = verbose

    def _proximal_D(self, dictionary):
        # normalize each column(atom)
        for i in range(dictionary.shape[1]):
            dictionary[:, i] = dictionary[:, i] / np.linalg.norm(dictionary[:, i])
        return dictionary

    def _greedy(self, x, n_nonzero):
        for c in range(x.shape[1]):
            y = np.argsort(np.abs(x[:, c]))[::-1]
            x_new = np.zeros(x.shape[0])
            for i in y[0:n_nonzero]:
                x_new[i] = x[i, c]
            x[:, c] = x_new
        return x

    def _soft_threshold(self, x, lambda_threshold, coef=0.5):
        for i in range(x.shape[1]):
            x[:, i] = np.multiply(np.sign(x[:, i]), np.maximum(0, np.abs(x[:, i]) - coef * lambda_threshold[i]))
        return x

    def get_x(self, y, x1=None, x2=None, proximal_use=None, learning_rate=2e-4, num_iter=1000,
              l_thres_x1=None, l_thres_x2=None, soft_coef=0.5,n_nonzero=6):
        # initial x
        np.random.seed(1)
        if x1 is None:
            x1 = np.random.randn(self.D1.shape[1], y.shape[1])
        if x2 is None:
            x2 = np.random.randn(self.D2.shape[1], y.shape[1])

        for step in range(1, num_iter + 1):
            """calculate x1 and x2 gradient"""
            partial_x1 = - self.D1.T @ (y - self.D1 @ x1 - self.D2 @ x2)
            partial_x2 = - self.D2.T @ (y - self.D1 @ x1 - self.D2 @ x2)

            """gradient decent step"""
            x1 -= learning_rate * partial_x1
            x2 -= learning_rate * partial_x2

            """priximal mapping step"""
            if proximal_use is "greedy":
                x1 = self._greedy(x1, n_nonzero=n_nonzero)
                x2 = self._greedy(x2, n_nonzero=n_nonzero)
            elif proximal_use is "soft":
                x1 = self._soft_threshold(x1, lambda_threshold=np.mean(abs(x1), axis=0), coef=soft_coef)
                x2 = self._soft_threshold(x2, lambda_threshold=np.mean(abs(x2), axis=0), coef=soft_coef)
            else:
                print("error: didn't choose proximal mapping method")
                break

            """print information (optional)"""
            if (step % 100) == 0 and self.verbose is True:
                print("step ", "%5i" % step,
                      " ||y-D1*x1-D2*x2|| =", "%4.4f" % nlg.norm(y - self.D1 @ x1 - self.D2 @ x2), "\n",
                      " ||x1||_0 =", "%d" % np.sum(x1 != 0), " ||x2||_0 =", "%d" % np.sum(x2 != 0), "\n",
                      " ||x1||_1 =", "%4.4f" % nlg.norm(x1, ord=1), " ||x2||_1 =", "%4.4f" % nlg.norm(x2, ord=1), "\n")

        return x1, x2


class SparseCoding1D:
    def __inti__(self, D, lambda_, verbose=True):
        """
        solving the following the problem:

        arg min {x} 1/2*||y-D*x||_F^2 + lambda_*||x||_0

        D: Dictionary
        x: Sparse Representation
        lambda: regulation parameter
        """
        self.D = D
        self.lambda_ = lambda_
        self.verbose = verbose

    def get_x(self, y, x=None):

        return x
