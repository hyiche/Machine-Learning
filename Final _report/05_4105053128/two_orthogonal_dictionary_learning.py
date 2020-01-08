import numpy as np
import numpy.linalg as nlg


class TwoOrthogonalDictionaryLearning:
    def __init__(self, D1, D2, x1, x2, n_nonzero_coefs=None, lmbda=0, lmbda1=0, lmbda2=0, verbose=True):
        """
                Args:
                        Solve following problem:
                        f = ||y_1-D_1*x_1||_F^2 + ||y_2-D_2*x_2||_F^2 + lambda*||D_1'*D_2||_F^2
                             + lmbda1*||D_1'*D_1||_F^2 + lmbda2*||D_2'*D_2||_F^2

                        g = Γ_x(x_1) + Γ_x(x_2) + Γ_D(D_1) + Γ_D(D_2)

                        D1: Dictionary for y1.
                        D2: Dictionary for y2.
                        x1: Sparse representation of y1 for D1.
                        x2: Sparse representation of y2 for D2.
                        n_nonzero_coefs: Number of non-zero entries in the solution. [10% of n_features]
                        lmbda: The coefficient controls orthogonality of D1 and D2.[0]
                        lmbda1: The coefficient controls orthogonality of D1 and D1.[0]
                        lmbda2: The coefficient controls orthogonality of D2 and D2.[0]
                        verbose: If True, it will show the information of learning rate and lost.
                """
        self.D1 = D1
        self.D2 = D2
        self.x1 = x1
        self.x2 = x2
        if n_nonzero_coefs is None:
            self.n_nonzero_coefs = int(self.D1.shape[1] / 10)  # m is nonzero coefficient for each column
        else:
            self.n_nonzero_coefs = n_nonzero_coefs
        self.lmbda = lmbda
        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2
        self.verbose = verbose
        self.build_parameter()

    def build_parameter(self):
        self.D1_old = None
        self.x1_old = None
        self.D2_old = None
        self.x2_old = None
        self.D1_new = None
        self.x1_new = None
        self.D2_new = None
        self.x2_new = None

    def _proximal_D(self, dictionary):
        # normalize each column(atom)
        for i in range(dictionary.shape[1]):
            dictionary[:, i] = dictionary[:, i] / np.linalg.norm(dictionary[:, i])
        return dictionary

    def _l0_proximal_x(self, x):
        for c in range(x.shape[1]):
            y = np.argsort(np.abs(x[:, c]))[::-1]
            x_new = np.zeros(x.shape[0])
            for i in y[0:self.n_nonzero_coefs]:
                x_new[i] = x[i, c]
            x[:, c] = x_new
        return x

    def _l1_proximal_x(self, x):
        x = np.multiply(np.sign(x), np.maximum(0, np.abs(x) - np.mean(np.abs(x))))
        return x

    def _get_lambda_threshold(self, x, non_zero):
        return np.sort(a=x, axis=0)[non_zero - 1, :]

    def _hard_threshold(self, x, lambda_threshold):
        for i in range(x.shape[1]):
            for j in range(x.shape[0]):
                if abs(x[j, i]) < lambda_threshold[i]:
                    x[j, i] = 0
        return x

    def _soft_threshold(self, x, lambda_threshold, coef=0.5):
        for i in range(x.shape[1]):
            x[:, i] = np.multiply(np.sign(x[:, i]), np.maximum(0, np.abs(x[:, i]) - coef * lambda_threshold[i]))
        return x

    def lipschitz(self, y1, y2, learning_rate=0.0002):
        # learning rate calculated by Lipschitz constance
        # L_t = ||f(v_{t})-f(v_{t-1})|| / ||v_{t} - v_{t-1}||
        f = np.trace((y1 - self.D1 @ self.x1) @ (y1 - self.D1 @ self.x1).T) \
            + np.trace((y2 - self.D2 @ self.x2) @ (y2 - self.D2 @ self.x2).T) \
            + self.lmbda * np.trace((self.D1.T @ self.D2) @ (self.D1.T @ self.D2).T) \
            + self.lmbda1 * np.trace((self.D1.T @ self.D1) @ (self.D1.T @ self.D1).T) \
            + self.lmbda2 * np.trace((self.D2.T @ self.D2) @ (self.D2.T @ self.D2).T)

        f_old = np.trace((y1 - self.D1_old @ self.x1_old) @ (y1 - self.D1_old @ self.x1_old).T) \
            + np.trace((y2 - self.D2_old @ self.x2_old) @ (y2 - self.D2_old @ self.x2_old).T) \
            + self.lmbda * np.trace((self.D1_old.T @ self.D2_old) @ (self.D1_old.T @ self.D2_old).T) \
            + self.lmbda1 * np.trace((self.D1_old.T @ self.D1_old) @ (self.D1_old.T @ self.D1_old).T) \
            + self.lmbda2 * np.trace((self.D2_old.T @ self.D2_old) @ (self.D2_old.T @ self.D2_old).T)

        norm_f = nlg.norm(f - f_old)

        # lambda_t = 1 / (2*L_t)
        # lambda_D1
        L_D1 = norm_f / nlg.norm(self.D1 - self.D1_old)
        lambda_D1 = 1 / (2*L_D1)
        if (lambda_D1 < learning_rate) or (lambda_D1 > 0):
            lambda_D1 = learning_rate

        # lambda_x1
        L_x1 = norm_f / nlg.norm(self.x1 - self.x1_old)
        lambda_x1 = 1 / (2*L_x1)
        if (lambda_x1 < learning_rate) or (lambda_x1 > 0):
            lambda_x1 = learning_rate

        # lambda_D2
        L_D2 = norm_f / nlg.norm(self.D2 - self.D2_old)
        lambda_D2 = 1 / (2*L_D2)
        if (lambda_D2 < learning_rate) or (lambda_D2 > 0):
            lambda_D2 = learning_rate

        # lambda_x2
        L_x2 = norm_f / nlg.norm(self.x2 - self.x2_old)
        lambda_x2 = 1 / (2*L_x2)
        if (lambda_x2 < learning_rate) or (lambda_x2 > 0):
            lambda_x2 = learning_rate

        return lambda_D1, lambda_x1, lambda_D2, lambda_x2

    def train(self, y1, y2, num_iter=1000, learning_rate=0.0002, soft_coef=None,
              proximal_use=None, use_lipschitz=False, l_thres_x1=None, l_thres_x2=None):
        # update D1, D2, x1, and x2
        for step in range(1, num_iter + 1):
            # calculate gradient
            partial_D1 = -2 * (y1 - self.D1 @ self.x1) @ self.x1.T \
                + self.lmbda * 2 * self.D2 @ self.D2.T @ self.D1 \
                + self.lmbda1 * 4 * self.D1 @ self.D1.T @ self.D1

            partial_x1 = -2 * self.D1.T @ (y1 - self.D1 @ self.x1)

            partial_D2 = -2 * (y2 - self.D2 @ self.x2) @ self.x2.T \
                + self.lmbda * 2 * self.D1 @ self.D1.T @ self.D2 \
                + self.lmbda2 * 4 * self.D2 @ self.D2.T @ self.D2

            partial_x2 = -2 * self.D2.T @ (y2 - self.D2 @ self.x2)

            # learning rate
            if (step == 1) or (use_lipschitz is False):
                # consistent learning rate
                lambda_D1 = learning_rate
                lambda_x1 = learning_rate
                lambda_D2 = learning_rate
                lambda_x2 = learning_rate
            else:
                lambda_D1, lambda_x1, lambda_D2, lambda_x2 = self.lipschitz(y1=y1, y2=y2, learning_rate=learning_rate)

            """gradient step"""
            D1_new = self.D1 - lambda_D1 * partial_D1
            x1_new = self.x1 - lambda_x1 * partial_x1
            D2_new = self.D2 - lambda_D2 * partial_D2
            x2_new = self.x2 - lambda_x2 * partial_x2

            """proximal step"""
            # proximal operator for dictionary
            D1_new = self._proximal_D(D1_new)
            D2_new = self._proximal_D(D2_new)

            # proximal operator for x
            if (proximal_use == 'l0') or (proximal_use == 'L0') or (proximal_use == 'greedy'):
                x1_new = self._l0_proximal_x(x1_new)
                x2_new = self._l0_proximal_x(x2_new)
            elif (proximal_use == 'l1') or (proximal_use == 'L1'):
                x1_new = self._l1_proximal_x(x1_new)
                x2_new = self._l1_proximal_x(x2_new)
            elif proximal_use == 'soft':
                x1_new = self._soft_threshold(x1_new, lambda_threshold=l_thres_x1, coef=soft_coef)
                x2_new = self._soft_threshold(x2_new, lambda_threshold=l_thres_x2, coef=soft_coef)
            elif proximal_use == 'hard':
                x1_new = self._hard_threshold(x1_new, lambda_threshold=l_thres_x1)
                x2_new = self._hard_threshold(x2_new, lambda_threshold=l_thres_x2)
            else:
                print("error: didn't choose l0 proximal or l1 proximal>")
                break

            """save D1, x1, D2, and x2"""
            # save V_{t-1}
            self.D1_old = self.D1
            self.x1_old = self.x1
            self.D2_old = self.D2
            self.x2_old = self.x2

            # save V_t
            self.D1 = D1_new
            self.x1 = x1_new
            self.D2 = D2_new
            self.x2 = x2_new

            """print information (optional)"""
            if (step % 100) == 0 and self.verbose is True:
                print("step ", "%5i" % step,
                      " lambda_D1 =", "%1.4f" % lambda_D1, " lambda_x1 =", "%1.4f" % lambda_x1,
                      " lambda_D2 =", "%1.4f" % lambda_D2, " lambda_x2 =", "%1.4f" % lambda_x2, "\n", " "*10,
                      " ||D1^T*D2|| =", "%1.4f" % nlg.norm(np.transpose(self.D1)@self.D2),
                      " ||D1^T*D1|| =", "%1.4f" % nlg.norm(np.transpose(self.D1)@self.D1),
                      " ||D2^T*D2|| =", "%1.4f" % nlg.norm(np.transpose(self.D2)@self.D2), "\n", " "*10,
                      " ||y1-D1*x1|| =", "%1.4f" % nlg.norm(y1 - self.D1 @ self.x1),
                      " ||y2-D2*x2|| =", "%1.4f" % nlg.norm(y2 - self.D2 @ self.x2), "\n")

    def train_fix_D(self, y1, y2, num_iter=1000, learning_rate=0.0002, soft_coef=0.3,
              proximal_use=None, use_lipschitz=False, l_thres_x1=None, l_thres_x2=None):
        # update x1 and x2
        for step in range(1, num_iter + 1):
            partial_x1 = -2 * self.D1.T @ (y1 - self.D1 @ self.x1)

            partial_x2 = -2 * self.D2.T @ (y2 - self.D2 @ self.x2)

            # learning rate
            if (step == 1) or (use_lipschitz is False):
                # consistent learning rate
                lambda_x1 = learning_rate
                lambda_x2 = learning_rate
            else:
                lambda_D1, lambda_x1, lambda_D2, lambda_x2 = self.lipschitz(y1=y1, y2=y2, learning_rate=learning_rate)

            """gradient step"""
            x1_new = self.x1 - lambda_x1 * partial_x1
            x2_new = self.x2 - lambda_x2 * partial_x2

            """proximal step"""
            # proximal operator for x
            if (proximal_use == 'l0') or (proximal_use == 'L0'):
                x1_new = self._l0_proximal_x(x1_new)
                x2_new = self._l0_proximal_x(x2_new)
            elif (proximal_use == 'l1') or (proximal_use == 'L1'):
                x1_new = self._l1_proximal_x(x1_new)
                x2_new = self._l1_proximal_x(x2_new)
            elif proximal_use == 'soft':
                x1_new = self._soft_threshold(x1_new, lambda_threshold=l_thres_x1, coef=soft_coef)
                x2_new = self._soft_threshold(x2_new, lambda_threshold=l_thres_x2, coef=soft_coef)
            elif proximal_use == 'soft_mean':
                x1_new = self._soft_threshold(x1_new, lambda_threshold=np.mean(abs(x1_new), axis=0), coef=soft_coef)
                x2_new = self._soft_threshold(x2_new, lambda_threshold=np.mean(abs(x2_new), axis=0), coef=soft_coef)
            elif proximal_use == 'hard':
                x1_new = self._hard_threshold(x1_new, lambda_threshold=l_thres_x1)
                x2_new = self._hard_threshold(x2_new, lambda_threshold=l_thres_x2)
            else:
                print("error: didn't choose l0 proximal or l1 proximal>")
                break

            """save D1, x1, D2, and x2"""
            # save V_{t-1}
            self.x1_old = self.x1
            self.x2_old = self.x2

            # save V_t
            self.x1 = x1_new
            self.x2 = x2_new

            """print information (optional)"""
            if (step % 100) == 0 and self.verbose is True:
                print("step ", "%5i" % step,
                      " lambda_x1 =", "%1.4f" % lambda_x1, " lambda_x2 =", "%1.4f" % lambda_x2, "\n", " "*10,
                      " ||D1^T*D2|| =", "%1.4f" % nlg.norm(np.transpose(self.D1)@self.D2),
                      " ||D1^T*D1|| =", "%1.4f" % nlg.norm(np.transpose(self.D1)@self.D1),
                      " ||D2^T*D2|| =", "%1.4f" % nlg.norm(np.transpose(self.D2)@self.D2), "\n", " "*10,
                      " ||y1-D1*x1|| =", "%1.4f" % nlg.norm(y1 - self.D1 @ self.x1),
                      " ||y2-D2*x2|| =", "%1.4f" % nlg.norm(y2 - self.D2 @ self.x2), "\n")

    def get_dictionary(self):
        return self.D1, self.D2

    def get_x(self):
        return self.x1, self.x2


class SparseCoding2D:
    """ solve || y - D1*x1 - D2*x2 ||_F^2 """
    def __init__(self, D1, D2, n_nonzero_x=10):
        self.D1 = D1
        self.D2 = D2
        self.n_nonzero_x = n_nonzero_x

    def _l0_proximal_x(self, x, n_nonzero):
        for c in range(x.shape[1]):
            y = np.argsort(np.abs(x[:, c]))[::-1]
            x_new = np.zeros(x.shape[0])
            for i in y[0: n_nonzero]:
                x_new[i] = x[i, c]
            x[:, c] = x_new
        return x

    def _l1_proximal_x(self, x):
        x = np.multiply(np.sign(x), np.maximum(0, np.abs(x) - np.mean(np.abs(x))))
        return x

    def _hard_threshold(self, x, lambda_threshold):
        for i in range(x.shape[1]):
            for j in range(x.shape[0]):
                if abs(x[j, i]) < lambda_threshold[i]:
                    x[j, i] = 0
        return x

    def _soft_threshold(self, x, lambda_threshold, coef=0.5):
        for i in range(x.shape[1]):
            x[:, i] = np.multiply(np.sign(x[:, i]), np.maximum(0, np.abs(x[:, i]) - coef * lambda_threshold[i]))
        return x

    def get_x(self, y, x1=None, x2=None, proximal_use=None, lambda_x=2e-4, num_iter=1000,
              l_thres_x1=None, l_thres_x2=None):

        # initial x
        np.random.seed(1)
        if x1 is None:
            x1 = np.random.randn(self.D1.shape[1], y.shape[1])
        if x2 is None:
            x2 = np.random.randn(self.D2.shape[1], y.shape[1])

        for step in range(int(num_iter)):
            # calculate gradient
            partial_x1 = -2 * self.D1.T @ (y - self.D1 @ x1 - self.D2 @ x2)
            partial_x2 = -2 * self.D2.T @ (y - self.D1 @ x1 - self.D2 @ x2)

            # gradient step
            x1 = x1 - lambda_x * partial_x1
            x2 = x2 - lambda_x * partial_x2

            # proximal step
            if (proximal_use == 'l0') or (proximal_use == 'L0') or (proximal_use == 'greedy'):
                x1 = self._l0_proximal_x(x1, n_nonzero=self.n_nonzero_x)
                x2 = self._l0_proximal_x(x2, n_nonzero=self.n_nonzero_x)
            elif (proximal_use == 'l1') or (proximal_use == 'L1'):
                x1 = self._l1_proximal_x(x1)
                x2 = self._l1_proximal_x(x2)
            elif proximal_use == 'soft':
                x1 = self._soft_threshold(x1, lambda_threshold=l_thres_x1)
                x2 = self._soft_threshold(x2, lambda_threshold=l_thres_x2)
            elif proximal_use == 'mean_soft':
                x1 = self._soft_threshold(x1, lambda_threshold=np.mean(abs(x1), axis=0))
                x2 = self._soft_threshold(x2, lambda_threshold=np.mean(abs(x2), axis=0))
            elif proximal_use == 'hard':
                x1 = self._hard_threshold(x1, lambda_threshold=l_thres_x1)
                x2 = self._hard_threshold(x2, lambda_threshold=l_thres_x2)
            else:
                print("error: didn't choose l0 proximal or l1 proximal>")
                break

            if step % 100 == 0:
                print(step, "/", num_iter)

        return x1, x2
