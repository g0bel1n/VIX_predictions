from Logistic_Regressions.computation_funcs import *


class LogitRegression:
    def __init__(self, nb_expl_var, penalization=None, alpha=None, r=None):
        self.beta = np.array([np.random.randint(-5, 5, nb_expl_var + 1)]).T
        self.iter = 0
        self.penalization = penalization
        self.alpha = alpha
        self.r = r

    def update_coef_nrm(self, x, y):
        print("beta shape", self.beta.shape)
        b = np.linalg.inv(log_likelihood_hessian(self.beta,
                                                                     x,
                                                                     y,
                                                                     self.penalization,
                                                                     self.r,
                                                                     self.alpha)) @ log_likelihood_gradient(self.beta,
                                                                                                            x,
                                                                                                            y,
                                                                                                            self.penalization,
                                                                                                            self.r,
                                                                                                            self.alpha)

        print("b shape  ", b.shape)
        self.beta = self.beta - np.linalg.inv(log_likelihood_hessian(self.beta,
                                                                     x,
                                                                     y,
                                                                     self.penalization,
                                                                     self.r,
                                                                     self.alpha)) @ log_likelihood_gradient(self.beta,
                                                                                                            x,
                                                                                                            y,
                                                                                                            self.penalization,
                                                                                                            self.r,
                                                                                                            self.alpha)
        self.iter += 1
        return None

    def get_log_likelihood(self, x, y):
        return log_likelihood(self.beta,
                              x,
                              y,
                              self.penalization,
                              self.r,
                              self.alpha)

    def predict_probas(self, x):
        return exp_fact(self.beta, x)

    def predict(self, x):
        return x @ self.beta

    def fit(self, x, y, epsilon=1e-5):
        prev_beta = self.beta
        self.update_coef_nrm(x, y)
        self.iter += 1
        while np.sqrt(np.transpose(self.beta - prev_beta) @ (self.beta - prev_beta))[0,0] > epsilon:
            prev_beta = self.beta
            self.update_coef_nrm(x, y)
