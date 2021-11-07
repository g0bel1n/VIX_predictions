# %%
import numpy as np
import matplotlib.pyplot as plt
from LogReg import LogitRegression
from computation_funcs import get_roc, get_auc


def simulate_data() -> tuple[np.ndarray, np.ndarray]:
    x_array = np.ones((500, 3))
    x_array[:, 1:] = np.random.rand(500, 2)
    z = 1 + 2 * x_array[:, 1] - 3 * x_array[:, 2]
    pr = 1 / (1 + np.exp(-z))

    return x_array, np.random.binomial(1, pr)


if __name__ == '__main__':
    x, y = simulate_data()

    reglog = LogitRegression(nb_expl_var=2)
    reglog_lasso = LogitRegression(nb_expl_var=2, penalization='l1', alpha=1)
    reglog_ridge = LogitRegression(nb_expl_var=2, penalization='l2', alpha=1)
    reglog_elastic_net = LogitRegression(nb_expl_var=2, penalization='Elastic Net', alpha=1, r=0.5)

    reglog.fit(x, y)
    reglog_ridge.fit(x, y)
    reglog_lasso.fit(x, y)
    reglog_elastic_net.fit(x, y)
    # print("Coefficients  :", reglog.beta)
    # print("Number of iterations  :", reglog.iter)
    tpr_values, fpr_values = get_roc(y, reglog.predict_probas(x))
    tpr_values_ridge, fpr_values_ridge = get_roc(y, reglog_ridge.predict_probas(x))
    tpr_values_lasso, fpr_values_lasso = get_roc(y, reglog_lasso.predict_probas(x))
    tpr_values_elastic_net, fpr_values_elastic_net = get_roc(y, reglog_elastic_net.predict_probas(x))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(fpr_values, tpr_values, label="No Penal")
    ax.plot(fpr_values_ridge, tpr_values_ridge, label="RIDGE")
    ax.plot(fpr_values_lasso, tpr_values_lasso, label="Lasso")
    ax.plot(fpr_values_elastic_net, tpr_values_elastic_net, label="Elastic Net")
    ax.plot(np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            label='baseline',
            linestyle='--')
    plt.title('Receiver Operating Characteristic Curve', fontsize=18)
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    plt.legend(fontsize=12)
    plt.show()
    print("AUC  :", get_auc(fpr_values, tpr_values))

# %%
