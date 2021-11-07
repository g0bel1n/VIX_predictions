# %% Imports & setup
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys             # Pour pouvoir importer 
sys.path.append("..")  # depuis le fichier parent
from Logistic_Regressions.LogReg import LogitRegression
from Logistic_Regressions.computation_funcs import get_roc, get_auc

lags = 7

vix = pd.DataFrame(yf.Ticker("^VIX").history(period="max"))
vix = vix[['Open']]["2020-01-01":]
y = (vix - vix.shift(1)).apply(lambda x: x>=0)
dico={"Intercept" : [1 for _ in range(vix.size)]}
for i in range(1, lags+1):
    dico["day -"+str(i)] = list(vix.shift(i)['Open'])

X_set = pd.DataFrame(dico)
X_set.dropna(inplace=True)

x = X_set.to_numpy()

y = y.to_numpy()
y = y[7:]
print(x.shape)


# %% Logistics Regressions
reglog = LogitRegression(nb_expl_var=7)
reglog_lasso = LogitRegression(nb_expl_var=7, penalization='l1', alpha=1)
reglog_ridge = LogitRegression(nb_expl_var=7, penalization='l2', alpha=1)
reglog_elastic_net = LogitRegression(nb_expl_var=7, penalization='Elastic Net', alpha=1, r=0.5)

reglog.fit(x, y)
reglog_ridge.fit(x, y)
reglog_lasso.fit(x, y)
reglog_elastic_net.fit(x, y)



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


# %%
