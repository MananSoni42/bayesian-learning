import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

x = pd.read_csv('heart_data', sep=' ',
                    names=['age', 'sex', 'chest_pain', 'rest_bp',
                    'chol', 'fast_bs_120', 'rest_ecg', 'max_rate',
                    'ex_angina', 'oldpeak', 'grad_peak', 'num_vessels', 'thal','heart_disease'],
                    index_col=False)

cols = np.array(x.columns)
data_type = np.array(['r', 'b', 'n','r', 'r', 'b', 'n', 'r', 'b', 'r', 'o', 'r', 'n'])

x_train, x_test = train_test_split(x,test_size=0.3)
y_train = np.array(x_train['heart_disease'])
y_test = np.array(x_test['heart_disease'])
x_train = np.array(x_train.drop(columns=['heart_disease']))
x_test = np.array(x_test.drop(columns=['heart_disease']))

"""
print(x.head())
for s in set(data_type):
    print(s)
    print(np.where(data_type == s)[0].shape)

print(data.iloc[:,np.where(data_type == 'r')[0]].describe())
"""

# 1. Basic model: logistic regression on all variables
X_data = theano.shared(x_train)
Y_data = theano.shared(y_train)
with pm.Model() as basic_model:
    # priors
    c1 = pm.Normal('c1', mu=0, sigma=25)
    m1 = pm.Normal('m1', mu=np.zeros((13,)), sigma=25*np.ones((13,)), shape=13)
    # likelihood
    #print(len(m.tag.test_value))
    phi1 = pm.Deterministic('phi1',1/(1 + np.exp(-(theano.tensor.dot(X_data,m1)+c1))))
    #print(np.array(phi.tag.test_value))
    y_pred1 = pm.Bernoulli('y_pred1', phi1, observed=1/(1+np.exp(-Y_data)))
    # sample from posterior distribution
    trace1 = pm.sample(3000, init='adapt_diag', tune=1000, cores=1)

"""
# sample from posterior distribution
X_data.set_value(x_test)
Y_data.set_value(y_test)
print(pm.sample_posterior_predictive(trace1_train,model=basic_model))
"""
"""
# Check the diagnostics and ensure convergence
print('Gelman-Rubin')
pprint(pm.diagnostics.gelman_rubin(trace1_train))

print('Actual Sample size')
pprint(3000)
print('Effective Sample Size')
pprint(pm.diagnostics.effective_n(trace1_train))
"""
for i in range(trace1['m1'].shape[1]):
    pm.traceplot(trace1['m1'][:,i])
    plt.title(cols[i])
    plt.show()
