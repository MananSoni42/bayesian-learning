import numpy as np
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
import pymc3 as pm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score

x, y = load_breast_cancer(return_X_y=True)
x  = x[:,:4]
sample_size = 3000
val_size = 0.25

ind = np.arange(x.shape[0])
np.random.shuffle(ind)

x_train = x[ind,:][int(val_size*x.shape[0]):,:]
y_train = y[ind][int(val_size*y.shape[0]):]

x_test = x[ind,:][:int(val_size*x.shape[0]),:]
y_test = y[ind][:int(val_size*y.shape[0])]

# Set up the model
x_shared = theano.shared(x_train)
y_shared = theano.shared(y_train)
with pm.Model() as lrh_model:
    # Model is Y ~ N(a0 + a1*x1+a2*x2+a3*x1*x2, sig^2)
    # priors
    a0 = pm.Normal('a0', mu=0, sigma=1000)
    a = pm.Normal('a', mu=0, sigma=1000,shape=4)

    # likelihood
    #phi = tt.nnet.softmax(a0+tt.dot(x1_shared,a))
    phi =  1/(1+np.exp(-(a0+tt.dot(x_shared,a))))
    y_pred = pm.Bernoulli('y', p=phi, observed=y_shared)

    # sample from posterior distribution
    trace = pm.sample(sample_size,tune=1500,cores=1,init='adapt_diag')

# Check the diagnostics and ensure convergence
"""
print('Gelman-Rubin')
print(pm.diagnostics.gelman_rubin(trace))

print('Actual Sample size')
print(sample_size)
print('Effective Sample Size')
"""
print(pm.diagnostics.effective_n(trace))

#pm.traceplot(trace)
#plt.show()

# set up scikikt model
sci_model = LogisticRegression()
sci_model.fit(x_train,y_train)
sci_y_pred_train = sci_model.predict(x_train)
print(sci_y_pred_train[:5])

y_pred_train = np.mean(pm.sample_posterior_predictive(trace,model=lrh_model)['y'],axis=0)
y_pred_train[np.where(y_pred_train<0.5)] = 0
y_pred_train[np.where(y_pred_train>=0.5)] = 1

print('train - scikit')
print(accuracy_score(y_train,sci_y_pred_train))
print(classification_report(y_train,sci_y_pred_train))

print('train - basyesian')
print(accuracy_score(y_train,y_pred_train))
print(classification_report(y_train,y_pred_train))

sci_y_pred_test = sci_model.predict(x_test)
x_shared.set_value(x_test)
y_shared.set_value(y_test)
y_pred_test = np.mean(pm.sample_posterior_predictive(trace,model=lrh_model)['y'],axis=0)
y_pred_test[np.where(y_pred_test<0.5)] = 0
y_pred_test[np.where(y_pred_test>=0.5)] = 1

print('test - scikit')
print(accuracy_score(y_test,sci_y_pred_test))
print(classification_report(y_test,sci_y_pred_test))

print('test - basyesian')
print(accuracy_score(y_test,y_pred_test))
print(classification_report(y_test,y_pred_test))
