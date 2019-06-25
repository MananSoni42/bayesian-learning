import numpy as np
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
import pymc3 as pm
from sklearn.linear_model import LinearRegression as lr
x1 = np.linspace(0,4,100)
y = 6 - 5*x1 + np.random.randn(100)/3
sample_size = 3000
num_samples = 3

# set up scikit lr model
sci_model = lr()
sci_model.fit(x1.reshape(-1,1),y)
y_sci_pred = sci_model.predict(x1.reshape(-1,1))

# Set up the model
x1_shared = theano.shared(x1)
y_shared = theano.shared(y)
with pm.Model() as lrh_model:
    # Model is Y ~ N(a0 + a1*x1+a2*x2+a3*x1*x2, sig^2)
    # priors
    a0 = pm.Normal('a0', 0, sigma=1000)
    a1 = pm.Normal('a1', 0, sigma=1000)
    sig = pm.InverseGamma('sig', alpha=4.2, beta=10)

    # likelihood
    y_pred = pm.Normal('y', mu=a0+a1*x1_shared, sigma=sig, observed=y_shared)

    # sample from posterior distribution
    trace = pm.sample(sample_size,tune=1000,cores=num_samples)

# Check the diagnostics and ensure convergence
print('Gelman-Rubin')
print(pm.diagnostics.gelman_rubin(trace))

print('Actual Sample size')
print(sample_size)
print('Effective Sample Size')
print(pm.diagnostics.effective_n(trace))

pm.traceplot(trace)
plt.show()

y_pred_train = np.mean(pm.sample_posterior_predictive(trace,model=lrh_model)['y'],axis=0)
plt.figure(1)
plt.title('train')
plt.scatter(x1,y,color='r')
plt.scatter(x1,y_pred_train,color='b')
plt.scatter(x1,y_sci_pred,color='g')

print(np.sum(np.square(y-y_pred_train)))
print(np.sum(np.square(y-y_sci_pred)))
x1_1 = np.linspace(6,7,100)
y_1 = 6 - 5*x1_1 + np.random.randn(100)/3
x1_shared.set_value(x1_1)
y_shared.set_value(y_1)
y_pred_test = np.mean(pm.sample_posterior_predictive(trace,model=lrh_model)['y'],axis=0)
y_sci_test = sci_model.predict(x1_1.reshape(-1,1))
plt.figure(2)
plt.title('test')
plt.scatter(x1_1,y,color='r')
plt.scatter(x1_1,y_pred_test,color='b')
plt.scatter(x1_1,y_sci_test,color='g')
plt.show()
print(np.sum(np.square(y_1-y_pred_test)))
print(np.sum(np.square(y_1-y_sci_test)))
