import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt
import pymc3 as pm

x1 = np.random.rand(100)
x2 = np.random.rand(100)
y = 6 - 5*x1 + 4*x2 + np.random.randn(100)
sample_size = 3000
num_samples = 3

# Set up the model
with pm.Model() as lrh_model:
    # Model is Y ~ N(a0 + a1*x1+a2*x2+a3*x1*x2, sig^2)
    # priors
    a0 = pm.Normal('a0', -100, sigma=1000)
    a1 = pm.Normal('a1', mu=100, sigma=1000)
    a2 = pm.Normal('a2', mu=-100, sigma=1000)
    a3 = pm.Normal('a3', mu=10, sigma=1000)
    sig = pm.InverseGamma('sig', alpha=4.2, beta=10)

    # likelihood
    y_pred = pm.Normal('y', mu=a0+a1*x1+a2*x2+a3*x1*x2,sigma=sig,observed=y)

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

"""
a0_95 = pm.stats.hpd(trace['a0'])
a1_95 = pm.stats.hpd(trace['a1'])
print(a0_95,a1_95)

a0_95 = np.linspace(a0_95[0],a0_95[1],3)
a1_95 = np.linspace(a1_95[0],a1_95[1],3)
#y_pred = np.mean(trace['a0']) + np.mean(trace['a1'])*x

for m in a1_95:
    for c in a0_95:
        y_pred = m*x + c
        plt.plot(x,y_pred,'r',alpha=0.4)

plt.scatter(x,y,color='k')
plt.plot(x,np.mean(trace['a0'])+np.mean(trace['a1'])*x,'b')
plt.show()
"""
