import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

colmap = ['r','g','b']

def plot_results(X,mu,sig,num_clusters,num_gaussians,colmap=colmap):
    for i in range(num_clusters):
        #plt.scatter(X[i], 0.1+0.4*np.random.random(X[i].shape), c=colmap(i*100//num_clusters))
        plt.scatter(X[i], np.zeros(X[i].shape), c=colmap[i])

    for i in range(num_gaussians):
        xvals = np.linspace(0,16,101)
        plt.plot(xvals,0.2+stats.norm.pdf(xvals,mu[i],sig[i]), c=colmap[i])

    plt.gca().set_xlim(0,16)
    plt.gca().set_ylim(-0.1,3)
    plt.show()

# make the clusters
X = []
X.append(np.random.normal(3,3,size=(50,)))
X.append(np.random.normal(7,3,size=(50,)))
X.append(np.random.normal(11,3,size=(50,)))
print('Actual values')
print((3,3),(7,3),(11,3))
X = np.array(X)
num_clusters = X.shape[0]
epochs = 50

# Define a Gaussian mixture model
num_gaussians = 3
weight = (1/num_gaussians)*np.ones(num_gaussians)
mu = 1+14*np.random.rand(num_gaussians)
sig = 3*np.random.random(num_gaussians)

print(*[(round(mu[i],2),round(sig[i],2)) for i in range(num_gaussians)],weight)
plot_results(X,mu,sig,num_clusters,num_gaussians)

for i in range(epochs):
    # E - step
    prob = np.zeros((num_gaussians,X.flatten().shape[0]))

    for i in range(num_gaussians):
        prob[i,:] = weight[i]*stats.norm.pdf(X.flatten(),mu[i],sig[i])
    prob = prob/np.sum(prob,axis=0)

    # M - step
    mu = np.dot(prob,X.flatten()) / np.sum(prob,axis=1)
    for i in range(num_gaussians):
        sig[i] = np.dot(prob[i,:],np.square(X.flatten()-mu[i])) / np.sum(prob[i,:])
    sig = np.sqrt(sig)
    weight = np.sum(prob,axis=1) / X.flatten().shape[0]

print(*[(round(mu[i],2),round(sig[i],2)) for i in range(num_gaussians)],weight)
plot_results(X,mu,sig,num_clusters,num_gaussians)
