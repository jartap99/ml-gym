import random 
import numpy as np 
import matplotlib.pyplot as plt 
from nn import Neuron, Layer, MLP 
from engine import Value 

np.random.seed(1337)
random.seed(1337)

from sklearn.datasets import make_moons, make_blobs 

# Loss function 
def loss(model, X, y, batch_size = None):
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0][:batch_size])
        Xb, yb = X[ri], y[ri]
    
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model to get scores 
    scores = list(map(model, inputs))

    # sum max-margin loss 
    losses = [(1 + -yi*scorei).tanh() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0/len(losses))

    # L2 regularization 
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters() ))
    total_loss = data_loss + reg_loss 

    # accuracy 
    accuracy = [(yi>0) == (scorei.data>0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy)/len(accuracy) 

# training loop / optimization 
def train(model, X, y, numEpochs=4):
    e = [i for i in range(numEpochs)]
    l = []
    for k in range(numEpochs):
        # forward 
        total_loss, acc = loss(model, X, y)
        l.append(total_loss.data)
        # backward 
        model.zero_grad()
        total_loss.backward() 

        # update (sgd)
        learning_rate = 1.0 - 0.9*k/100 
        for p in model.parameters():
            p.data -= learning_rate * p.grad 

        if (k%1)==0:
            print(f"Step {k} Loss:{total_loss.data} Acciracy:{acc*100}%")
    
    plt.plot(e, l)
    plt.savefig("classified_1.png")
    plt.close()

if __name__ == "__main__":
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y*2 - 1 # make y be -1 or 1

    plt.scatter(X[:,0], X[:, 1], c=y, s=20, cmap='jet')
    plt.savefig("classifier_0.png")
    plt.close()

    model = MLP(2, [16, 16, 1])
    print(model)

    
    train(model, X, y, numEpochs=150)
    
    # visualize decision boundary

    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig("classifier_2.png")
    plt.close()