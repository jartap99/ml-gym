from engine import Value 
import random 
import pydot
from drawgraph import DrawGraph
import matplotlib.pyplot as plt 

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return [] 

class Neuron(Module):
    def __init__(self, nin):
        # nin: number of inputs 
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        # w*x + b 
        #print(list(zip(self.w, x)))
        s = self.b
        for wi, xi in zip(self.w, x):
            s += wi*xi
        out = s.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs 

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts):
        """ nin is num of inputs and 
        nouts is a list of num of outputs 
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

if __name__ == "__main__":
    test = False
    if test == True:
        x = [2.0, 3.0]
        n = Neuron(2)
        print(n(x))

        l = Layer(2, 3)
        print(l(x))

        x = [2.0, 3.0, -1.0]
        mlp = MLP(3, [4, 4, 1])
        print(mlp(x))

        dot = DrawGraph.draw_dot_graph(mlp(x))
        dot.render(filename="mlp.dot")
        (graph,) = pydot.graph_from_dot_file('mlp.dot')
        graph.write_png('mlp.png')
    
    mlp = MLP(3, [4, 4, 1])

    # training dataset
    xs = [  [2, 3, -1],
            [3, -1, 0.5],
            [0.5, 1, 1],
            [1, 1, -1],
        ]
    ys = [1, -1, -1, 1]

    
    if False: # these are done in trianing loop below
        ypred = [mlp(x) for x in xs]

        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        print(loss)
        print(mlp.layers[0].neurons[0].w[0].data, mlp.layers[0].neurons[0].w[0].grad)
        loss.backward()
        print(mlp.layers[0].neurons[0].w[0].data, mlp.layers[0].neurons[0].w[0].grad)

    """
    dot = DrawGraph.draw_dot_graph(loss)
    dot.render(filename="mlp_loss.dot")
    (graph,) = pydot.graph_from_dot_file('mlp_loss.dot')
    graph.write_png('mlp_loss.png')
    """
    print(mlp.parameters(), len(mlp.parameters()))

    # gradient descent training loop
    lossHist = []
    epochHist = []
    numEpochs = 100
    learningRate = 0.01
    for epoch in range(numEpochs):
        # forward pass
        ypred = [mlp(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        lossHist.append(loss.data)
        epochHist.append(epoch)
        print(epoch, loss.data)
        
        # zero gra d
        mlp.zero_grad()

        # backward pass 
        loss.backward()
        
        # update
        for p in mlp.parameters():
            p.data += p.grad*(-learningRate)
    

    plt.plot(epochHist, lossHist)
    plt.savefig("mlp_grad_descent.png")

    for i in range(4):
        print("ys, ypred: ", ys[i], ypred[i].data)
        