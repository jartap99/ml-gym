#!/usr/bin/env python3

import math
import numpy as np
import torch 
import pydot
from drawgraph import DrawGraph

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self.grad =  0.0 # gradient
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None  # by defautl doesnt do anyting (eg leaf node)

    def __repr__(self):
        return f"Value(data={self.data}, {self._op=})"#, {self._prev=}) "

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other)    :
        assert isinstance(other, (int, float)), "Only supporting ints and floats for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other - 1) )  * out.grad
        out._backward = _backward 
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other 

    def __truediv__(self, other): 
        return self * other**-1 

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)
        
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward 
        return out

    def log(self):
        x = self.data
        out = Value(math.log(x), (self,), 'log')

        def _backward():
            self.grad += (1/x) * out.grad
        out._backward = _backward 
        return out

    def backward(self):
        self.grad = 1.0

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        for node in reversed(topo):
            node._backward()

def test1():
    a = Value(3.0, label='a')
    b = Value(4.1, label='b')
    c = Value(-2, label='c')
    d = a*b; d.label='d'
    e = d + c; e.label='e'
    f = Value(1.1, label='f')
    L = e*f; L.label='L'
    L.grad = 1.0 # manual step for last node

    dot = DrawGraph.draw_dot_graph(L)
    dot.render(filename="dot_graph.dot")
    (graph,) = pydot.graph_from_dot_file('dot_graph.dot')
    graph.write_png('dot_graph.png')

def torch_equivalent():
    import torch 
    x1 = torch.Tensor([2.0]).double()
    x2 = torch.Tensor([0.0]).double()
    # weights: w1, w2
    w1 = torch.Tensor([-3.0]).double()
    w2 = torch.Tensor([1.0]).double()
    x1.requires_grad = True
    x2.requires_grad = True
    w1.requires_grad = True
    w2.requires_grad = True
    # bias of he neuron
    b = torch.Tensor([6.8813735870195432]).double()
    b.requires_grad = True

    print("torch x1: ", x1)
    print("torch x2: ", x2)
    print("torch w1: ", w1)
    print("torch w2: ", w2)

    n = x1*w1 + x2*w2 + b
    print("torch n: ", n)
    o = torch.tanh(n)
    print("torch o: ", o.data.item())
    o.backward()

    print("torch x1.grad: ", x1.grad.item())
    print("torch w1.grad: ", w1.grad.item())
    print("torch x2.grad: ", x2.grad.item())
    print("torch w2.grad: ", w2.grad.item())

def test3():
    # without referencing our code/video __too__ much, make this cell work
    # you'll have to implement (in some cases re-implemented) a number of functions
    # of the Value object, similar to what we've seen in the video.
    # instead of the squared error loss this implements the negative log likelihood
    # loss, which is very often used in classification.

    # this is the softmax function
    # https://en.wikipedia.org/wiki/Softmax_function
    def softmax(logits):
        counts = [logit.exp() for logit in logits]
        denominator = sum(counts)
        out = [c / denominator for c in counts]
        return out

    # this is the negative log likelihood loss function, pervasive in classification
    logits = [Value(0.0), Value(3.0), Value(-2.0), Value(1.0)]
    print("logits: ", logits)
    probs = softmax(logits)
    print("probs: ", probs)
    loss = -probs[3].log() # dim 3 acts as the label for this input example
    print("loss: ", loss)
    loss.backward()
    print("*"*10)
    print("logits: ", logits)
    print("probs: ", probs)
    print("loss: ", loss)
    print("*"*10)

    print(loss.data)

    ans = [0.041772570515350445, 0.8390245074625319, 0.005653302662216329, -0.8864503806400986]
    for dim in range(4):
        ok = 'OK' if abs(logits[dim].grad - ans[dim]) < 1e-5 else 'WRONG!'
        print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {logits[dim].grad}")

if __name__ == "__main__":
    # MLP
    # inputs: x1, x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights: w1, w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias of he neuron
    b = Value(6.8813735870195432, label='b')
    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label ='n'
    #o = n.tanh()
    e = (2*n).exp(); e.label='e'
    o = (e-1)/(e+1); o.label='o'

    dot = DrawGraph.draw_dot_graph(o)
    dot.render(filename="dot_graph_mlp.dot")
    (graph,) = pydot.graph_from_dot_file('dot_graph_mlp.dot')
    graph.write_png('dot_graph_mlp.png')

    o.backward()
    
    dot = DrawGraph.draw_dot_graph(o)
    dot.render(filename="dot_graph_mlp_backward.dot")
    (graph,) = pydot.graph_from_dot_file('dot_graph_mlp_backward.dot')
    graph.write_png('dot_graph_mlp_backward.png')

    # torch equivalent
    torch_equivalent()


    test3()
   