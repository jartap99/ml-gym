import graphviz
import matplotlib.pyplot as plt
import pydot 


class DrawGraph:
    def __init__(self):
        pass

    @classmethod
    def trace(cls, rootNode):
        nodes, edges = set(), set()
        
        def build(v):
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
        
        build(rootNode)
        return nodes, edges

    @classmethod 
    def draw_dot_graph(cls, rootNode):
        dot = graphviz.Digraph(format='svg', graph_attr={'rankdir':'LR'}) # LR = Left to Right

        nodes, edges = cls.trace(rootNode)
        for n in nodes:
            uid = str(id(n))
            dot.node(name=uid, label="{%s | data %.4f | grad %.4f }"%(n.label, n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=uid+n._op, label=n._op)
                dot.edge(uid+n._op, uid)
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2))+n2._op)
        
        return dot
