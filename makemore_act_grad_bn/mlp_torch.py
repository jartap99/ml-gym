import torch 
import matplotlib.pyplot as plt 
import random 

from mlp import create_dataset

g = torch.Generator().manual_seed(2147483647)

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / (fan_in**0.5)
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self,x ):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters trained with backpropagation
        self.gamma = torch.ones(dim) #bngain
        self.beta = torch.zeros(dim) #bnbias
        # buffers trained with a running momentum update 
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            xmean = x.mean( 0, keepdim=True ) # batch mean 
            xvar = x.var( 0, keepdim=True ) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps)
        self.out = (self.gamma * xhat) + self.beta

        # update buffers 
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x.mean()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x.var()
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
    
if __name__ == "__main__":
    block_size =3
    emb_dim = 10
    hidden_neurons = 100
    vocab_size = 27
    
    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = create_dataset('names.txt', block_size = block_size)
    
    # this is our model 
    # BN can be placed after non linearity as well and people have gotten very similar result with that
    C = torch.randn((vocab_size, emb_dim), generator=g) 
    layers = [
            Linear(emb_dim*block_size, hidden_neurons), BatchNorm1D(hidden_neurons), Tanh(),
            Linear(hidden_neurons, hidden_neurons), BatchNorm1D(hidden_neurons), Tanh(),
            Linear(hidden_neurons, hidden_neurons), BatchNorm1D(hidden_neurons), Tanh(),
            Linear(hidden_neurons, hidden_neurons), BatchNorm1D(hidden_neurons), Tanh(),
            Linear(hidden_neurons, hidden_neurons), BatchNorm1D(hidden_neurons), Tanh(),
            Linear(hidden_neurons, vocab_size), BatchNorm1D(vocab_size), 
        ]
    
    # initialization 
    gain = 5/3
    with torch.no_grad():
        if False: # when there was no BN
            # make last layer less confident 
            layers[-1].weight *= 0.1
        else:
            layers[-1].gamma *= 0.1 

        # apply gain as in kaiming init to all other layers
        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= gain
                # if the gain is 1, all subsequent layers will be squashed by the tanh. so some gain is necessary to fight back the squashing caused by the tanh
                # to get a good idea, generate hist_out_tanh.png with gain of 1, 3 and with 5/3 and compare!
        
    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print(f"Num of parameters in totral: {sum(p.nelement() for p in parameters)}")
    for p in parameters:
        p.requires_grad = True

    # same optimization as mlp.py
    max_epochs = 200000
    batch_size = 32
    loss_epoch = []
    update_to_data_ratio = []

    for epoch in range(max_epochs):
        # minibatch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward
        embeddings = C[Xb] # embed the characters into vectors
        x = embeddings.view(embeddings.shape[0], -1) # concatenate the  vectors 
        for layer in layers:
            x = layer(x)
        
        loss = torch.nn.functional.cross_entropy(x, Yb)
        
        # backward pass 
        for layer in layers:
            layer.out.retain_grad() # for plotting hist in 156-167 - comment it after
        for p in parameters:
            p.grad = None
        loss.backward()

        # update 
        lr = 0.1 if epoch < 100000 else 0.01 # step learning rate decay 
        for p in parameters:
            p.data -= p.grad * lr
        
        # track stats 
        if epoch % 10000 == 0:
            print(f"{epoch}/{max_epochs} Training loss: {loss.item():.4f} ")
        loss_epoch.append(loss.log10().item())
        
        # update to data ratio is important while training as well

        with torch.no_grad():
            update_to_data_ratio.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

        if epoch >=1000:
            break 

    # visualize activation hist after 1st epoch
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t = layer.out
            print(f"Layer[{i}] : {layer.__class__.__name__} : {t.mean()=} {t.std()=} saturated={(t.abs() > 0.97).float().mean()*100}%")
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f"layer {i} ({layer.__class__.__name__})")
    plt.legend(legends)
    plt.title("activation distribution")
    plt.savefig("hist_bn_tanh_outs_gain=%f.png"%(gain))
    """
    This is without BN - gain = 5/3
    Layer[1] : Tanh : t.mean()=tensor(-0.0234, grad_fn=<MeanBackward0>) t.std()=tensor(0.7498, grad_fn=<StdBackward0>) saturated=20.25%
    Layer[3] : Tanh : t.mean()=tensor(-0.0029, grad_fn=<MeanBackward0>) t.std()=tensor(0.6864, grad_fn=<StdBackward0>) saturated=8.375%
    Layer[5] : Tanh : t.mean()=tensor(0.0013, grad_fn=<MeanBackward0>) t.std()=tensor(0.6732, grad_fn=<StdBackward0>) saturated=6.624999523162842%
    Layer[7] : Tanh : t.mean()=tensor(-0.0060, grad_fn=<MeanBackward0>) t.std()=tensor(0.6569, grad_fn=<StdBackward0>) saturated=5.46875%
    Layer[9] : Tanh : t.mean()=tensor(-0.0207, grad_fn=<MeanBackward0>) t.std()=tensor(0.6626, grad_fn=<StdBackward0>) saturated=6.125%
    
    This is without BN - gain = 3
    Layer[1] : Tanh : t.mean()=tensor(-0.0315, grad_fn=<MeanBackward0>) t.std()=tensor(0.8534, grad_fn=<StdBackward0>) saturated=47.65625%
    Layer[3] : Tanh : t.mean()=tensor(0.0014, grad_fn=<MeanBackward0>) t.std()=tensor(0.8375, grad_fn=<StdBackward0>) saturated=40.46875%
    Layer[5] : Tanh : t.mean()=tensor(-0.0090, grad_fn=<MeanBackward0>) t.std()=tensor(0.8433, grad_fn=<StdBackward0>) saturated=42.375%
    Layer[7] : Tanh : t.mean()=tensor(-0.0139, grad_fn=<MeanBackward0>) t.std()=tensor(0.8433, grad_fn=<StdBackward0>) saturated=42.0%
    Layer[9] : Tanh : t.mean()=tensor(-0.0267, grad_fn=<MeanBackward0>) t.std()=tensor(0.8447, grad_fn=<StdBackward0>) saturated=42.40625%
    
    This is without BN - gain = 0.5
    Layer[1] : Tanh : t.mean()=tensor(-0.0104, grad_fn=<MeanBackward0>) t.std()=tensor(0.4102, grad_fn=<StdBackward0>) saturated=0.0%
    Layer[3] : Tanh : t.mean()=tensor(-0.0028, grad_fn=<MeanBackward0>) t.std()=tensor(0.1953, grad_fn=<StdBackward0>) saturated=0.0%
    Layer[5] : Tanh : t.mean()=tensor(0.0002, grad_fn=<MeanBackward0>) t.std()=tensor(0.0994, grad_fn=<StdBackward0>) saturated=0.0%
    Layer[7] : Tanh : t.mean()=tensor(0.0007, grad_fn=<MeanBackward0>) t.std()=tensor(0.0478, grad_fn=<StdBackward0>) saturated=0.0%
    Layer[9] : Tanh : t.mean()=tensor(-0.0014, grad_fn=<MeanBackward0>) t.std()=tensor(0.0245, grad_fn=<StdBackward0>) saturated=0.0%
    
    This is with BN - gain = 5/3
    Layer[2] : Tanh : t.mean()=tensor(-0.0036, grad_fn=<MeanBackward0>) t.std()=tensor(0.6323, grad_fn=<StdBackward0>) saturated=2.625%
    Layer[5] : Tanh : t.mean()=tensor(8.5111e-05, grad_fn=<MeanBackward0>) t.std()=tensor(0.6423, grad_fn=<StdBackward0>) saturated=2.46875%
    Layer[8] : Tanh : t.mean()=tensor(-0.0010, grad_fn=<MeanBackward0>) t.std()=tensor(0.6446, grad_fn=<StdBackward0>) saturated=2.15625%
    Layer[11] : Tanh : t.mean()=tensor(0.0014, grad_fn=<MeanBackward0>) t.std()=tensor(0.6471, grad_fn=<StdBackward0>) saturated=1.8125%
    Layer[14] : Tanh : t.mean()=tensor(-0.0025, grad_fn=<MeanBackward0>) t.std()=tensor(0.6465, grad_fn=<StdBackward0>) saturated=1.78125%
    """


    # visualize gradient hist after 1st epoch
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t = layer.out.grad
            print(f"Layer[{i}] : {layer.__class__.__name__} : {t.mean()=} {t.std()=} saturated={(t.abs() > 0.97).float().mean()*100}%")
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f"layer {i} ({layer.__class__.__name__})")
    plt.legend(legends)
    plt.title("gradient distribution")
    plt.savefig("hist_bn_tanh_outs_grad_gain=%f.png"%(gain))
    """
    This is without BN - gain = 5/3
    Layer[1] : Tanh : t.mean()=tensor(1.0004e-05) t.std()=tensor(0.0004) saturated=0.0%
    Layer[3] : Tanh : t.mean()=tensor(-3.4323e-06) t.std()=tensor(0.0004) saturated=0.0%
    Layer[5] : Tanh : t.mean()=tensor(3.0796e-06) t.std()=tensor(0.0004) saturated=0.0%
    Layer[7] : Tanh : t.mean()=tensor(1.5073e-05) t.std()=tensor(0.0003) saturated=0.0%
    Layer[9] : Tanh : t.mean()=tensor(-1.3680e-05) t.std()=tensor(0.0003) saturated=0.0%

    This is without BN - gain = 3
    Layer[1] : Tanh : t.mean()=tensor(-1.3384e-06) t.std()=tensor(0.0010) saturated=0.0%
    Layer[3] : Tanh : t.mean()=tensor(9.5665e-06) t.std()=tensor(0.0007) saturated=0.0%
    Layer[5] : Tanh : t.mean()=tensor(3.2144e-06) t.std()=tensor(0.0006) saturated=0.0%
    Layer[7] : Tanh : t.mean()=tensor(1.7020e-05) t.std()=tensor(0.0004) saturated=0.0%
    Layer[9] : Tanh : t.mean()=tensor(-1.3779e-05) t.std()=tensor(0.0003) saturated=0.0%

    This is without BN - gain = 0.5
    Layer[1] : Tanh : t.mean()=tensor(3.2467e-07) t.std()=tensor(1.8924e-05) saturated=0.0%
    Layer[3] : Tanh : t.mean()=tensor(-5.2128e-07) t.std()=tensor(3.9435e-05) saturated=0.0%
    Layer[5] : Tanh : t.mean()=tensor(3.7275e-06) t.std()=tensor(8.0354e-05) saturated=0.0%
    Layer[7] : Tanh : t.mean()=tensor(9.3031e-06) t.std()=tensor(0.0002) saturated=0.0%
    Layer[9] : Tanh : t.mean()=tensor(-1.3613e-05) t.std()=tensor(0.0003) saturated=0.0%
    
    This is with BN - gain = 5/3
    Layer[2] : Tanh : t.mean()=tensor(-4.6566e-12) t.std()=tensor(0.0037) saturated=0.0%
    Layer[5] : Tanh : t.mean()=tensor(7.5670e-12) t.std()=tensor(0.0033) saturated=0.0%
    Layer[8] : Tanh : t.mean()=tensor(4.6566e-12) t.std()=tensor(0.0030) saturated=0.0%
    Layer[11] : Tanh : t.mean()=tensor(1.7462e-11) t.std()=tensor(0.0027) saturated=0.0%
    Layer[14] : Tanh : t.mean()=tensor(-2.0373e-12) t.std()=tensor(0.0026) saturated=0.0%
    """

    # visualize weight gradient hist after 1st epoch
    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2: # only for weights
            t = p.grad
            print(f"weight {i} : {p.shape} {t.mean()=} {t.std()=} grad/data ratio={t.std()/p.std()}")
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f"{i} {tuple(p.shape)}")
    plt.legend(legends)
    plt.title("weight gradient distribution")
    plt.savefig("hist_bn_weights_grad_gain=%f.png"%(gain))
    """
    If grad/data ratio number is too large, that means grad step is much larger than actual data and convergence will not be possible

    This is without BN - gain = 5/3
    weight 0 : torch.Size([27, 10]) t.mean()=tensor(-3.0730e-05) t.std()=tensor(0.0014) grad/data ratio=0.0013640898978337646
    weight 1 : torch.Size([30, 100]) t.mean()=tensor(-4.9288e-05) t.std()=tensor(0.0012) grad/data ratio=0.00387165998108685
    weight 3 : torch.Size([100, 100]) t.mean()=tensor(1.6358e-05) t.std()=tensor(0.0011) grad/data ratio=0.00660198787227273    
    weight 5 : torch.Size([100, 100]) t.mean()=tensor(-1.0099e-05) t.std()=tensor(0.0010) grad/data ratio=0.005893091205507517  
    weight 7 : torch.Size([100, 100]) t.mean()=tensor(-1.1118e-05) t.std()=tensor(0.0009) grad/data ratio=0.0051581235602498055 
    weight 9 : torch.Size([100, 100]) t.mean()=tensor(-4.1436e-06) t.std()=tensor(0.0007) grad/data ratio=0.004415211267769337  
    weight 11 : torch.Size([100, 27]) t.mean()=tensor(-9.3822e-11) t.std()=tensor(0.0236) grad/data ratio=2.328202724456787
    
    This is with BN - gain = 5/3
    weight 0 : torch.Size([27, 10]) t.mean()=tensor(-5.5189e-11) t.std()=tensor(0.0102) grad/data ratio=0.010188532061874866
    weight 1 : torch.Size([30, 100]) t.mean()=tensor(9.2228e-05) t.std()=tensor(0.0082) grad/data ratio=0.02635848894715309
    weight 5 : torch.Size([100, 100]) t.mean()=tensor(3.0526e-05) t.std()=tensor(0.0073) grad/data ratio=0.043771594762802124   
    weight 9 : torch.Size([100, 100]) t.mean()=tensor(-1.8532e-05) t.std()=tensor(0.0067) grad/data ratio=0.03949902579188347   
    weight 13 : torch.Size([100, 100]) t.mean()=tensor(-1.9149e-05) t.std()=tensor(0.0058) grad/data ratio=0.034681666642427444 
    weight 17 : torch.Size([100, 100]) t.mean()=tensor(5.2631e-05) t.std()=tensor(0.0054) grad/data ratio=0.03243374451994896   
    weight 21 : torch.Size([100, 27]) t.mean()=tensor(-0.0002) t.std()=tensor(0.0105) grad/data ratio=0.06281279772520065
    """

    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2: # only for weights
            plt.plot([update_to_data_ratio[j][i] for j in range(len(update_to_data_ratio))])
            legends.append(f"param %d" % i)
    plt.plot([0, len(update_to_data_ratio)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
    plt.legend(legends)
    plt.title("update to data ratio")
    plt.savefig("plot_bn_update_to_data_ratiogain=%f.png"%(gain))