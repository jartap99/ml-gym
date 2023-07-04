import torch 
import matplotlib.pyplot as plt 
import random 

# starting code is almost same as mlp


def build_dataset(words, stoi, block_size=3):
    """
    block_size = 3 # this is context length, i.e. how many characters do we take to predict next one 
    """
    X, Y = [], []
    for w in words:
        #print(w) 
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch] 
            X.append(context)
            Y.append(ix)
            #print(''.join(itos[c] for c in context), ' --> ', itos[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(f"X.shape : {X.shape}")
    print(f"Y.shape : {Y.shape}")
    return X, Y

def create_dataset(input_file, block_size=3):
    # load and pre-process input file 
    with open(input_file, 'r') as f:
        words = f.readlines()
    words = [w.strip() for w in words ]
    words = [w for w in words if len(w) > 0]
    unique_chars = sorted(set(''.join(words)))
    min_len = min(len(w) for w in words)
    max_len = max(len(w) for w in words)
    print(f"Number of words in dataset \'{input_file}\' : {len(words)}")
    print(f"Number of unique_chars in dataset : {len(unique_chars)}")
    print(f"Min len of word in dataset : {min_len}")
    print(f"Max len of word in dataset : {max_len}")
    print(f"Vocabulary : {unique_chars}")

    stoi = {c:i+1 for i, c in enumerate(unique_chars)}
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    print(f"stoi : {stoi}")
    print(f"itos : {itos}")
    
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], stoi, block_size=block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size=block_size)
    Xte, Yte = build_dataset(words[n2:], stoi, block_size=block_size)

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos

@torch.no_grad()
def eval(X, Y, C, W1, b1, W2, b2, bnmean_running, bnstd_running, bngain, bnbias, lossFn):
    embeddings = C[X]
    emb = embeddings.view(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
    hpreact = emb @ W1 + b1
    #hpmean = hpreact.mean(0, keepdim=True)
    #hpstd = hpreact.std(0, keepdim=True)
    hpreact = bngain*(hpreact - bnmean_running) / bnstd_running  + bnbias # batch norm 
    out1 = torch.tanh(hpreact)
    logits = out1 @ W2 + b2
    loss = lossFn(logits, Y)
    return loss 

if __name__ == "__main__":
    block_size =3
    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = create_dataset('names.txt', block_size = block_size)
    g = torch.Generator().manual_seed(2147483647)

    emb_dim = 10
    hidden_neurons = 200
    vocab_size = 27
    C = torch.randn((vocab_size, emb_dim), generator=g) # 27 characters in 2 D space embedding 
    
    # W1 = torch.randn((block_size*emb_dim, hidden_neurons), generator=g)
    # instead of doing the above, W1 should be initialized uniformly to have best approx of initial loss.
    # Initially, all elements are equally possible, so prob: 1/27
    # loss = -torch.tensor(1/27.0).log() = ~ 3.298 - this is also achieved by making W2 close to 0 as shown below.

    W1 = torch.randn((block_size*emb_dim, hidden_neurons), generator=g) *(5/3)/(block_size*emb_dim)**0.5 # using kaimint init instead of 0.01
    # W1 = torch.randn((block_size*emb_dim, hidden_neurons), generator=g) * 0.01 # to make hpreact small and a true gaussian around 0, avoiding asymptotic regions of tanh
    # gradient of tanh vanishes if W1 is not small
    # grad of tanh = (1-t**2)*out. if t is close to either 1 or 01, grad vanishes. if t -s 0, grad passes through.
    # sigmoidm relu also suffer from this. leaky relu doesnt suffer from this as much as it doesnt have flat tails
    # for same reason, squash b1 to small number too by multiplying by 0.01 

    b1 = torch.randn((1, hidden_neurons), generator=g) * 0.01 
    W2 = torch.randn((hidden_neurons, vocab_size), generator=g) *(1/(hidden_neurons)**0.5) # using kaimint init instead of 0.01
    #W2 *= 0.01 # to mke logits close to 0 
    # we should not initialize W2 to 0 to get logits=0 
    b2 = torch.randn((1, vocab_size), generator=g) * 0 # multiply by 0 to make logits close to 0
    # with these settings, logits is close to what we want, but has some entropy. This entropy is used for symmetry-breaking. 

    # if we have large neural nets, manually initializing the Ws and bs and choosing close to 0 is manual and tedious. no one does this.
    # Instead of multiplying by 0.01, if we multiply by 1/sqrt(fan-in count), it keeps output his of x @ W same as x. Fan in of W1 is block_size * emb_dim 
    # Kaiming init is sqrt(2/fan-in count)

    bngain = torch.ones((1, hidden_neurons))
    bnbias = torch.zeros((1, hidden_neurons))
    bnmean_running = torch.zeros((1, hidden_neurons))
    bnstd_running = torch.zeros((1, hidden_neurons))

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    s = 0
    for p in parameters:
        p.requires_grad = True
        s += p.nelement()
        #print(p)
    print(f"s : {s}")

    loss_function = torch.nn.CrossEntropyLoss()    

    max_epochs = 200000
    batch_size = 32
    loss_epoch = []
    for epoch in range(max_epochs):
        # minibatch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward pass 
        embeddings = C[Xb] # embed the characters into vectors
        embcat = embeddings.view(embeddings.shape[0], -1) # concatenate the  vectors 
        hpreact = embcat @ W1 + b1 # hidden layer pre-activation
        # we want hpreact to be roughly gaussian (at initialization, but not always) - not too small and not too wide. BN says normalize it for a batch
        # hpreact is (32, 200) - we need to find mean for every example of batch
        
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = bngain*(hpreact - bnmeani) / bnstdi  + bnbias # batch norm 
        # without bngain and bnbias, using this may not give good result as these force the hpreact to be always gaussian, not just at init.
        # h will have jitter based on other examples within the batch, i.e. other examples affect h through bngain and bnbias 
        # this acts like a bit of data augmenter.
        # layer norm, instance norm, group norm do not couple examples - they are independent
        # bnbias does the job of bias. So b1 can be eliminated from out model!!

        # instead of using bnmean per batch and then calibrating bn at the end of training, we use bnmean_running
        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani # momentum here is 0.001
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

        h = torch.tanh(hpreact) # hidden layer
        logits = h @ W2 + b2 # output layer
        # we want logits to be much closer to 0 
        loss = loss_function(logits, Yb) # loss function 

        # backward 
        for p in parameters:
            p.grad = None
        loss.backward()

        # update 
        lr = 0.1 if epoch < 100000 else 0.01
        for p in parameters:
            p.data += p.grad * (-lr)
    
        if epoch % 10000 == 0:
            print(f"{epoch}/{max_epochs} Training loss: {loss.item():.4f} ")
        loss_epoch.append(loss.log10().item())

        
        """
        plt.imshow(h.abs()>0.99, cmap='gray', interpolation='nearest')
        plt.savefig("h_vals.png")
        # if there a neuron which is always "white" for all examples, it is a dead neuron. this neuron never learns.
        break 
        
        hlist = hpreact.view(-1).detach().tolist()
        plt.hist(hlist, 50)
        plt.savefig("hpreact_hist.png")
        plt.close()

        hlist = h.view(-1).detach().tolist()
        plt.hist(hlist, 50)
        plt.savefig("h_hist.png")
        plt.close()
        break
        """

    plt.plot(loss_epoch)
    plt.savefig("loss_epoch.png")
    
    print(f"Train loss: {eval(Xtr, Ytr, C, W1, b1, W2, b2, bngain, bnbias, lossFn=loss_function)}")
    print(f"Val loss: {eval(Xdev, Ydev, C, W1, b1, W2, b2, bngain, bnbias, lossFn=loss_function)}")
    print(f"Val loss: {eval(Xte, Yte, C, W1, b1, W2, b2, bngain, bnbias, lossFn=loss_function)}")

    # when batch norm is enabled, after the training, bn has to be calibrated once for all inferences
    # by using bnmean_running and bnstd_running we eliminate the need to do this calibration
    if False:
        with torch.no_grad():
            emb = C[Xtr]
            embcat = emb.view(emb.shape[0], -1)
            hpreact = embcat @ W1 + b1
            bnmean = hpreact.mean(0, keepdim=True)
            bnstd = hpreact.std(0, keepdim=True)

    # sample from trained model 
    g = torch.Generator().manual_seed(2147483647 + 10)
    for _ in range(20):
        out = []
        context = [0] * block_size 
        while True:
            embeddings = C[torch.tensor(context)] 
            embcat = embeddings.view(1, -1) 
            hpreact = embcat @ W1 + b1
            hpreact = bngain*(hpreact - bnmean_running) / bnstd_running  + bnbias # batch norm 
            
            h = torch.tanh(hpreact) # hidden layer
            logits = h @ W2 + b2 # output layer
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item() 
            context = context[1:] + [ix] 
            out.append(ix)
            if ix==0:
                break
        print(_, ''.join(itos[i] for i in out))
    

    # 

"""
Default: 

0/200000 Training loss: 27.8817  -> this rapidly comes down. So initialization is not good. 
10000/200000 Training loss: 2.8341 
20000/200000 Training loss: 2.5523 
30000/200000 Training loss: 2.8717 
40000/200000 Training loss: 2.0855 
50000/200000 Training loss: 2.5842 
60000/200000 Training loss: 2.4150 
70000/200000 Training loss: 2.1321 
80000/200000 Training loss: 2.3674 
90000/200000 Training loss: 2.3077 
100000/200000 Training loss: 2.0464 
110000/200000 Training loss: 2.4816 
120000/200000 Training loss: 1.9383 
130000/200000 Training loss: 2.4820 
140000/200000 Training loss: 2.1662 
150000/200000 Training loss: 2.1765 
160000/200000 Training loss: 2.0685 
170000/200000 Training loss: 1.7901 
180000/200000 Training loss: 2.0546 
190000/200000 Training loss: 1.8380 
Train loss: 2.125401020050049
Val loss: 2.1713311672210693
Val loss: 2.1599643230438232
0 mora.
1 mayah.
2 seel.
3 nah.
4 yam.
5 rensleighdrae.
6 caileed.
7 elin.
8 shy.
9 jen.
10 eden.
11 estanaraelyn.
12 malke.
13 cayshuberlyni.
14 jest.
15 jair.
16 jenipanthono.
17 ubelleda.
18 kylynn.
19 els.
"""

"""
Fix softmax 
With W2 multiplied by 0.01 and b2 by 0, we get better loss

0/200000 Training loss: 3.7837 
10000/200000 Training loss: 2.2260 
20000/200000 Training loss: 2.2587 
30000/200000 Training loss: 2.5009 
40000/200000 Training loss: 1.9375 
50000/200000 Training loss: 2.4694 
60000/200000 Training loss: 2.4907 
70000/200000 Training loss: 2.0421 
80000/200000 Training loss: 2.2290 
90000/200000 Training loss: 2.1644 
100000/200000 Training loss: 1.8858 
110000/200000 Training loss: 2.2093 
120000/200000 Training loss: 1.9301 
130000/200000 Training loss: 2.4853 
140000/200000 Training loss: 2.2980 
150000/200000 Training loss: 2.1501 
160000/200000 Training loss: 1.9110 
170000/200000 Training loss: 1.7392 
180000/200000 Training loss: 1.9817 
190000/200000 Training loss: 1.7152 
Train loss: 2.0654759407043457
Val loss: 2.114626407623291
Val loss: 2.115468978881836
0 mora.
1 mayah.
2 seel.
3 ndheyah.
4 rensleigh.
5 raeg.
6 adee.
7 daelin.
8 shi.
9 jenleigh.
10 estanaraelyn.
11 malaia.
12 noshubrighairiel.
13 kin.
14 renlee.
15 jose.
16 caylee.
17 geder.
18 yarulyeha.
19 kayshayveyah.
"""

"""
fix tanh layer, too saturated at init 

0/200000 Training loss: 3.2972 
10000/200000 Training loss: 2.1634 
20000/200000 Training loss: 2.3241 
30000/200000 Training loss: 2.4002 
40000/200000 Training loss: 1.9654 
50000/200000 Training loss: 2.6275 
60000/200000 Training loss: 2.3651 
70000/200000 Training loss: 2.0442 
80000/200000 Training loss: 2.2509 
90000/200000 Training loss: 2.1626 
100000/200000 Training loss: 1.9017 
110000/200000 Training loss: 2.2478 
120000/200000 Training loss: 1.9956 
130000/200000 Training loss: 2.5327 
140000/200000 Training loss: 2.4242 
150000/200000 Training loss: 2.3073 
160000/200000 Training loss: 1.9175 
170000/200000 Training loss: 1.7827 
180000/200000 Training loss: 2.0671 
190000/200000 Training loss: 1.7780 
Train loss: 2.105128049850464
Val loss: 2.138484239578247
Val loss: 2.1426920890808105
0 mora.
1 mayah.
2 see.
3 mad.
4 ryla.
5 rensrei.
6 jdraegulie.
7 kaielii.
8 shi.
9 jen.
10 edelisson.
11 arleitzima.
12 kamin.
13 shubvrgiagriel.
14 kinde.
15 jelionnie.
16 casubenteder.
17 yarleyeks.
18 kaysh.
19 sanyamihan.
"""

"""
with kaiming init for W1 and W2 
0/200000 Training loss: 3.6625 
10000/200000 Training loss: 2.2123 
20000/200000 Training loss: 2.3782 
30000/200000 Training loss: 2.5114 
40000/200000 Training loss: 1.9937 
50000/200000 Training loss: 2.3217 
60000/200000 Training loss: 2.4265 
70000/200000 Training loss: 2.1442 
80000/200000 Training loss: 2.2340 
90000/200000 Training loss: 2.1547 
100000/200000 Training loss: 1.8714 
110000/200000 Training loss: 2.0707 
120000/200000 Training loss: 1.9715 
130000/200000 Training loss: 2.3380 
140000/200000 Training loss: 2.1489 
150000/200000 Training loss: 2.1644 
160000/200000 Training loss: 1.7945 
170000/200000 Training loss: 1.7688 
180000/200000 Training loss: 1.9060 
190000/200000 Training loss: 1.8048 
Train loss: 2.038471221923828
Val loss: 2.103687286376953
Val loss: 2.105879783630371
0 mora.
1 mayah.
2 seel.
3 ndheyah.
4 reish.
5 jendraeg.
6 adelyn.
7 elin.
8 shi.
9 jenne.
10 elisson.
11 arleigh.
12 malaia.
13 noshub.
14 roshira.
15 sten.
16 joselle.
17 jose.
18 cayus.
19 kynder.
"""

"""
with batch norm 
0/200000 Training loss: 3.5772 
10000/200000 Training loss: 2.1652 
20000/200000 Training loss: 2.4167 
30000/200000 Training loss: 2.4436 
40000/200000 Training loss: 2.0111 
50000/200000 Training loss: 2.3681 
60000/200000 Training loss: 2.4294 
70000/200000 Training loss: 2.1056 
80000/200000 Training loss: 2.3539 
90000/200000 Training loss: 2.1389 
100000/200000 Training loss: 1.9417 
110000/200000 Training loss: 2.3648 
120000/200000 Training loss: 1.9323 
130000/200000 Training loss: 2.4825 
140000/200000 Training loss: 2.3174 
150000/200000 Training loss: 2.1054 
160000/200000 Training loss: 1.9776 
170000/200000 Training loss: 1.8040 
180000/200000 Training loss: 1.9797 
190000/200000 Training loss: 1.8384 
Train loss: 2.068721294403076
Val loss: 2.108036756515503
Val loss: 2.1097121238708496
0 mora.
1 mayah.
2 see.
3 madhayla.
4 remmani.
5 jarlee.
6 adelyn.
7 elin.
8 shi.
9 jen.
10 eden.
11 sana.
12 arleigh.
13 malaia.
14 noshubergihiriel.
15 kin.
16 renlynn.
17 novana.
18 uberteda.
19 jamyli.
"""