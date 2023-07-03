import torch 
import matplotlib.pyplot as plt
import random 



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


def eval(X, Y, C, W1, b1, W2, b2, lossFn):
    embeddings = C[X]
    emb = embeddings.view(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
    out1 = torch.tanh(emb @ W1 + b1)
    logits = out1 @ W2 + b2
    loss = lossFn(logits, Y)
    return loss 


if __name__ == "__main__":
    block_size =3
    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = create_dataset('names.txt', block_size = block_size)
    # X is (228146, 3) and Y is (228146)
    #Xtr.shape : torch.Size([182778, 3])
    #Ytr.shape : torch.Size([182778])
    #Xdev.shape : torch.Size([22633, 3])
    #Ydev.shape : torch.Size([22633])
    #Xte.shape : torch.Size([22735, 3])
    #Yte.shape : torch.Size([22735])
    
    g = torch.Generator().manual_seed(2147483647)
    
    if False:
        # Bengio et al mapped 17000 words into 30 dimensional space 
        # we will map 27 characters into 2 dimensional space 
        C = torch.randn((27, 2), generator=g) # 27 characters in 2 D space embedding 
        # C is (27, 2)
        embeddings = C[X]
        print(f"embeddings.shape : {embeddings.shape}")
        # embeddings is (228146, 3, 2)

        
        #emb = torch.cat((embeddings[:, 0, :], embeddings[:, 1, :], embeddings[:, 2, :]), 1)
        #emb = torch.cat(torch.unbind(embeddings, 1), 1)
        # below expression does the same thing as above 2
        emb = embeddings.view(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
        print(f"emb.shape : {emb.shape}")

        # first layer gets inputs: 3 symbols of 2D embedding (3 letters based on block_size = 3)
        # lets say we want 100 neurons in the first layer
        # outputs = (n, 6)^T @ (6, 100) = (n, 100)
        W1 = torch.randn((6, 100), generator=g)
        b1 = torch.randn((1, 100), generator=g)
        
        out1 = torch.tanh(emb @ W1 + b1)
        print(f"out1.shape : {out1.shape}")

        W2 = torch.randn((100, 27), generator=g) # input is 100 and outputs is 27 characters!
        b2 = torch.randn((1, 27), generator=g)

        logits = out1 @ W2 + b2
        print(f"logits.shape : {logits.shape}")

        parameters = [C, W1, b1, W2, b2]
        s = 0
        for p in parameters:
            p.require_grad = True
            s += p.nelement()
        print(f"s : {s}")

        probs = torch.nn.functional.softmax(logits, dim=1)  
        print(f"probs.shape : {probs.shape}")
        print(probs[0])
        print(Y[0])
        print(probs[0, Y[0]])

        loss = -probs[torch.arange(probs.shape[0]), Y].log().mean()
        print(f"loss : {loss}")

        loss = torch.nn.functional.cross_entropy(logits, Y)
        print(f"loss : {loss}")

        for p in parameters:
            p.grad = None
    else:
        emb_dim = 10
        hidden_neurons = 200
        C = torch.randn((27, emb_dim), generator=g) # 27 characters in 2 D space embedding 
        W1 = torch.randn((block_size*emb_dim, hidden_neurons), generator=g)
        b1 = torch.randn((1, hidden_neurons), generator=g)
        W2 = torch.randn((hidden_neurons, 27), generator=g) # input is 100 and outputs is 27 characters!
        b2 = torch.randn((1, 27), generator=g)

        parameters = [C, W1, b1, W2, b2]
        s = 0
        for p in parameters:
            p.requires_grad = True
            s += p.nelement()
            #print(p)
        print(f"s : {s}")

        loss_function = torch.nn.CrossEntropyLoss()    

        #lre = torch.linspace(-3, 0, 1000)
        #lrs = 10**lre 
        #lrei = []
        #lossi = []
        
        # training loop
        for epoch in range(50000):
            # mini batch 
            ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)

            embeddings = C[Xtr[ix]]
            emb = embeddings.view(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
        
            # forward pass 
            out1 = torch.tanh(emb @ W1 + b1)
            logits = out1 @ W2 + b2
            loss = loss_function(logits, Ytr[ix])
            #print(f"{i} loss : {loss} {loss.item()}")

            # backward 
            for p in parameters:
                p.grad = None
                #print(p)
            #import pdb 
            #pdb.set_trace()
            loss.backward()

            # update 
            #lr = lrs[i]
            #lrei.append(lre[i])
            #lossi.append(loss.item())
            # sweeping lr, we found lr = 0.1 to be most optimal - seee lr_analysis.png
            # 1. First sweep through possible lrs 
            # 2. The find optimal lr (here lr = 0.1)
            # 3. train witht hat lr for some time (here 30k steps)
            # 4. then decay lr to smaller values (here lr = 0.01) and train for few steps
            if epoch<30000: # lr decay
                lr = 0.1
            else:
                lr = 0.001
            for p in parameters:
                p.data += -p.grad * lr
        
            if epoch % 5000 == 0:
                loss_eval = eval(Xdev, Ydev, C, W1, b1, W2, b2, lossFn= loss_function)
                print(f"{epoch} Training loss: {loss} Dev loss: {loss_eval}")
                if False:
                    plt.figure(figsize=(10, 10))
                    plt.scatter(C[:, 0].data, C[:, 1].data, s=200, c='r')
                    for i in range(C.shape[0]):
                        plt.text(C[i,0].item(), C[i, 1].item(), itos[i], ha='center', va='center', color='white')
                    plt.grid('minor')
                    plt.savefig("embedding_vis_%d.png"%epoch)

        #plt.plot(lrei, lossi)
        #plt.savefig("lr_analysis.png")

        # sample from trained model 
        for _ in range(20):
            out = []
            context = [0] * block_size # initialize with ... 
            while True:
                emb = C[torch.tensor([context])] # (1, block_size, emb_dim)
                #print(f"emb.shape : {emb.shape}, {emb.view(1,-1).shape}, W1.shape : {W1.shape}, ")
                out1 = torch.tanh(emb.view(1, -1) @ W1 + b1)
                logits = out1 @ W2 + b2
                probs = torch.nn.functional.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item() 
                context = context[1:] + [ix] 
                out.append(ix)
                if ix==0:
                    break
            print(_, ''.join(itos[i] for i in out))
                
"""
Number of words in dataset 'names.txt' : 32033
Number of unique_chars in dataset : 26
Min len of word in dataset : 2
Max len of word in dataset : 15
Vocabulary : ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
stoi : {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '.': 0}
itos : {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '.'}
X.shape : torch.Size([182625, 3])
Y.shape : torch.Size([182625])
X.shape : torch.Size([22655, 3])
Y.shape : torch.Size([22655])
X.shape : torch.Size([22866, 3])
Y.shape : torch.Size([22866])
s : 11897
0 Training loss: 27.881732940673828 Dev loss: 25.140993118286133
5000 Training loss: 2.469109058380127 Dev loss: 2.719364881515503
10000 Training loss: 2.834111213684082 Dev loss: 2.6272904872894287
15000 Training loss: 2.284478187561035 Dev loss: 2.5587594509124756
20000 Training loss: 2.5523133277893066 Dev loss: 2.5141522884368896
25000 Training loss: 2.298656940460205 Dev loss: 2.4638264179229736
30000 Training loss: 2.8716726303100586 Dev loss: 2.4302148818969727
35000 Training loss: 2.4054901599884033 Dev loss: 2.2504801750183105
40000 Training loss: 2.0036261081695557 Dev loss: 2.2418313026428223
45000 Training loss: 1.804913878440857 Dev loss: 2.2380499839782715
0 kemr.
1 lonterichallyn.
2 ana.
3 lig.
4 aleegh.
5 raveu.
6 ten.
7 mor.
8 pyn.
9 yas.
10 frannaeoe.
11 azlira.
12 iver.
13 jose.
14 gyl.
15 avran.
16 jotyan.
17 jacy.
18 rakotusnicenrin.
19 gar.
"""