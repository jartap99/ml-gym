import os
import numpy 
import time 
import sys 
import argparse 
import torch 
from dataclasses import dataclass 
import matplotlib.pyplot as plt 

from bigrams import create_dataset


def create_dataset_splits(input_file):
    # load and pre-process input file 
    """
    Create dataset splits by loading and pre-processing the input file.
    
    Parameters:
        input_file (str): The path to the input file.
    
    Returns:
        tuple: A tuple containing the training words, test words, and unique characters.
    """
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

    # partition dataset into training and test sets
    # test set is 20% of the dataset or 1000 samples
    test_set_size = min(len(words) * 0.2, 1000)
    rp = torch.randperm(len(words)).tolist() # random permutation
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[test_set_size:]]
    print(f"Split the dataset, first {len(train_words)} words as train data and {len(test_words)} words as test data")
    return train_words, test_words, unique_chars


def trigram_model(dataset, unique_chars):
    unique_chars = ['.'] + unique_chars # make speial char as first char
    num_unique_chars = len(unique_chars)  
    print(f"Number of unique characters : {num_unique_chars}")
    a = torch.zeros((num_unique_chars, num_unique_chars, num_unique_chars), dtype=torch.int32) 

    stoi = {c:i for i, c in enumerate(unique_chars)}
    itos = {i:s for s, i in stoi.items()}
    
    # first attempt at training once
    for w in dataset:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip (chs, chs[1:], chs[2:]):
            a[stoi[ch1], stoi[ch2], stoi[ch3]] += 1
    print(f"Number of Trigrams : {torch.count_nonzero(torch.reshape(a, (num_unique_chars*num_unique_chars*num_unique_chars, 1)))}")
    #N = torch.zeros((num_unique_chars*num_unique_chars, num_unique_chars), dtype=torch.int32)
    
    if False:
        plt.figure(1, figsize=(16,384))
        plt.imshow(torch.zeros((num_unique_chars*num_unique_chars, num_unique_chars)), cmap='Blues')
        for i in range(num_unique_chars):
            for j in range(num_unique_chars):
                for k in range(num_unique_chars):
                    chstr = itos[i] + itos[j] + itos[k]
                    plt.text(k, i*num_unique_chars + j, chstr, ha='center', va='bottom', color='black')
                    plt.text(k, i*num_unique_chars + j, a[i, j, k].item(), ha='center', va='top', color='black' if a[i, j, k] > 0 else 'gray')
        plt.savefig("trigrams.png")

    # generate 5 new words using bigram - inference
    g = torch.Generator().manual_seed(args.seed)
    P = (a+2).float() # adding 2 to smooth out and make log pro non-zero - see like 125
    P /= P.sum(dim=2, keepdim=True) # normalize along row
    print(f"{P.shape=} {P[0,0,:]=}")
    for i in range(10):
        out = []
        idx, idy = 0, 0
        while True:
            p = P[idx, idy] # p=a[idx].float(); p/= p.sum()
            idz = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[idz])
            if idz==0:
                break 
            else:
                idx, idy = idy, idz
        print(f"Generated word {i} : {''.join(out)}")
    
    log_likelihood = 0        
    num = 0
    for w in dataset:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip (chs, chs[1:], chs[2:]):
            prob = P[stoi[ch1], stoi[ch2], stoi[ch3]]
            log_prob = torch.log(prob)
            log_likelihood += log_prob
            num += 1
    neg_log_likelihood = -log_likelihood/num
    print(f"Trigrams {log_likelihood=} {neg_log_likelihood=}")
    print(f"Bigrams nll:  neg_log_likelihood=tensor(2.4549)")

    # job of model would be find parameters that minimize neg log (max log likelihood)
    # a neural network approxh is shown in next function below
    

def trigram_neural_net(dataset, unique_chars):
    unique_chars = ['.'] + unique_chars # make speial char as first char
    num_unique_chars = len(unique_chars)  
    print(f"Number of unique characters : {num_unique_chars}")
    a = torch.zeros((num_unique_chars, num_unique_chars, num_unique_chars), dtype=torch.int64) 
    stoi = {c:i for i, c in enumerate(unique_chars)}
    itos = {i:s for s, i in stoi.items()}
    
    # creating a train set
    xs, ys = [], []
    pairs = [(a,b) for a in unique_chars for b in unique_chars]
    ptoi = {pair:i for i, pair in enumerate(pairs)}
    itop = {i:pair for pair, i in ptoi.items()}
    print(f"{len(ptoi)=} {len(itop)=}")

    for w in dataset:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip (chs, chs[1:], chs[2:]):
            xs.append(ptoi[(ch1, ch2)])
            ys.append(stoi[ch3])
    xs = torch.tensor(xs, dtype=torch.int64) 
    ys = torch.tensor(ys, dtype=torch.int64)
    print(f"{xs.shape=} {ys.shape=}")
    
    # create neural net 
    g = torch.Generator(device='cuda').manual_seed(args.seed)
    W = torch.randn((num_unique_chars**2, num_unique_chars), requires_grad=True, generator=g, device='cuda')
    # inputs = (n, 729), w = (729, 27)
    
    print('number of examples: ', xs.nelement())
    
    xenc =  torch.nn.functional.one_hot(xs, num_classes=num_unique_chars*num_unique_chars).float()
    print(f"{xenc.shape=}")
    #print(xs[15])
    #print(xenc[15], len(xenc[15]))
    optim3 = True 
    for epoch in range(3000):
        # forward pass 
        logits = xenc.to('cuda') @ W # predict log counts (n, 729) * (729, 27)
        counts = logits.exp() # count equivalent to a 
        if optim3 is False:
            probs = counts / counts.sum(dim=1, keepdim=True) # normalize counts - prob for next char 
            loss = -probs[torch.arange(xs.nelement()), ys].log().mean() + 0.001*(W**2).mean()
        else:
            loss = torch.nn.functional.cross_entropy(logits, ys.to('cuda'), label_smoothing=0.0)
        print(f"{epoch=} {loss.item()=}")

        # backward pass
        W.grad = None    # clear gradients
        #b.grad = None    # clear gradients
        loss.backward()

        # update 
        W.data += W.grad * -30 # update weights
        
    # generate 10 new words using bigram - inference
    g = torch.Generator(device='cuda').manual_seed(args.seed)
    optim1 = True 
    optim2 = True
    for i in range(15):
        out = []
        ch1, ch2 = '.', '.'
        while True:
            ii = ptoi[(ch1, ch2)]
            if optim1 is False:
                xenc =  torch.nn.functional.one_hot(torch.tensor([ii]), num_classes=num_unique_chars*num_unique_chars).float().to('cuda')
                logits = xenc @ W 
            else:
                logits = torch.tensor(W[ii])
            if optim1 is False:
                counts = logits.exp() # count equivalent to a 
                p = counts / counts.sum(1, keepdims=True) # normalize counts - prob for next char 
            else:
                if optim2 is False:
                    counts = logits.exp() # count equivalent to a 
                    p = counts / counts.sum(0, keepdims=True) # normalize counts - prob for next char 
                else:
                    p = torch.nn.functional.softmax(logits, dim=0)
             
            idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[idx])
            ch1 = ch2 
            ch2 = itos[idx]
            if idx==0:
                break 
        print(f"Generated word {i} : {''.join(out)}")
    
    """
    epoch=1999 loss.item()=2.2296104431152344
    Generated word 0 : khya.
    Generated word 1 : ephyling.
    Generated word 2 : yulbgolbiahen.
    Generated word 3 : ramson.
    Generated word 4 : macxonnan.
    Generated word 5 : rine.
    Generated word 6 : delenlian.
    Generated word 7 : ermarishan.
    Generated word 8 : any.
    Generated word 9 : aleedon.
    Generated word 10 : lyashily.
    Generated word 11 : disonie.
    Generated word 12 : oakistente.
    Generated word 13 : sanismarilyn.
    Generated word 14 : avaugfnfy.
    """
    """
    with cross entropy loss
    epoch=1999 loss.item()=2.2393276691436768
    Generated word 0 : khya.
    Generated word 1 : ephyling.
    Generated word 2 : yulbgolbiahzimanisha.
    Generated word 3 : macxonnan.
    Generated word 4 : rine.
    Generated word 5 : delenlian.
    Generated word 6 : ermarishan.
    Generated word 7 : any.
    Generated word 8 : aleedon.
    Generated word 9 : lyashily.
    Generated word 10 : disonie.
    Generated word 11 : oakistente.
    Generated word 12 : sanismarilyn.
    Generated word 13 : avaugfnfy.
    Generated word 14 : jayantefabdugud.

    label_smoothing=0.01 and -30
    epoch=1999 loss.item()=2.253926992416382
    Generated word 0 : khya.
    Generated word 1 : ephyling.
    Generated word 2 : yulbgolbiahen.
    Generated word 3 : ramsha.
    Generated word 4 : macxonnan.
    Generated word 5 : rine.
    Generated word 6 : delenlian.
    Generated word 7 : ermarishan.
    Generated word 8 : any.
    Generated word 9 : aleedon.
    Generated word 10 : lyashily.
    Generated word 11 : disonie.
    Generated word 12 : oakistente.
    Generated word 13 : sanismarilyn.
    Generated word 14 : avaugfnfy.

    label_smoothing=0.001 and -30
    epoch=1999 loss.item()=2.232088565826416
    Generated word 0 : khya.
    Generated word 1 : ephyling.
    Generated word 2 : yulbgolbiahen.
    Generated word 3 : ramsha.
    Generated word 4 : macxonnan.
    Generated word 5 : rine.
    Generated word 6 : delenlian.
    Generated word 7 : ermarishan.
    Generated word 8 : any.
    Generated word 9 : aleedon.
    Generated word 10 : lyashily.
    Generated word 11 : disonie.
    Generated word 12 : oakistente.
    Generated word 13 : sanismarilyn.
    Generated word 14 : avaugfnfy.

    epoch=2999 loss.item()=2.2192389965057373
    Generated word 0 : khya.
    Generated word 1 : ephyling.
    Generated word 2 : yulbgolbiahen.
    Generated word 3 : ramson.
    Generated word 4 : macxonnan.
    Generated word 5 : rine.
    Generated word 6 : delenlian.
    Generated word 7 : ermarishan.
    Generated word 8 : any.
    Generated word 9 : aleedon.
    Generated word 10 : lyashily.
    Generated word 11 : disonie.
    Generated word 12 : oakistente.
    Generated word 13 : sanismarilyn.
    Generated word 14 : avaugfnfy.

    epoch=2999 loss.item()=2.2166786193847656
    Generated word 0 : khya.
    Generated word 1 : ephyling.
    Generated word 2 : yulbgolbiahen.
    Generated word 3 : ramson.
    Generated word 4 : macxonnan.
    Generated word 5 : rine.
    Generated word 6 : delenlian.
    Generated word 7 : ermarishan.
    Generated word 8 : any.
    Generated word 9 : aleedon.
    Generated word 10 : lyashily.
    Generated word 11 : disonie.
    Generated word 12 : oakistente.
    Generated word 13 : sanismarilyn.
    Generated word 14 : avaugfnfy.
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", '-i', type=str, default='names.txt', help='input file with one data sample per line', required=False)
    parser.add_argument("--seed", type=int, default=2147483647, help='random seed number for math, numpy and torch', required=False)
    args = parser.parse_args()
    print(args, vars(args))
    words, unique_chars = create_dataset(args.input_file)
    #trigram_model(words, unique_chars)
    trigram_neural_net(words, unique_chars)