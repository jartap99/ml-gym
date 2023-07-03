import os
import numpy 
import time 
import sys 
import argparse 
import torch 
from dataclasses import dataclass 
import matplotlib.pyplot as plt 


def create_dataset(input_file):
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

    # partition dataset into training and test sets
    # test set is 10% of the dataset or 1000 samples
    test_set_size = min(len(words) * 0.1, 1000)
    rp = torch.randperm(len(words)).tolist() # random permutation
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[test_set_size:]]
    print(f"Split the dataset, first {len(train_words)} words as train data and {len(test_words)} words as test data")
    return words, unique_chars

def bigram_model(dataset, unique_chars):
    if False:
        # build and print bigrams
        bigrams = {}
        for w in dataset:
            chs = ['<S>'] + list(w) + ['<E>']
            #print(chs)
            for ch1, ch2 in zip (chs, chs[1:]):
                bigram = (ch1, ch2)
                bigrams[bigram] = bigrams.get(bigram, 0) + 1
        bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
        print(f"Bigrams : {bigrams}")
        print(f"Number of bigrams : {len(bigrams)}")  
        unique_chars += ['<S>', '<E>']
        num_unique_chars = len(unique_chars)  
        print(f"Number of unique bigrams : {num_unique_chars}")
        a = torch.zeros((num_unique_chars, num_unique_chars), dtype=torch.int64) 
        stoi = {c:i for i, c in enumerate(unique_chars)}
        for i, c in enumerate(unique_chars):
            print(f"{unique_chars[i]=} {c=}: {stoi[c]=}")
        for bigram in bigrams:
            ch1, ch2 = bigram[0] 
            print(bigram, ch1, ch2)
            a[stoi[ch1], stoi[ch2]] = bigram[1]
        print("a: ", a)
    elif False:
        # visualize bigrams
        unique_chars += ['<S>', '<E>']
        num_unique_chars = len(unique_chars)  
        print(f"Number of unique characters : {num_unique_chars}")
        a = torch.zeros((num_unique_chars, num_unique_chars), dtype=torch.int64) 
        stoi = {c:i for i, c in enumerate(unique_chars)}
        itos = {i:s for s, i in stoi.items()}
        for w in dataset:
            chs = ['<S>'] + list(w) + ['<E>']
            for ch1, ch2 in zip (chs, chs[1:]):
                a[stoi[ch1], stoi[ch2]] += 1
        print(f"Number of Bigrams : {torch.count_nonzero(torch.reshape(a, (num_unique_chars*num_unique_chars, 1)))}")

        plt.figure(1, figsize=(16,16))
        plt.imshow(a, cmap='Blues')
        for i in range(num_unique_chars):
            for j in range(num_unique_chars):
                chstr = itos[i] + itos[j]
                plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
                plt.text(j, i, a[i, j].item(), ha='center', va='top', color='gray' if a[i, j] > 0 else 'black')
        
        plt.savefig("bigrams_0.png")

    else:
        unique_chars = ['.'] + unique_chars # make speial char as first char
        num_unique_chars = len(unique_chars)  
        print(f"Number of unique characters : {num_unique_chars}")
        a = torch.zeros((num_unique_chars, num_unique_chars), dtype=torch.int64) 
        stoi = {c:i for i, c in enumerate(unique_chars)}
        itos = {i:s for s, i in stoi.items()}
        
        # first attempt at training once
        for w in dataset:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip (chs, chs[1:]):
                a[stoi[ch1], stoi[ch2]] += 1
        print(f"Number of Bigrams : {torch.count_nonzero(torch.reshape(a, (num_unique_chars*num_unique_chars, 1)))}")

        plt.figure(1, figsize=(16,16))
        plt.imshow(a, cmap='Blues')
        for i in range(num_unique_chars):
            for j in range(num_unique_chars):
                chstr = itos[i] + itos[j]
                plt.text(j, i, chstr, ha='center', va='bottom', color='black')
                plt.text(j, i, a[i, j].item(), ha='center', va='top', color='black' if a[i, j] > 0 else 'gray')
        
        plt.savefig("bigrams.png")

        # generate 5 new words using bigram - inference
        g = torch.Generator().manual_seed(args.seed)
        P = (a+2).float() # adding 2 to smooth out and make log pro non-zero - see line 127
        P /= P.sum(dim=1, keepdim=True) # normalize along row
        for i in range(5):
            out = []
            idx = 0 
            while True:
                p = P[idx] # p=a[idx].float(); p/= p.sum()
                idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(itos[idx])
                if idx==0:
                    break 
            print(f"Generated word {i} : {''.join(out)}")
        
        # evaluate loss
        # max likelihood log prob 
        # negative log likelihood is our loss function for optimization!!!
        # prob goes to infinity if the word in inference sees a bigram with 0 prob. to avoid this, we can add fake counts
        log_likelihood = 0        
        num = 0
        for w in dataset:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip (chs, chs[1:]):
                prob = P[stoi[ch1], stoi[ch2]]
                log_prob = torch.log(prob)
                log_likelihood += log_prob
                num += 1
        neg_log_likelihood = -log_likelihood/num
        print(f"{log_likelihood=} {neg_log_likelihood=}")

        # job of model would be find parameters that minimize neg log (max log likelihood)
        # a neural network approxh is shown in next function below


def bigram_neural_net(dataset, unique_chars):
    """
    A function that implements a bigram neural network.

    Parameters:
    - dataset (list): A list of strings representing the dataset.
    - unique_chars (list): A list of unique characters in the dataset.

    Returns:
    None
    """
    unique_chars = ['.'] + unique_chars # make speial char as first char
    num_unique_chars = len(unique_chars)  
    print(f"Number of unique characters : {num_unique_chars}")
    a = torch.zeros((num_unique_chars, num_unique_chars), dtype=torch.int64) 
    stoi = {c:i for i, c in enumerate(unique_chars)}
    itos = {i:s for s, i in stoi.items()}
    
    # creating a train set
    xs, ys = [], []
    for w in dataset:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip (chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    xs = torch.tensor(xs, dtype=torch.int64) # do you know the difference between torch.tensor and torch.Tensor?
    ys = torch.tensor(ys, dtype=torch.int64)
    #print(xs)
    #print(ys)
    print(f"{xs.shape=} {ys.shape=}")
    xenc = torch.nn.functional.one_hot(xs, num_classes=num_unique_chars).float()
    yenc = torch.nn.functional.one_hot(ys, num_classes=num_unique_chars).float()
    #print(xenc)
    #print(yenc)
    print(f"{xenc.shape=} {yenc.shape=}")

    # create neural net 
    g = torch.Generator().manual_seed(args.seed)
    W = torch.randn((num_unique_chars, num_unique_chars), requires_grad=True, generator=g)
    b = torch.randn((1, 1), requires_grad=True, generator=g)
    ypred = xenc @ W # torch matmul
    ypred += b
    print(f"{ypred.shape=}")
    #print(f"{ypred=}")
    # intuitively we are trying to come up with 27 (num_unique_chars) probabilities for each neuron, each prob is for each char
    # we want these prob to represent some probability distribution 
    # these 27 numbers are giving us log of count . neural net is going to output log of counts
    # the way to get it is take exp of the output of neural net , then normalize it
    ypred = torch.exp(ypred) # this is equivalent to the "a" matrix above in line 96 
    ypred /= ypred.sum(dim=1, keepdim=True) # this is softmax 
    print(f"{ypred.shape=}")    
    #print(f"{ypred.sum=}")

    """
    xs.shape=torch.Size([228146]) ys.shape=torch.Size([228146])
    xenc.shape=torch.Size([228146, 27]) yenc.shape=torch.Size([228146, 27])
    ypred.shape=torch.Size([228146, 27])
    ypred.shape=torch.Size([228146, 27])
    """
    # now lets train 
    W = torch.randn((num_unique_chars, num_unique_chars), requires_grad=True, generator=g)
    #b = torch.randn((1, 1), requires_grad=True, generator=g)
    print('number of examples: ', xs.nelement())
    
    for epoch in range(500):
        # forward pass 
        xenc =  torch.nn.functional.one_hot(xs, num_classes=num_unique_chars).float()
        #print(f"{xenc.shape=}")
        #print(f"{W.shape=}")
        logits = xenc @ W #+ b # predict log counts 
        counts = logits.exp() # count equivalent to a 
        probs = counts / counts.sum(dim=1, keepdim=True) # normalize counts - prob for next char 
        loss = -probs[torch.arange(xs.nelement()), ys].log().mean() + 0.01*(W**2).mean()
        print(f"{epoch=} {loss.item()=}")
        # regularization loss: 0.01*(W**2).mean() - 0.01 is regularization strength
        # this pushes W to be 0. 

        # backward pass
        W.grad = None    # clear gradients
        #b.grad = None    # clear gradients
        loss.backward()

        # update 
        W.data += W.grad * -50 # update weights with a learning rate
        
        #should we update bias?
        # b.data -= b.grad * 0.01 # update biases

    # in line 109, we added '2' to all elements to get rid of cases where log is infinity, i.e. we smoothened the labels. 
    # this process of label smoothing products somewhat uniform probablity. smoothening gets fine when the number added is relatively larger 
    # than the counts of the a, e.g. adding 10000 to this example

    # in neural nets, the equivalent of that is initializing W's are all equal to each other and close to 0 
    # then logits will become all 0 and counts will become all 1s and probabilities will become all uniform
    # the more incentivize this in loss function, the more smooth distribution would be achieved
    # this is achieved by regularization. incentiving W to be around 0 results in smoothening. This is regulaization - check next few comments.

    # add regularization loss to loss function 
    # example: (W**2).mean() = regulatization loss (strength = 0.01)
    # the strength of regularization corresponds to adding a higher number in line 111. 

    # now sample - i.e. inference 
    # generate 5 new words using bigram - inference
    g = torch.Generator().manual_seed(args.seed)
    for i in range(10):
        out = []
        idx = 0 
        while True:
            xenc =  torch.nn.functional.one_hot(torch.tensor([idx]), num_classes=num_unique_chars).float()
            logits = xenc @ W #+ b # predict log counts 
            #print(f"{logits.shape=}")
            counts = logits.exp() # count equivalent to a 
            #print(f"{counts.shape=}")
            #print(f"{counts=}")
            p = counts / counts.sum(1, keepdims=True) # normalize counts - prob for next char 
            

            idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[idx])
            if idx==0:
                break 
        print(f"Generated work {i} : {''.join(out)}")

    """
    ...
    ...
    epoch=497 loss.item()=2.480742931365967
    epoch=498 loss.item()=2.480741262435913
    epoch=499 loss.item()=2.4807395935058594
    Generated work 0 : junide.
    Generated work 1 : janasah.
    Generated work 2 : p.
    Generated work 3 : cfay.
    Generated work 4 : a.
    Generated work 5 : nn.
    Generated work 6 : kohin.
    Generated work 7 : tolian.
    Generated work 8 : juwe.
    Generated work 9 : ksahnaauranilevias.
    """
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", '-i', type=str, default='names.txt', help='input file with one data sample per line', required=False)
    parser.add_argument("--seed", type=int, default=2147483647, help='random seed number for math, numpy and torch', required=False)
    args = parser.parse_args()
    print(args, vars(args))
    words, unique_chars = create_dataset(args.input_file)
    #bigram_model(words, unique_chars)
    bigram_neural_net(words, unique_chars)