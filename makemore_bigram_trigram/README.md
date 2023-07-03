This repo is a gym to practice 
https://github.com/karpathy/makemore

This is part 1 of Makemore using Bigram, Trigram

# Bigram NN

## Structure 
Each x example is (1, 27) one hot encoded vector. 

Each x has a subsequent example of y (1, 27) one hot encoded vector. 

Take an example fo the word: ***emma***. The following are bigrams x y: 

. e

e m 

m m 

m a 

a . 


Each neuron outputs probability of the corresponding symbol. There shall be 27 neurons. Each neuron outputs 27 values, each value tells the "firing rate" of corresponding input. 

```
xenc  @ W = yenc
(1, 27) @ (27, 27) = (1, 27) 
```
xenc is a one hot vector. 

yenc is a some vector with positive and negative numbers. We want this to represent some form of probability vector. In order to do this, we must interpret these positive and negative numbers as log(counts). To get the "counts" equivalent, we must take exp(log(counts)) and then normalize them all to find out element of highest probability. All this can be achieved by doing a softmax on yenc. 

``` 
y_prob = softmax(xenc @ W) 
```
Read the comments in bigrams.py for details.

## Loss function
```
log likelihood = log (yprob) -> getting back the counts info after softmax
negative log likelihood = - log likelihood
minimize nll -> this is the loss function
```

### Regularization loss

Add regularization loss to loss function 

Example: (W**2).mean() = regulatization loss (strength = 0.01)

The strength of regularization corresponds to adding a higher number to counts (line 111). 

# Trigram NN
## Structure 
Each x example is (1, 27^2) one hot encoded vector. 1st symbol can be one of 27 one hot and 2nd symbol can be 27 one hots for ever 1st symbol.

Each x has a subsequent example of y (1, 27) one hot encoded vector. 

Each neuron outputs probability of the corresponding symbol. There shall be 27^2 neurons. Each neuron outputs 27 values, each value tells the "firing rate" of corresponding input. 

```
xenc  @ W = yenc
(1, 27^2) @ (27^2, 27) = (1, 27) 
```
xenc is a one hot vector. 

yenc is a some vector with positive and negative numbers. We want this to represent some form of probability vector. In order to do this, we must interpret these positive and negative numbers as log(counts). To get the "counts" equivalent, we must take exp(log(counts)) and then normalize them all to find out element of highest probability. All this can be achieved by doing a softmax on yenc. 

``` 
y_prob = softmax(xenc @ W) 
```
