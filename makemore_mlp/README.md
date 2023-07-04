# MLP 

This is makemore part 2 (MLP).

Based on paper by Bengio et al. : https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

Fighitng the curse of dimensionality.

Whats new from N-gram model?

1. Instead of using one-hot, assign some features for each work/character. Lets say 30. 
2. Corpus can have 100K word. Each word will have a unique vector of real numbers of size (1, 30). Had this been n-gram model, we would have needed 100L^n -1 one-hot vectors! Instead, now we always have (1, 30) vectors for every single word. 
3. The model learns the vectors and association of (x, y) together!

The objective is to learn a good model f(wt ,··· ,wt−n+1) = Pˆ(wt|w1 to t−1 ), in the sense that
it gives high out-of-sample likelihood. Below, we report the geometric average of 1/Pˆ(wt|w1 to t−1), also known as **perplexity**, which is also the exponential of the average negative log-likelihood.

Compare this with **Word2Vec**. 

Paper talks about distributed data parallel method of training network on cluster of Athlon processors using message passing library in 2003!

The paper builds word level language model with 17000 words. We apply this paper to character level model with whatever corpus we get from names.txt.

# Model in paper
* C is (17000 x 30) for embedding size of 30 (m). paper says 30, 60, 90. 
* Input layer has 30 neuron for 3 words, i.e. 90 neurons in total. 
* Hidden layer is hyper-parameter. 
* Output is 17000 neurons as there are 17000 possible answers, i.e. 17000 logits.
* * Embedding matrix C, weights biases of hidden layer, weights, biases of output layer are learnt through backprop.



# Adopting to out character/names model in makemore.

## Embedding 

We have 27 characters

So we will have a look-up table of (27 x embedding_size). This is C. 

In our case, embedd_size = 2, so C is (27,2)

Embedding_size dim is a hyper-parameter. 

Embeddings = C[X] !!!

X is (228146, 3) and Y is (228146) (before train-test-val split)

So Embeddings = (228146, 3, 2)

"Embeddings" can be viewed as first layer of this NN, without any non-linearity. C[X] is essentially ```F.one_hot(X, num_classes=27).float() @ C```.

## Hidden Layer

Size of hidden layer is another hyper-parameter.

Hidden layer gets 3 previous characters (block_size = 3) for our implementation. Each char is (3,2) since embed_size = 2.

W1 = (3*2, hidden_neurons) hidden_neurons = 100 for example - this is hyperparameter. 

b1 = (1, hidden_neurons)

## Output layer

Output has to predict one of the 27 characters. So 27 logits will be needed.

W2 = (hidden_neurons , 27)

b2 = (1, 27)


# Pytorch internals

http://blog.ezyang.com/2019/05/pytorch-internals/
