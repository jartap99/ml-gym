
# What?
Effects of BN, initialization, how to visualize and decide best starting point, best initialization, structure, etc. 

# Without Batch Norm

## Why gain of 5/3 is better?

5/3 comes from Kaiming init. The histogram of activations and gradients explain what happens when gain is varied. These are captured after 1st epoch. 

All these are without BN and with TanH. Without TanH, subsequent layers go through diffusion, i.e. activations and gradiesnt spread wide and distribution is no longer concentrated at center. Without BN, these were the dificulties. In this case, when gain = 1, forward and backward pass have non shrinking distributions. 
Before Batch Norm, Adam, RMS prop, training NNs required these fine grain conrol of initialization, etc. The following 8 histograms show this pain. 

Without batch norm, the gain of 5/3 is the "art" for training the NNs. Bringing BN eliminates this gymnastics.

## Histogram of outputs with different gains without Batch Norm

### Gain = 0.5
![gain = 0.5](.\hist_tanh_outs_gain=0.500000.png)

### Gain = 1
![gain = 1](.\hist_tanh_outs_gain=1.000000.png)

### Gain = 5/3
![gain = 5/3](.\hist_tanh_outs_gain=1.666667.png)

### Gain = 3
![gain = 3](.\hist_tanh_outs_gain=3.000000.png)

## Histogram of output gradients with different gains

### Gain = 0.5
![gain = 0.5](.\hist_tanh_outs_grad_gain=0.500000.png)

### Gain = 1
![gain = 1](.\hist_tanh_outs_grad_gain=1.000000.png)

### Gain = 5/3
![gain = 5/3](.\hist_tanh_outs_grad_gain=1.666667.png)

### Gain = 3
![gain = 3](.\hist_tanh_outs_grad_gain=3.000000.png)

## Histogram of weight gradients with gain = 5/3
Last layer has really high std - dev!!
If you train for longer, this fixes itself. But it is still troubling if we use SGD instead of Adam. 

![gain = 5/3](.\hist_weights_grad_gain=1.666667.png)

## Plot of update to data ratio after 1000 epochs (gain = 5/3)
Andrej points out that he prefers these not too much above 1e-3. If it is below 1e-3, then the params are not trained fast enough. if lr is much lower, tihs plot will be much below the 1e-3 line!
We see that one param is being learnt way too slower than others.

![gain = 5/3](.\plot_update_to_data_ratiogain=1.666667.png)

When something is off, the above histograms would be weird, such as saturation being super high, weights all cramped at the end, weights and their gradients are all squashed, etc.

# With Batch Norm

## Histogram of outputs with Batch Norm (gain = 5/3)
Pretty good uniform std dev across all layers!
![gain = 5/3](.\hist_bn_tanh_outs_gain=1.666667.png)

## Histogram of outputs gradients with Batch Norm (gain = 5/3)
Compared to without BN, gradients are uniform across layers.
![gain = 5/3](.\hist_bn_tanh_outs_grad_gain=1.666667.png)

## Histogram of weight gradients with Batch Norm (gain = 5/3)
![gain = 5/3](.\hist_bn_weights_grad_gain=1.666667.png)

## Plot of update to data ratio after 1000 epochs with Batch Norm (gain = 5/3)
All updates are happening relatively at the same time. Improved from without BN.
![gain = 5/3](.\plot_bn_update_to_data_ratiogain=1.666667.png)

Now we should be not as sensitive to gain!! We do not get rid of this completely, but sensitivty to gain is slightly reduced with BN. Everything is significantly robust. fan_in initialization can be removed as well. lr may needed to be adjusted but that is a tunable hyper-param.      