Copy of abstract:


From the list: Show how you can avoid learning the last layer classifier when you train a deep RELU classifier. How does this help? Experiments.
 
 
Title: Impact of Simplex ETF Final Layer on Intermediate Layer Neural Collapse in Deep Neural Nets

The phenomenon of neural collapse states that the features and weights of a deep overparameterized network will converge in the terminal phase of training such that (NC1) the within-class variability of features drops to zero, (NC2) the class means converge to simplex ETF, and (NC3) class means converge to last-layer weights. This suggests the possibility that we can reduce the complexity of the final layer of a deep classification network by constraining it to an ETF problem, since class means will converge. Additionally, since class means converge to classifier weights in NC, we can also avoid learning the last-layer weights by letting the weights equal to the features.
 
In this project, we seek to investigate the effect of the ETF last layer on neural collapse in intermediate layers. It is known that although an overparameterized model can exhibit NC even on random noise, models with better generalization capacity exhibit NC on intermediate layers as well. This project will investigate if setting the last layer to ETF allows for intermediate layers to exhibit NC phenomenon in fewer training steps when compared to a network exhibiting NC without an ETF final layer. We will test the link between generalization and intermediate NC under this new framework and compare it to generalization and intermediate NC in the non-ETF neural net.
 
Additionally, we will explore when and how the ETF-based network exhibits NC under different hyperparameter spaces (learning rate, batch size, lambda) compared to the non-ETF-based network. For example, it has been previously observed that NC will not occur if the learning rate is too low. Part of our investigation is to examine the link between hyperparameters, NC, and the ETF last layer in order to gain a better understanding of NC failure modes. Observing the effects different hyperparameters have on neural collapse in the ETF and non-ETF settings can potentially shed light on how this phenomenon occurs in neural networks.
 
The architectures we will use are MLP (L layers, change L), VGG, [ImageNet, DenseNet] with ReLU Activation. We will use the CIFAR-10 and CIFAR-100 datasets to see how the behavior changes with number of classes. 
 
References
https://arxiv.org/pdf/2008.08186.pdf – original NC paper
https://arxiv.org/pdf/2105.02375.pdf
https://arxiv.org/abs/2202.08384
https://arxiv.org/pdf/2206.04041.pdf 
review of NC, good resource
section 4 talks about generalization
https://arxiv.org/abs/2202.09028