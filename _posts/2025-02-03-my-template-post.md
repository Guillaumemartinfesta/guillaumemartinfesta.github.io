---
layout: post
title: "Mixture of Experts: From the ground up to use in LLMs."" 
date: 2024-10-02
description: ""
tags: formatting links
categories: llm
---


# I/ Fondations of Mixture of Experts (MoE)

## 1) Problem Description

Let's say we want to fit this distribution of y given x: 

We first introduce the direct problem:

{% include figure.liquid loading="eager" path="assets/img/mixture_of_experts/direct_problem.jpg" class="img-fluid rounded z-depth-1" %}

As we will see shortly after we swaped the x and y axis as the problem of interest will actually be the inverse of this one. 
We can see here that predicting x given y corresponds to predicting a trend x=f(y) and accounting for some measuring noise. 
As the trend is a function mapping from y to x (with only one x associated to a given y) the distribution we are trying to predict is unimodal, a sensible model to fit to this data would thus have this form:

$y \sim \mathcal{N}(\mu(x),\sigma^2)$

The parameters to fit for this model are the parameters $\theta$ of $\mu(x)$ estimating the mode/trend and $\sigma^2$ accounting for the residual noise. $\mu(x)$ could be any sort of model: linear regression, polynomial regression, kernel based model, neural network... 

This problem description is very classic and enables to simply fit by trying to minimize the NLL(Negative Log Likelihood). 



However considering the inverse problem by predicting y as a function of x does not lead to such nice properties:  

{% include figure.liquid loading="eager" path="assets/img/mixture_of_experts/inverse_problem.jpg" class="img-fluid rounded z-depth-1" %}

While the direct problem led to a simple unimodal distribution, we see that the inverse problem has 2 modes for x between 0.4 and 0.6 making the previous model obsolete. We see that such a phenomenon of a multimodal inverse problem arises whenever we consider the inverse of a function that is not injective.

Luckily Mixture of Experts(MoE) is a type of models that enables to solve this issue. MoEs belong to the class Latent Variable Models(LVM) that rely on some hidden variables (the latent variables) to make predictions. 

More formally we could introduce a categorical latent variable z such that $p(z|x)=Cat(z|Softmax(V^T(1,x)))$ that will determine the "expert" to use.The function p(z|x) is called the gating function. Each of the N experts can then be a model of the form described earlier $p(y|x,z=k)=\mathcal{N}(y|\mu_k(x),\sigma_k^2)$ describing an unimodal distribution. As for a given x we could have a nonzero probability of "activating" each expert, we can then describe any distribution having less than N modes. 

The model parameters are then $\theta_1,...,\theta_k$ and $V \in \mathcal{M}_{2 \times N}$. We denote by $\theta$ all these parameters. 

## 2) Fitting MoEs: the EM algorithm

Fitting a Latent Variable Model is not straight forward. The joint log probability $log(p(y,z|x,\theta))$ is easy to compute but the observed data log likelihood is hard to compute since $log(p(y|x,\theta))=log(\sum p(y,z|x,theta))$. 

The log can't be pushed into the sum. Moreover most of the simple models lead to a convex log likelihood but this property is lost for LVM as the logarithm of the sum of 2 log convex functions is not necessarly convex making the optimization problem much harder. 

As the complete data log probability is easy to compute we estimate the observed data log likelihood log(p(x)) by the expected complete data log likelihood $\mathbb{E}_{z \sim q(z|x,y)}[log(P(y,z|x))]$

The EM algorithm does exactly that in order by alternating between 2 steps:

- The E (Estimation) step  where we estimate $q(z_i|x_i,y_i)=p(z_i=k|x_i,y_i,\hat{\theta}^t)$

- The M (Maximization) step were we compute $\hat{\theta}^{t+1}=\underset{\theta}{argmax}\mathbb{E}_{z \sim q(z|x,y)}[log(P(y,z|x))]$

In other words the EM algorithm iteratively computes $\hat{\theta}^{t+1}=\underset{\theta}{argmax}\sum_{i=1}^n\sum_{k=1}^Nlog[p(y_i,z_i=k|x_i,\theta)]p(z_i=k|x_i,\hat{\theta^t})$

## 3) Theoretical proofs for the EM algorithm

We will show in this part that the EM algorithm monotically increases the observed data log likelihood. 

Our first goal is to derive a formula showing how far of our estimation of the observed data log likelihood by the expected complete data log likelihood is.

To make the point more general we assume the variable z is continuous (the same results also hold for discrete z)

\[
\begin{align}
\mathbb{E}_{z \sim q(z|x,y)}[log(p(y,z|x,\theta))] \\
= \int_zq(z|x,y)log(p(y,z|x,\theta))dz \\
= \int_zq(z|x,y) \left[ log(\frac{p(y,z|x,\theta)}{q(z|x,y)})+log(q(z|x,y))\right]dz \\
=\int_zq(z|x,y)log(\frac{p(z|x,y,\theta)p(y|x,\theta)}{q(z|x,y)})dz + \mathbb{H}_q \\
=\int_zq(z|x,y)log(p(y|x,\theta))dz - \int_zq(z|x,y)log(\frac{q(z|x,y)}{p(z|x,y,\theta)})dz + \mathbb{H}_q \\
=log(p(y|x,\theta))+D_{KL}(q(.|x,y)||p(.|x,y,\theta))+\mathbb{H}_q
\end{align}
\]

# II/ Use of MoEs for LLMs


