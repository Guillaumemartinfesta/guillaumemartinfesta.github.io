layout: post
title: Mixture of Experts: From the ground up to use in LLMs. 
date: 2024-10-02
description: .
tags: formatting links
categories: sample-posts
---


I/ Fondations of Mixture of Experts (MoE)

1) Theoretical Basis

Mixture of Experts are a special case of Latent (models such as Mixture of Gaussians) meaning that in order to fit a given distribution they introduce and rely on hidden variables.

We will here give the guiding example we will refer to in this example: 

Let's say we want to fit this distribution: 

{% include figure.liquid loading="eager" path="assets/img/mixture_of_experts/output.jpg" class="img-fluid rounded z-depth-1" %}


II/ Use of MoEs for LLMs


