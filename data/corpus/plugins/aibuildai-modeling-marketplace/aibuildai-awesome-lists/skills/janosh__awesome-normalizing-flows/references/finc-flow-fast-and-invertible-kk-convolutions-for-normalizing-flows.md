---
title: "FInC Flow: Fast and Invertible k×k Convolutions for Normalizing Flows"
entry_type: paper
source: "https://arxiv.org/abs/2301.09266"
upstream_list: "janosh/awesome-normalizing-flows"
year: "2023"
authors: "Kallapa, Nagar et al."
code_url: "https://github.com/aditya-v-kallappa/FInCFlow"
description: "propose a k×k convolutional layer and Deep Normalizing Flow architecture which i) has a fast parallel inversion algorithm with running time O(nk^2) (n is height and width of the input image and k is kernel size), ii) masks the minimal amount of learnable parameters in a layer. iii) gives better forward pass and sampling times comparable to other k×k convolution-based models on real-world benchmarks. We provide an implementation of the proposed parallel algorithm for sampling using our invertible convolutions on GPUs. [[Code](https://github.com/aditya-v-kallappa/FInCFlow)]"
---

# FInC Flow: Fast and Invertible k×k Convolutions for Normalizing Flows

**Source**: [https://arxiv.org/abs/2301.09266](https://arxiv.org/abs/2301.09266)

**Code**: [https://github.com/aditya-v-kallappa/FInCFlow](https://github.com/aditya-v-kallappa/FInCFlow)

**Year**: 2023

**Authors**: Kallapa, Nagar et al.

## Description

propose a k×k convolutional layer and Deep Normalizing Flow architecture which i) has a fast parallel inversion algorithm with running time O(nk^2) (n is height and width of the input image and k is kernel size), ii) masks the minimal amount of learnable parameters in a layer. iii) gives better forward pass and sampling times comparable to other k×k convolution-based models on real-world benchmarks. We provide an implementation of the proposed parallel algorithm for sampling using our invertible convolutions on GPUs. [[Code](https://github.com/aditya-v-kallappa/FInCFlow)]
