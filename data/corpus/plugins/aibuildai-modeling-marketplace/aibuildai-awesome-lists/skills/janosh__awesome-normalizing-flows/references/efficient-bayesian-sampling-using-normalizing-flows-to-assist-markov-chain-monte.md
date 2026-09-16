---
title: "Efficient Bayesian Sampling Using Normalizing Flows to Assist Markov Chain Monte Carlo Methods"
entry_type: paper
source: "https://arxiv.org/abs/2107.08001"
upstream_list: "janosh/awesome-normalizing-flows"
year: "2021"
authors: "Gabrié, Rotskoff et al."
description: "Normalizing flows have potential in Bayesian statistics as a complementary or alternative method to MCMC for sampling posteriors. However, their training via reverse KL divergence may be inadequate for complex posteriors. This research proposes a new training approach utilizing direct KL divergence, which involves augmenting a local MCMC algorithm with a normalizing flow to enhance mixing rate and utilizing the resulting samples to train the flow. This method requires minimal prior knowledge of the posterior and can be applied for model validation and evidence estimation, offering a promising strategy for efficient posterior sampling."
---

# Efficient Bayesian Sampling Using Normalizing Flows to Assist Markov Chain Monte Carlo Methods

**Source**: [https://arxiv.org/abs/2107.08001](https://arxiv.org/abs/2107.08001)

**Year**: 2021

**Authors**: Gabrié, Rotskoff et al.

## Description

Normalizing flows have potential in Bayesian statistics as a complementary or alternative method to MCMC for sampling posteriors. However, their training via reverse KL divergence may be inadequate for complex posteriors. This research proposes a new training approach utilizing direct KL divergence, which involves augmenting a local MCMC algorithm with a normalizing flow to enhance mixing rate and utilizing the resulting samples to train the flow. This method requires minimal prior knowledge of the posterior and can be applied for model validation and evidence estimation, offering a promising strategy for efficient posterior sampling.
