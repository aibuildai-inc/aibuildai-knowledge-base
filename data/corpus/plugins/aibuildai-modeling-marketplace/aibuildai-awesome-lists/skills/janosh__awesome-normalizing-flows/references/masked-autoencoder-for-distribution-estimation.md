---
title: "Masked Autoencoder for Distribution Estimation"
entry_type: paper
source: "https://arxiv.org/abs/1502.03509"
upstream_list: "janosh/awesome-normalizing-flows"
year: "2015"
authors: "Germain, Gregor et al."
description: "Introduces MADE, a feed-forward network that uses carefully constructed binary masks on its weights to control the precise flow of information through the network. The masks ensure that each output unit receives signals only from input units that come before it in some arbitrary order. Yet all outputs can be computed in a single pass.    A popular and efficient way to make flows autoregressive is to construct them from MADE nets.    <a href=\"https://github.com/janosh/diagrams/tree/main/assets/made\">      <picture>        <source media=\"(prefers-color-scheme: dark)\" srcset=\"https://raw.githubusercontent.com/janosh/diagrams/main/assets/made/made-white.svg\">        <img alt=\"Masked Autoencoder for Distribution Estimation\" src=\"https://raw.githubusercontent.com/janosh/diagrams/main/assets/made/made.svg\">      </picture>    </a>"
---

# Masked Autoencoder for Distribution Estimation

**Source**: [https://arxiv.org/abs/1502.03509](https://arxiv.org/abs/1502.03509)

**Year**: 2015

**Authors**: Germain, Gregor et al.

## Description

Introduces MADE, a feed-forward network that uses carefully constructed binary masks on its weights to control the precise flow of information through the network. The masks ensure that each output unit receives signals only from input units that come before it in some arbitrary order. Yet all outputs can be computed in a single pass.
   A popular and efficient way to make flows autoregressive is to construct them from MADE nets.
   <a href="https://github.com/janosh/diagrams/tree/main/assets/made">
     <picture>
       <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/janosh/diagrams/main/assets/made/made-white.svg">
       <img alt="Masked Autoencoder for Distribution Estimation" src="https://raw.githubusercontent.com/janosh/diagrams/main/assets/made/made.svg">
     </picture>
   </a>
