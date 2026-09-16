---
title: "Masked Autoregressive Flow for Density Estimation"
entry_type: paper
source: "https://arxiv.org/abs/1705.07057"
upstream_list: "janosh/awesome-normalizing-flows"
year: "2017"
authors: "Papamakarios, Pavlakou et al."
description: "Introduces MAF, a stack of autoregressive models forming a normalizing flow suitable for fast density estimation but slow at sampling. Analogous to Inverse Autoregressive Flow (IAF) except the forward and inverse passes are exchanged. Generalization of RNVP.    <a href=\"https://github.com/janosh/diagrams/tree/main/assets/masked-autoregressive-flow\">      <picture>        <source media=\"(prefers-color-scheme: dark)\" srcset=\"https://raw.githubusercontent.com/janosh/diagrams/main/assets/masked-autoregressive-flow/masked-autoregressive-flow-white.svg\">        <img alt=\"Diagram of the slow (sequential) forward pass of a Masked Autoregressive Flow (MAF) layer\" src=\"https://raw.githubusercontent.com/janosh/diagrams/main/assets/masked-autoregressive-flow/masked-autoregressive-flow.svg\">      </picture>    </a>"
---

# Masked Autoregressive Flow for Density Estimation

**Source**: [https://arxiv.org/abs/1705.07057](https://arxiv.org/abs/1705.07057)

**Year**: 2017

**Authors**: Papamakarios, Pavlakou et al.

## Description

Introduces MAF, a stack of autoregressive models forming a normalizing flow suitable for fast density estimation but slow at sampling. Analogous to Inverse Autoregressive Flow (IAF) except the forward and inverse passes are exchanged. Generalization of RNVP.
   <a href="https://github.com/janosh/diagrams/tree/main/assets/masked-autoregressive-flow">
     <picture>
       <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/janosh/diagrams/main/assets/masked-autoregressive-flow/masked-autoregressive-flow-white.svg">
       <img alt="Diagram of the slow (sequential) forward pass of a Masked Autoregressive Flow (MAF) layer" src="https://raw.githubusercontent.com/janosh/diagrams/main/assets/masked-autoregressive-flow/masked-autoregressive-flow.svg">
     </picture>
   </a>
