---
title: "ERRA: An Embodied Representation and Reasoning Architecture for Long-horizon Language-conditioned Manipulation Tasks"
entry_type: paper
source: "http://arxiv.org/pdf/2304.02251v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Chao Zhao,  Shuai Yuan,  Chunli Jiang,  Junhao Cai,  Hongyu Yu,  M. Y. Wang,  Qi-Feng Chen"
description: "This letter introduces ERRA, an embodied learning architecture that enables robots to jointly obtain three fundamental capabilities (reasoning, planning, and interaction) for solving long-horizon language-conditioned manipulation tasks. ERRA is based on tightly-coupled probabilistic inferences at two granularity levels. Coarse-resolution inference is formulated as sequence generation through a large language model, which infers action language from natural language instruction and environment state. The robot then zooms to the fine-resolution inference part to perform the concrete action corresponding to the action language. Fine-resolution inference is constructed as a Markov decision process, which takes action language and environmental sensing as observations and outputs the action. The results of action execution in environments provide feedback for subsequent coarse-resolution reasoning. Such coarse-to-fine inference allows the robot to decompose and achieve long-horizon tasks interactively. In extensive experiments, we show that ERRA can complete various long-horizon manipulation tasks specified by abstract language instructions. We also demonstrate successful generalization to the novel but similar natural language instructions."
---

# ERRA: An Embodied Representation and Reasoning Architecture for Long-horizon Language-conditioned Manipulation Tasks

**Source**: [http://arxiv.org/pdf/2304.02251v1](http://arxiv.org/pdf/2304.02251v1)

**Year**: 2023

**Authors**: Chao Zhao,  Shuai Yuan,  Chunli Jiang,  Junhao Cai,  Hongyu Yu,  M. Y. Wang,  Qi-Feng Chen

## Description

This letter introduces ERRA, an embodied learning architecture that enables robots to jointly obtain three fundamental capabilities (reasoning, planning, and interaction) for solving long-horizon language-conditioned manipulation tasks. ERRA is based on tightly-coupled probabilistic inferences at two granularity levels. Coarse-resolution inference is formulated as sequence generation through a large language model, which infers action language from natural language instruction and environment state. The robot then zooms to the fine-resolution inference part to perform the concrete action corresponding to the action language. Fine-resolution inference is constructed as a Markov decision process, which takes action language and environmental sensing as observations and outputs the action. The results of action execution in environments provide feedback for subsequent coarse-resolution reasoning. Such coarse-to-fine inference allows the robot to decompose and achieve long-horizon tasks interactively. In extensive experiments, we show that ERRA can complete various long-horizon manipulation tasks specified by abstract language instructions. We also demonstrate successful generalization to the novel but similar natural language instructions.
