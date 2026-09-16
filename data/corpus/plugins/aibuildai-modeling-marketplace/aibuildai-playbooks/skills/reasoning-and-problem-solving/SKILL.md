---
description: >-
  ML playbook for reasoning and problem solving competitions. Use when tackling a Kaggle-style competition involving reasoning and problem solving. Teaches how to reason about classify by problem structure to choose architecture, choose problem representation before model architecture, scale inference-time compute over training compute, use test-time fine-tuning when each test instance is novel. 45+ top-solution writeups across 15 competitions including ARC Prize 2024/2025, AI Mathematical Olympiad (3 iterations), Hungry Geese, Lux AI 2021, Conway's Reverse Game of Life, LLM Prompt Recovery, and Integer Sequence Learning.
---

# Reasoning and Problem-Solving Playbook

Reasoning and problem-solving competitions span abstract pattern recognition (ARC), mathematical olympiads, strategic games, inverse problems, and prompt recovery. Unlike domain-specific prediction tasks, these require discovering novel solutions, generalizing from few examples, or outthinking adversaries. The core challenge is combining search over solution spaces with learned representations, adapting to test-time constraints, and scaling inference-time computation.

**Source material:** 45+ top-solution writeups across 15 competitions including ARC Prize 2024/2025, AI Mathematical Olympiad (3 iterations), Hungry Geese, Lux AI 2021, Conway's Reverse Game of Life, LLM Prompt Recovery, and Integer Sequence Learning.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Classify by Problem Structure to Choose Architecture | All competitions - each type has distinct winning patterns | principles/01.md |
| 2 | Choose Problem Representation Before Model Architecture | 5/5 top ARC solutions, 4/4 top math solutions, 3/3 top Game of Life solutions | principles/02.md |
| 3 | Scale Inference-Time Compute Over Training Compute | 5/5 top math solutions, 4/4 top ARC solutions using test-time adaptation, 6/6 top game solutions using search | principles/03.md |
| 4 | Use Test-Time Fine-Tuning When Each Test Instance is Novel | 2/2 top ARC solutions (2024 and 2025) | principles/04.md |
| 5 | Generate Synthetic Training Data When Real Data is Scarce | 5/6 top math solutions, 3/3 top LLM prompt recovery solutions, 2/2 top ARC solutions | principles/05.md |
| 6 | Combine Neural Networks with Algorithmic Components for Verifiable Tasks | 5/5 top math solutions use code execution, 3/3 top Game of Life use SAT solvers, 6/6 top game-playing use MCTS | principles/06.md |
| 7 | Train Separate Verification or Reward Models for Weighted Voting | 4/5 top math solutions, 2/2 top ARC solutions attempted (with mixed success) | principles/07.md |
| 8 | Apply Aggressive Data Augmentation for Geometric and Permutation Invariances | 5/5 top ARC solutions, 6/6 top game-playing solutions, 2/2 top Game of Life solutions | principles/08.md |
| 9 | For Multi-Agent Games, Bootstrap from Strong Agents Then Self-Play | 4/4 top Hungry Geese solutions, 3/3 top Lux AI solutions using neural nets | principles/09.md |
| 10 | Use Constraint Solvers for Inverse Problems with Deterministic Rules | 3/3 top Game of Life solutions, 2/2 top ciphertext challenge solutions | principles/10.md |
| 11 | For Opponent Modeling, Learn Unpredictability Rather Than Predicting Perfectly | 3/5 top Hungry Geese solutions, 2/3 top Lux AI solutions | principles/11.md |
| 12 | Multi-Task Training for Shared Representation Learning | 2/2 top ARC solutions using Omni-ARC approach | principles/12.md |
| 13 | Optimize Mean/Template Solutions for Underspecified Problems | 4/4 top LLM Prompt Recovery solutions | principles/13.md |
| 14 | Balance Exploration and Exploitation in Search via Temperature and Penalties | 5/6 top game-playing solutions using MCTS, 3/5 top math solutions using sampling | principles/14.md |
| 15 | Validate on Held-Out Test-Like Data, Not Just Training Distribution | 4/5 top math solutions created custom validation sets, 2/2 top ARC solutions validated on evaluation set | principles/15.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles compose: start by classifying your problem structure to choose the right architecture family, then apply test-time adaptation if test instances are novel, scale inference compute via sampling and search, and use verification/voting to select the best candidate. The key insight is that reasoning problems reward inference-time computation more than training-time computation.
