# 10th Place Solution – Boey – End-to-End JAX RL

Competition: lux-ai-season-3
Rank: #10
Source: https://www.kaggle.com/c/lux-ai-season-3/discussion/570196

First of all, I’d like to thank the organizers and the Lux AI team for designing such a performant and engaging environment for Lux S3. This was both my first Kaggle competition and my first reinforcement learning competition, and I’m incredibly grateful for the opportunity to apply and test my RL knowledge in such a competitive setting.

# Overview of My Approach
My solution is built on top of [PureJaxRL][1] (a CleanRL-like minimal PPO implementation in JAX) which I extended into a fully end-to-end JAX reinforcement learning pipeline. Every component, from observation preprocessing to the training loop, is fully JIT-compilable, enabling high-speed training performance.

### Key Components:
1. End-to-End JAX RL – Fully jittable pipeline.
2. PPO with Backpropagation Through Time (BPTT)
3. Multi-agent Learning with Prioritized Fictitious Self-Play (PFSP)

### Performance Benchmarks:
- Training throughput: 80,000 steps/sec
- Model size: ~1.8M parameters, (+1.4M for separate Critic network)

---

# Input Features
### Unit Features (Per 32 units):
- Ally/enemy flag
- Visibility
- Last 5 turns energy
- Turns since last seen
- Current position/Last known position

### Spatial Features (24×24 grid):
- Energy field
- Nebula & Asteroid fields
- Relic nodes & points
- Sensor mask
- Visited tiles

*Spatial information persists across multiple steps.
*Nebula/asteroid tiles dynamically shift based on drift speed.
*Most spatial features are symmetrical across the map (excluding the sensor mask).

### Scalar Features:
- Binary encoded team points and wins
- Binary encoded Match steps
- Hidden game parameters (only included if deducible)

---

# Action Space
| Component	 | Description | Masking |
| --- | --- | --- |
| Action Type | noop, move_up, move_down, move_left, move_right, sap | Insufficient energy, map bounds, asteroid collision |
| SAP Head | 15×15 = 225 possible sap targets (only used when sap action is chosen) | Only allow targeting around visible enemy units, and known relic points |

---

# Reward Engineering
| Phase | Rewards |
| --- | --- |
| Early to Mid Training | +0.002 per point gain, +1 per match win |
| Late Training (Sparse) | +1 per match win only |

All rewards are zero-sum. I only switched to sparse rewards in the final week of the competition, so both submissions used sparse reward training for just the last 10% of their total training steps. Despite this limited use, the switch still led to significant performance gains.

---

# Network Architecture

This is the final network architecture I used for my submission. It’s similar to my initial design and relatively lightweight at ~1.8M trainable parameters. While I experimented with larger models, they consumed more VRAM, reducing the number of parallel environments and slowing down training without significant performance gains during mid-stage training.

### Experiments Tried
- Replace Conv2d with ResBlocks.
- Add 3x ResBlocks upsampling for Sap to create 15x15 Sap map.
- Using a Pointer Network to select SAP targets

## Centralized Critic (Full Observability)
For my second submission, I used a centralized critic with access to global information. The policy and critic used separate input pipelines and were trained via two independent networks (+1.4M parameters).
- Pros:
    - Improved value estimation
    - Approximately 2x better sample efficiency
- Cons:
    - Slower training (drops to ~40k SPS)
    - Wall-time performance gain was neutral

## Prioritized Fictitious Self-Play (PFSP)  
I implemented Prioritized Fictitious Self-Play (PFSP), inspired by [AlphaStar][2]'s methodology, to enhance the training efficiency and robustness of my reinforcement learning agents. Unlike standard Fictitious Self-Play (FSP), where opponents are sampled uniformly from past versions, PFSP assigns higher selection probabilities to opponents based on their win rates against the current agent. This targeted sampling ensures that the agent focuses more on challenging opponents, thereby promoting continuous improvement and avoiding stagnation.
- 75% games: Self-play
- 25% games: Frozen past versions
- PFSP matches more frequently with opponents the agent struggles against (tracked via win rate)

## Training Setup
I ran most of my experiment on my local machine, AMD Ryzen 7950x, 64gb RAM and Nvidia RTX 4090. For hyperparameter tuning, I rented 2-4 cloud GPUs (RTX 4090) over roughly 10 days.

## Challenges
While building a fully end-to-end JAX pipeline provided excellent training performance, it significantly reduced code readability, making it harder to spot bugs. I only discovered a few crucial bugs, ones that were seriously hindering the agent’s performance about two-thirds into the competition. In hindsight, I should have included extensive unit tests from the start, as they were essential for ensuring correctness and stability.

## Final submission
| Submission | Setup | Env Steps Trained | Wall time | Final Leaderboard Rating | 
| --- | --- | --- | --- | --- |
| #1 | Shared network | ~58B | 8 days | 1771.5 |
| #2 | Centralized Critic | ~23B | 7 days | 1884.5 |

---

# Closing Thoughts
I’m incredibly proud to have placed in the Top 10 in my very first Kaggle competition.

Looking back, I believe that opting for a higher-capacity model (10M+ parameters) rather than maximizing training throughput would have yielded better results. My final submission, which used a centralized critic with ~3.2M parameters, achieved a significantly higher rating than the earlier 1.8M model.

[1]: https://github.com/luchris429/purejaxrl
[2]: https://www.nature.com/articles/s41586-019-1724-z
