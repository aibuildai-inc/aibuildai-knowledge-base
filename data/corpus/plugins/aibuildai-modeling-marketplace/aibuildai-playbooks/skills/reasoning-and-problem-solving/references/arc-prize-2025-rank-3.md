# MindsAI & Tufa Labs – ARC Prize 2025 Solution

Competition: arc-prize-2025
Rank: #3
Source: https://www.kaggle.com/c/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution

# MindsAI & Tufa Labs – ARC Prize 2025 (3rd Place)

Please see the full write-up with ablations: https://github.com/jcole75/arc_2025_mindsai/blob/main/MindsAI_Tufa_Labs_2025_Solution.pdf

## A1. Background on You / Your Team
**Competition:** ARC Prize 2025  
**Team Name:** MindsAI & Tufa Labs  
**Private Leaderboard Score:** 15.42%  
**Private Leaderboard Place:** 3rd

**Team Members:**  
- Jack Cole – Olney, Illinois, USA – jackcole@mindware.mobi  
- Dries Smit – Somerset West, South Africa – dries.epos@gmail.com  
- Isaiah Pressman – Cleveland Heights, Ohio, USA – isaiahpressman16@gmail.com  
- Mohamed Osman – Calgary, Alberta, Canada – mothman198@outlook.com  
- Michael Hodel – Switzerland – hodelmichi@gmail.com  

---

## A2. Background on You / Your Team (Full Bios)

### Jack Cole (team lead & primary contributor)
- **Academic/Professional Background:** PhD in Clinical Psychology; part-time private practice psychologist; parallel career as an AI researcher and mobile app developer (author of Mind Games – 30M+ downloads).  
- **Prior Experience:** Led the 1st-place team in ARCathon 2023; held state-of-the-art scores on ARC-AGI-1 through 2024; pioneered Test-Time Training (TTT) and AIRV techniques that are now used by nearly all top ARC solutions.  
- **Why ARC Prize:** ARC has been the central focus of my research program since mid-2022.  
- **Time Spent:** Approximately 3.5 years of near full-time work on ARC (minus ~6 months off at the end of 2024 / early 2025); hundreds of experiments and training runs lasting up to 2.5 years on TPUs.  
- **2025 Role:** Designed, trained, and submitted the solution; code, datasets, ablations, and the final submission were produced by me.  

### Dries Smit
- **Academic/Professional Background:** PhD in Electrical & Electronic Engineering; specialist in reinforcement learning, multi-agent systems, distributed training, and large language models.  
- **Prior Experience:** Led the winning solution in the ARC-AGI-3 Preview competition; developed Laila (biology-assistant fine-tune of Llama 3.1); extensive work scaling RL and reasoning systems.  
- **Why ARC Prize:** ARC is a core reasoning benchmark that aligns perfectly with research on adaptive, test-time reasoning systems.  
- **Time Spent:** Part-time contributions through part of the season.  
- **2025 Role:** Provided refinement-based training experiments and strategic insights; some of those ideas were explored but ultimately not additive with the TTT/AIRV core.  

### Isaiah Pressman
- **Academic/Professional Background:** AI researcher/engineer since 2019; previous work in computer vision for histopathology, data/ML pipelines for startups, four top-2 Kaggle finishes.  
- **Prior Experience:** Deep experience improving LLM training and reasoning capabilities at Tufa Labs.  
- **Why ARC Prize:** Fascination with pure reasoning challenges that are hard for current foundation models.  
- **Time Spent:** ~20 hours/week for one month, then full-time for another month toward the end of the competition.  
- **2025 Role:** Explored a promising diffusion-based language modeling approach for ARC (not included in the final submission but deepened architectural understanding).  

### Mohamed Osman
- **Academic/Professional Background:** Master’s in Electrical Engineering; 5+ years as an ML practitioner and researcher.  
- **Prior Experience:** Co-developed early versions of TTT for ARC in previous seasons; strong ML engineering background.  
- **Why ARC Prize:** Strong agreement with the ARC Prize definition of intelligence and interest in the core problem.  
- **Time Spent:** Very limited in 2025 due to external commitments.  
- **2025 Role:** Minimal direct involvement this year.  

### Michael Hodel
- **Academic/Professional Background:** AI researcher; creator of RE-ARC and ARC-DSL.
- **Prior Experience:** Machine learning experience, Prior work on ARC. 
- **Why ARC Prize:** Long-standing passion for the ARC challenge and building tools/datasets for the community.  
- **Time Spent:** Limited in 2025.  
- **2025 Role:** Provided foundational datasets (ARC 1.5, RE-ARC variants) that were used for ablations and portions of training data.  

---

## A3. Summary

Our 3rd-place solution is built around a pruned 660M-parameter encoder–decoder model derived from Salesforce CodeT5-Large, trained for many months on >100 million reasoning examples (~70M of them ARC-style tasks). Almost all submitted performance comes from heavy test-time adaptation: Test-Time Training (TTT) with permutation-based labeling (~45k steps) and Augment–Inference–Reverse–Vote (AIRV) using 10k augmented inferences per task — techniques introduced by our team in 2023 and now often used across the leaderboard. These two methods combine almost perfectly additively, delivering 8–12× gains over zero-shot in ablations. Additional improvements come from new mixup/combine augmentations, reversal augmentations, tokenizer BPE dropout, T5 span-corruption, and ensembling two checkpoints. Full code, models, and the complete 100M+ example training corpus are publicly released.

---

## A4. Feature Selection / Engineering

No traditional features — performance is driven entirely by data augmentations (the ARC equivalent of feature engineering):

| Rank | Augmentation                                    | Impact |
|------|--------------------------------------------------|--------|
| 1    | Geometric (rotations/flips) + color permutations| Baseline essential |
| 2    | New 2025: Mixup, Combine, Combine-mixup         | +6.3% top-2 on ARC 1.5 |
| 3    | Input/output swap (30% of training)             | Small score improvement |
| 4    | Prompt/answer reversals (training only)         | Model flexibility |
| 5    | BPE tokenizer dropout (TTT & inference)        | Small score improvement |

---

## A5. Training Method(s)

- 660M CodeT5-Large → encoder kept at 24 layers, decoder pruned to 16 layers  
- Supervised training on >100M examples (Google TPUs, some runs >2 years cumulative)  
- T5 span-corruption + reversal augmentations + BPE dropout  
- Test-time: TTT (~45k examples) + AIRV (10k augmentations) + self-ensembling two checkpoints  

Ensembling: combining of predictions from two strong checkpoints of the same architecture.

---

## A6. Interesting Findings

- TTT + AIRV gains are nearly perfectly additive (~812% combined vs. ~410–430% each alone).  
- Self-ensembling with different seeds beats using 2× more TTT/AIRV samples (compute-matched +6.2%).  
- Encoder depth >> decoder depth (removing encoder layers hurts badly; decoder can be heavily pruned).  
- Augmentation-driven data expansion can break multi-month training plateaus.  
- Several otherwise strong methods (refinement, DPO on beam pairs, targeted ARC-2 data) were non-additive with TTT+AIRV.  
- ARC-AGI-2 appears partially adversarial to the current TTT/AIRV paradigm (at least with models this small) — new ideas are needed.

---

## A7. Simple Features and Methods

A dramatically simplified yet still strong version (90–95% of the full relative gain -- linked in resources):

- Only classic geometric + color permutation augmentations  
- TTT with 20–30k examples + AIRV with 5–10k samples (single run, no ensembling, no BPE dropout)  

This runs in minutes to a few hours on a single consumer GPU and is ideal for research iteration.

---

## A8. Model Execution Time

| Component                             | Hardware                | Approximate Time                  |
|---------------------------------------|-------------------------|-----------------------------------|
| Full 660M training                    | Google TPUs             | Many months to 2.5 years cumulative |
| 77M ablation model training           | TPU v2-8 / v3-8 + v4-64 | ~2 years + 7 days                 |
| Final submission inference (2× ensemble) | 4× L4 GPUs (Kaggle)   | ~11 hours                         |
| Simplified 77M single-run inference   | Single P100        | 10–60 minutes                     |

---

## A9. References & Public Resources

- Complete code, training scripts, models, and 100M+ example dataset:  
  → **https://github.com/jcole75/arc_2025_mindsai**  
  → Full Write-up with Ablations: https://github.com/jcole75/arc_2025_mindsai/blob/main/MindsAI_Tufa_Labs_2025_Solution.pdf
  → Dataset: https://huggingface.co/datasets/mindware/arc-agi-mega  
  → Models: https://huggingface.co/mindware  

- Cole & Osman (2025). Don’t throw the baby out with the bathwater: How and why deep learning for ARC. arXiv:2506.14276  (https://arxiv.org/abs/2506.14276)
- Hodel (2024). RE-ARC procedural generation. arXiv:2404.07353  

Everything is released under permissive licenses to help the community push toward AGI.
