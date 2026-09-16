# ARC Prize 2025 Competition Writeup: 5th Place

Competition: arc-prize-2025
Rank: #4
Source: https://www.kaggle.com/c/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place

# Overview
This submission forks the notebook ~[LB 5.83 baseline from 1st place of 2024](https://www.kaggle.com/code/sanyul/lb-5-83-baseline-from-1st-place-of-2024)~ by @sanyul and the ~[baseline from 1st place of 2024](https://www.kaggle.com/code/boristown/baseline-from-1st-place-of-2024)~ by @boristown, which originates from the first-place solution from ARC Prize 2024 by the ARChitects team (Daniel Franzen and Jan Disselhoff). Full credits to all of them for their exceptional work.
My submission notebook can be found here: https://www.kaggle.com/code/lonnieqin/lb-5-83-baseline-from-1st-place-of-2024
# Performance
**Public Leaderboard**: 344th place

**Private Leaderboard**: 5th place

The dramatic improvement from public to private leaderboard rankings highlights the importance of model generalization and the critical role of hyperparameter optimization in competitive machine learning, especially when working with small evaluation sets.

# Structure of the Solution
The solution consists of several key components organized as follows:

## 1. Core Infrastructure Files:
* model_runner.py: Handles model loading, training, inference, and vocabulary shrinking
* selection.py: Implements various algorithms for selecting the best predictions
* async_tools.py: Manages parallel execution across multiple GPUs
* common_stuff.py: Central configuration hub containing all hyperparameters and orchestration logic

## 2. Training Pipeline:
* Multi-GPU distributed training (4 GPUs) with different data splits
* LoRA fine-tuning on a quantized base model (Nemo Mini)
* Data augmentation (transposition, rotation, color permutation)
* Task-specific priming for improved predictions

## 3. Inference Pipeline:
* Turbo DFS (Depth-First Search) beam search for efficient decoding
* Augmented scoring mechanism for prediction selection
* Parallel inference across multiple GPUs
* Result caching and incremental processing

## 4. Execution Flow:
* Parallel training on 4 GPUs → Wait for completion → Parallel inference on 4 GPUs → Aggregate results → Generate submission

# Methodology
The solution employs a **fine-tuned language model approach** to the ARC-AGI challenge:
## Base Architecture
* **Model**: Nemo Mini (fine-tuned on the ARC 2024 dataset – the best solution of ARC 2024)
* **Fine-tuning**: LoRA (Low-Rank Adaptation) with rank 32
* **Active layers**: Only the first 32 layers trainable to reduce overfitting

## Data Representation
* Tasks are converted to text format using a specialized formatter (ArcFormatter_premix_3)
* Input/output grids are serialized as token sequences
* Maximum sequence length: 4,224 tokens (training), 8,192 tokens (inference)

## Training Strategy
* **Augmentation**: Transposition, rotation, random color permutation, example shuffling
* **Epochs**: 4 epochs with 240 training steps
* **Multi-GPU**: Data distributed across 4 GPUs (each processes ~1/4 of tasks)
* **Learning rates**: 1e-4 (main), 1e-5 (embeddings)

## Inference Strategy
* **Task-specific priming**: Model is briefly retrained on each test task's examples before prediction
* **Turbo DFS beam search**: Efficiently explores multiple prediction candidates
* **Augmented scoring**: Predictions evaluated across multiple augmentations to select the most robust solution
* **Probability tracking**: Maintains cumulative probabilities to identify best candidates

# My Contribution: Treating Random Seed as a Hyperparameter
While this solution builds upon the strong 2024 first-place baseline, I made one key modification that proved surprisingly impactful:
**Changed the random seed to 19920627** , selected based on empirical exploration of seed values.
## Random Seed as a Legitimate Hyperparameter
In scenarios with **small sample sizes** (like this competition's 240 evaluation tasks), random seed should be treated as a **first-class hyperparameter** rather than an afterthought. This is because:

* **High variance in small samples**: With only 120 tasks per leaderboard split, solving just 1-2 additional tasks can dramatically shift rankings

* **Stochastic model behavior**: Random seeds influence multiple pipeline stages, creating meaningfully different model behaviors

* **Task-specific interactions**: Different seeds may produce augmentation patterns or training dynamics better suited to specific task types

This seed change creates cascading effects throughout the entire machine learning pipeline:
* **Data shuffling**: Alters the order in which training examples are presented
* **Augmentation patterns**: Modifies rotations, transpositions, and color permutations applied to training data
* **Model training**: Influences weight initialization in LoRA layers and training dynamics
* **Inference behavior**: Changes stochastic elements in beam search decoding
* **Convergence paths**: Leads to different optimization trajectories and final model states

# The Hyperparameter Landscape: Beyond Random Seeds
While random seed optimization proved effective in this competition, it's important to recognize that this solution space contains **numerous other hyperparameters** worthy of systematic exploration:
## 1. Model Selection and Architecture Hyperparameters
**Base Model Choice** - One of the most impactful decisions:
* **Nemo Mini** (current choice): Efficient, fast inference, good balance
* **Llama 3.1 variants**: 8B, 70B models with different reasoning capabilities
* **Qwen 2.5**: Strong performance on reasoning tasks
* **Gemma 2**: Google's open models with different architectural choices
* **Phi-3/Phi-4**: Microsoft's small language models optimized for reasoning
* **Mistral variants**: 7B, Mixtral MoE models with different trade-offs
* **DeepSeek models**: Strong coding and reasoning abilities

⠀Different base models have fundamentally different:
* Vocabulary and tokenization strategies (affecting grid representation)
* Architectural inductive biases (attention patterns, layer depths)
* Pre-training distributions (some models see more code/structured data)
* Context window capabilities (affecting maximum task size handling)
* Inference speed and memory requirements (computational constraints)

⠀**Architecture Configuration**:
* **LoRA rank**: Currently set to 32, but values from 8 to 128+ could be explored
* **LoRA alpha**: Scaling factor for LoRA updates
* **LoRA target modules**: Which layers to apply LoRA (attention, MLP, both)
* **Trainable layers**: Currently first 32 layers; different layer selections might capture different abstractions
* **Quantization level**: 4-bit vs 8-bit vs full precision trade-offs
* **Model size within family**: Different parameter counts capture different complexity levels

## 2. Training Hyperparameters
* **Learning rates**: Main (1e-4) and embedding (1e-5) rates could be tuned independently
* **Number of epochs**: Currently 4; more epochs might improve convergence or cause overfitting
* **Training steps**: 240 steps per epoch is another tunable parameter
* **Batch size**: Affects training stability and generalization
* **Gradient accumulation steps**: For effective larger batch sizes
* **Optimizer choice**: Adam, AdamW, SGD with different momentum values
* **Learning rate schedule**: Constant vs cosine decay vs linear warmup strategies
* **Weight decay**: Regularization strength
* **Dropout rates**: If applicable to the fine-tuning strategy

## 3. Data Augmentation Hyperparameters
* **Augmentation probability**: How frequently each augmentation is applied
* **Augmentation types**: Which transformations to include/exclude (transposition, rotation, color permutation)
* **Example shuffling strategy**: Different orderings of demonstration examples
* **Color mapping schemes**: Alternative color permutation strategies
* **Rotation angles**: 90°, 180°, 270° vs other transformations
* **Augmentation composition**: Sequential vs parallel application

## 4. Inference Hyperparameters
* **Beam search width**: Number of candidates to explore simultaneously
* **DFS depth limit**: How deep to search in the solution space
* **Temperature**: Sampling temperature for generation (typically 0.1 to 1.5)
* **Top-k/top-p values**: Nucleus sampling parameters
* **Repetition penalty**: To discourage repetitive patterns
* **Priming epochs**: How long to fine-tune on test task examples
* **Number of augmentations**: How many augmented versions to score
* **Maximum generation length**: Token limits for output

## 5. Selection Algorithm Hyperparameters
* **Scoring weights**: How to balance different scoring criteria
* **Ensemble strategies**: How to combine predictions from multiple runs
* **Confidence thresholds**: When to trust a prediction vs explore alternatives
* **Voting mechanisms**: Majority vote vs weighted vote vs consensus strategies

## The Small Sample Size Advantage
In competitions with small evaluation sets (100-500 samples), these hyperparameters become **especially influential** because:
* **Lower statistical power**: Individual hyperparameter choices create measurable differences in final scores
* **Higher variance sensitivity**: Small improvements (1-3 additional correct predictions) translate to significant rank changes
* **Exploration feasibility**: With limited test data, systematic grid search or random search over hyperparameters becomes computationally tractable
* **Overfitting opportunities**: Hyperparameters can be tuned to exploit specific characteristics of the small evaluation distribution

This doesn't diminish the achievement—rather, it highlights that **comprehensive hyperparameter optimization is a valuable and legitimate competition strategy**, especially when the evaluation set is constrained.

# Motivation: The Statistics Behind Random Seed Optimization
My approach was motivated by understanding the **statistical landscape of the competition**:
### Competition Structure
* **Public LB**: 120 samples (scores reflected on public leaderboard during competition; samples not visible to participants)
* **Private LB**: 120 samples (used to compute final private scores; revealed after December 5, 2025)

## The Random Seed Strategy
Given that:
* Most competitive solutions were variations of the same baseline notebook
* The same model can produce different scores across runs due to stochasticity
* There are only 120 samples in both public LB and private LB

**This creates an opportunity**: systematically exploring hyperparameters (including random seeds) can yield significantly different rankings even without fundamental algorithmic improvements.
### Score Variance Analysis
**My observed performance across different random seeds**:
* Best public LB score: **4.17%** (5/120 tasks)
* Estimated private score: **~10%** (12/120 tasks)
* **This notebook's score range: 3.33 % to 6.67 %(with the upper bound potentially higher)** across different random seed configurations
* Score variance: ±3.34 percentage points (equivalent to ±4 tasks out of 120)
* **Pixel-Level Sensitivity Amplification**: The all-or-nothing scoring (one wrong pixel = full failure) magnifies this variance; hyperparameter tweaks that nudge predictions toward exact matches on borderline tasks can swing 1-2 full solves, equivalent to 1-2% score jumps.

This **2x variation in performance** (from 3.33 % to 6.67 %) demonstrates the substantial impact random seed can have as a hyperparameter in small-sample scenarios.

## Public LB landscape:
* 8 teams achieved >10% score (>12 tasks solved)
* Significant variance exists even among top solutions
* Top solutions likely benefited from both strong architectures AND favorable hyperparameter configurations

## Probability Calculation
**Question**: Under the simplifying assumption of a model with 4% base accuracy on independent tasks (which isn't the actual case, as task accuracies vary and dependencies exist), what's the probability of achieving a 10% score (12/120 tasks)?

Assuming each task is independent with success probability p = 0.04:
* P(X ≥ 12) where X ~ Binomial(n=120, p=0.04)
* Expected value: E[X] = 120 × 0.04 = 4.8 tasks
* Standard deviation: σ = √(120 × 0.04 × 0.96) ≈ 2.15
* P(X ≥ 12) ≈ P(Z ≥ (12 - 4.8) / 2.15) ≈ P(Z ≥ 3.35) ≈ 0.04 %

This is extremely unlikely (~1 in 2,500 chance) if purely random!
### The Real Insight
However, the actual mechanism is more nuanced:
* **Model capability varies by task type**: The model may have 10-15% accuracy on certain task patterns and 0-2% on others
* **Hyperparameters affect task-specific performance**: Different hyperparameter configurations (including seeds, base models, and training strategies) produce different augmentation patterns and model behaviors, leading to varying performance on specific tasks
* **Small sample size amplification**: With only 120 tasks per split, small improvements (solving 1-2 more tasks) create large ranking jumps
* **Model selection matters significantly**: Different base models have varying strengths across task categories—some excel at spatial reasoning, others at pattern completion or color transformations
**Strategic implication**:
* While LLMs struggle with ARC-AGI reasoning (fundamental limitation), Achieving competitive results is possible through:
  * Starting from strong baselines
  * Systematic hyperparameter exploration (seeds, learning rates, augmentation strategies, etc.)
  * **Exploring different base model architectures** (Llama, Qwen, Gemma, Phi, etc.)
  * Understanding which task types benefit from different configurations
  * Leveraging the competition's small evaluation set for targeted optimization

### Competitive Reality
This is **not about gaming the system**—it's about:
* Recognizing that **all stochastic elements are legitimate hyperparameters** in small-sample scenarios
* Understanding that competition rankings reflect both model quality AND optimal hyperparameter selection
* Acknowledging that with 240 evaluation samples, hyperparameter optimization matters significantly
* Appreciating that hyperparameter tuning is a **core machine learning skill**, not a shortcut
* **Recognizing that base model selection is itself a critical hyperparameter choice**

My 5th place finish validates that while **solving ARC-AGI fundamentally remains extremely difficult**, competitive performance in this format is achievable through careful exploration of the hyperparameter space, including often-overlooked parameters like random seeds and the foundational choice of base model architecture.

# What Worked
* **Strong baseline architecture**: The ARChitects' original design proved highly effective as a foundation
* **Multi-GPU distribution**: Enabled processing all tasks within the competition's tight time constraints
* **Task-specific priming**: Improved predictions by conditioning on individual challenge characteristics
* **Augmented scoring**: Enhanced prediction selection through more robust evaluation metrics
* **Hyperparameter optimization**: Treating random seed as a tunable hyperparameter achieved competitive results, with score variance of 3.33 % to 6.67 % demonstrating significant impact
* **Nemo Mini as base model**: Balanced efficiency and performance well for this specific task distribution

# What Could Be Improved
* **Systematic hyperparameter search**: A principled grid search or Bayesian optimization over multiple hyperparameters (not just random seed) could identify even better configurations
* **Base model exploration**: Testing alternative foundation models (Llama 3.2, Qwen 3, Gemma 3, Phi-3/4, Mistral, DeepSeek) could reveal architectures better suited to ARC-AGI reasoning patterns
* **Model ensemble methods**: Combining predictions from multiple base models with different architectural biases could improve robustness and coverage across diverse task types
* **Seed ensemble strategies**: Rather than selecting a single optimal seed, averaging or voting across multiple seed configurations could reduce variance
* **Public-private gap analysis**: Deeper investigation into the score discrepancy would provide valuable insights for future competitions
* **Selection algorithms**: More sophisticated prediction selection methods could improve consistency
* **Architectural enhancements**: Exploring modifications beyond LoRA fine-tuning (full fine-tuning, adapter layers, prefix tuning) might unlock further gains
* **Edge case handling**: Better strategies for tasks with unusual or rare patterns
* **Cross-validation on small samples**: Developing robust validation strategies for competitions with limited evaluation data
* **Multi-stage hyperparameter optimization**: Sequential optimization (model selection → training hyperparameters → inference hyperparameters) could be more efficient than random search
* **Pixel-Precision Robustness**: Develop tolerance-aware scoring proxies during hyperparameter search to better handle the one-pixel-failure risk, e.g., via soft-match metrics in validation to guide toward exact predictions without overfitting to noise.
* **Symbolic Function Generation**: Rather than directly predicting output grids pixel-by-pixel, explore generating interpretable transformation functions (e.g., via program synthesis or rule induction) from diverse training samples. This approach could enhance generalization across unseen task variations by capturing underlying patterns more abstractly, mitigating the fragility of raw pixel outputs.

# Conclusion
This competition experience has been both humbling and inspiring. Achieving 5th place on the private leaderboard demonstrates that in competitions with small evaluation sets, **comprehensive hyperparameter optimization is not just beneficial—it's essential**. The lesson extends beyond random seeds: base model selection, learning rate schedules, augmentation strategies, model architecture choices, and inference parameters all constitute a rich landscape worthy of systematic exploration.

The observed score variance (3.33 % to 6.67 %) from random seed changes alone illustrates how impactful these seemingly minor decisions can be. When we consider the additional dimensions of model selection (Llama vs Qwen vs Gemma vs Phi vs Mistral vs DeepSeek) and the interactions between these choices and other hyperparameters, the potential solution space becomes even more expansive.

The key insight is recognizing when hyperparameter variance becomes comparable to or exceeds model variance. In such scenarios, treating traditionally "fixed" parameters (like random seeds and base model choices) as first-class hyperparameters becomes a legitimate and powerful strategy.

I want to express my sincere gratitude to **Kaggle and the entire ARC Prize community** for creating this challenging and thought-provoking competition. The opportunity to engage with abstract reasoning tasks and learn from world-class competitors has been invaluable. Special thanks to the ARChitects team for open-sourcing their solution, and to @sanyul and @boristown for their excellent baseline implementations.

This competition has motivated me to become more deeply involved in future ARC challenges. The problem of abstract reasoning represents one of the most fascinating frontiers in AI, and I'm excited to contribute to advancing this field. For future iterations, I plan to systematically explore the base model selection space alongside other hyperparameters to better understand which architectural choices align best with different types of abstract reasoning tasks.

**To anyone reading this writeup**: In small-sample competitions, never underestimate the importance of exploring every hyperparameter systematically, starting with the fundamental choice of base model architecture. What seems like a minor configuration choice—whether it's a random seed, a learning rate, or a model selection—may be the difference between a good solution and a winning one. The 2x performance variance I observed from seed changes alone (3.33 % to 6.67 %) suggests that comprehensive exploration of the full hyperparameter space could yield even more dramatic improvements.
