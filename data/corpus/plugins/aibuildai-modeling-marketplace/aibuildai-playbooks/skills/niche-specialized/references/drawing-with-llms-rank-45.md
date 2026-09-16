# 45th Place Solution

Competition: drawing-with-llms
Rank: #45
Source: https://www.kaggle.com/c/drawing-with-llms/discussion/582242

# Acknowledgements and Reflections on the Competition

First of all, I would like to express my **sincere gratitude to Kaggle** for organizing this competition, which provided an excellent opportunity to expand my knowledge and skills.  
I would also like to thank Kawsar Hossain @kawchar85 for his insightful post  ([https://www.kaggle.com/competitions/drawing-with-llms/discussion/576181#3207144](url)), which served as a significant source of inspiration for my work.

---

## Competition Dynamics

In the final week of the competition, the dynamics on the leaderboard **intensified considerably**, leading me to suspect that a novel state-of-the-art (SOTA) text-to-image model might have been discovered by some participants.

Consequently, I conducted a thorough review of recent publications on **arXiv**, **Hugging Face**, and **OpenReview**, yet found no substantial breakthroughs.  
This prompted me to re-evaluate my strategy and focus on **optimizing my existing solution**.

[Figure 1]

---

## Key Experimental Findings

Below is a summary of key experimental findings from the competition:

### 1. **SDXL Base / Turbo and SDXL Hyper**
- **User-friendly** and require minimal parameter tuning.
- Facilitate a **lower barrier to entry**.
- I adopted this approach during the mid-stage, which initially achieved a **high ranking** but declined towards the end.

### 2. **SANA 1.5**
- Performs well with **straightforward parameter settings**, yielding relatively high scores.
- I hypothesize that **more advanced hyperparameter optimization** could further improve performance; however, time constraints precluded additional exploration.

### 3. **DMD2 (4-step UNet generation)**
- One of my final choices.
- By utilizing the **demo code provided in the Model Card**, I achieved a high score with minimal effort.
- **Simple parameter adjustments** resulted in an approximate **0.02 improvement**.
- Further fine-tuning could yield even better results.
- **Demonstrated considerable stability**.

### 4. **SDXL Flash**
- Also among my final submissions.
- Achieved a score of **0.67 on the public leaderboard** using the demo code.
- **Basic parameter tuning** allowed for further improvements.
- **Limitation:** Instability—identical configurations often produced significantly different results across multiple submissions.

---

## Summary

Throughout the competition, I experimented with a **variety of models and strategies**. While some models offered **ease of use and robust baseline performance**, others required more careful tuning to achieve optimal results. **Stability and reproducibility** emerged as important considerations, particularly in the final evaluation stage.
