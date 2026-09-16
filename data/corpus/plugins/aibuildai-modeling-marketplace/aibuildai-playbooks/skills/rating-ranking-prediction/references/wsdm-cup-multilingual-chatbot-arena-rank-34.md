# 34th Place Solution

Competition: wsdm-cup-multilingual-chatbot-arena
Rank: #34
Source: https://www.kaggle.com/c/wsdm-cup-multilingual-chatbot-arena/discussion/568535

First and foremost, I would like to extend my heartfelt thanks to the organizers of this competition and all participants for sharing their valuable ideas and insights. Thanks to the teammate @guanyiming @jiuxinfeng . Participating in this event has been an incredibly enriching experience, allowing me to learn a great deal and deepen my understanding of cutting-edge techniques in model training and optimization. It was both educational and inspiring. 

---

### Model Training

#### Base Model Selection

I chose **gemma2-9b-it-fp16** as the base model. This model has demonstrated strong performance in previous competitions.

#### Supervised Fine-Tuning with QLoRA

For supervised fine-tuning, I utilized **QLoRA** with the following hyperparameters:
- Learning rate: 2e-4
- Per-device train batch size: 2
- Gradient accumulation steps: 4
- Max token length: 3074 (for sequences exceeding this length, the middle portion of the response was retained, and the rest was truncated)

---

### Multi-Stage Training Process

#### Stage 1: Initial Fine-Tuning
In the first stage, I fine-tuned the base model using the **lmsys-33k-deduplicated** dataset along with additional data from **lmsys**. This resulted in **Weight 1**, which served as the starting point for subsequent stages.

#### Stage 2: Domain-Specific Fine-Tuning
Building on **Weight 1**, I further fine-tuned the model using the **WSDM competition dataset**, resulting in **Weight 2**. This stage aimed to adapt the model to the specific domain of the competition.

#### Stage 3: Pseudo-Labeling
In the third stage, I used **Weight 2** to generate pseudo-labels on two additional datasets:
1. The dataset provided in this discussion: [WSDM Multilingual Chatbot Arena Discussion](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/557746#3101848)
2. The **mlabonne/orpo-dpo-mix-40k** dataset from Hugging Face: [ORPO-DPO-Mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)

These pseudo-labeled datasets were then used to enhance the model's performance.

#### Stage 4: Re-training with Soft Label Distribution Optimization
Using the pseudo-labeled data from Stage 3, I re-trained the model alongside the **lmsys-33k-deduplicated** and **lmsys** datasets. Notably, I employed **soft label distribution optimization** during this stage, as hard labels did not yield satisfactory results. Soft labels provided a more nuanced representation of the data, allowing the model to learn more effectively. This produced an updated **Weight 1**, which incorporated the knowledge from both the original and pseudo-labeled data.

#### Stage 5: Final Fine-Tuning
Finally, I fine-tuned the model using the updated **Weight 1** on the **WSDM competition dataset** to obtain the **Final Weight**. This step ensured the model was fully optimized for the competition's specific requirements.

---

### Techniques and Optimization

#### Dynamic Batching
To accelerate inference, I implemented **dynamic batching**, which dynamically adjusted the batch size based on the input sequence length. This significantly improved inference speed without compromising accuracy.

#### Swap Response with TTA
I employed **swap response** as a form of test-time augmentation (TTA). However, due to time constraints, TTA was only applied to samples where the predicted probability difference was less than 0.25 after the first round of inference. This selective approach improved efficiency while still enhancing accuracy.

---

### Additional Notes
- **Token Truncation**: For sequences exceeding the maximum token length of **3074**, the middle portion of the response was retained to preserve the most relevant information.
- **Soft Label Optimization**: The use of soft label distributions in Stage 4 proved crucial, as hard labels failed to deliver the desired performance improvements.

---

### Conclusion

This multi-stage training approach, combined with techniques like pseudo-labeling, soft label optimization, dynamic batching, and selective TTA, allowed me to optimize the model's performance effectively. While the process was computationally intensive, the results demonstrated significant improvements in both accuracy and efficiency. Through this competition, I gained invaluable insights into advanced model training strategies, data optimization techniques, and the importance of iterative experimentation. These learnings have not only enhanced my technical skills but also deepened my appreciation for the challenges and opportunities in the field of AI. I am excited to apply these experiences to future projects and continue growing as a practitioner in this dynamic domain.
