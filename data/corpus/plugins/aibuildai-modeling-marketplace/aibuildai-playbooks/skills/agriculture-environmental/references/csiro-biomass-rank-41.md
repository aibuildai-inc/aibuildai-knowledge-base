# 🏅 37st Place Solution: Self Prompt Tuning with DINOv3

Competition: csiro-biomass
Rank: #41
Source: https://www.kaggle.com/c/csiro-biomass/writeups/41st-place-solution-self-prompt-tuning-with-di

# 🏅 37st Place Solution: **Self Prompt Tuning with DINOv3**

### *Generalization Without Validation or Fine-Tuning*

---

[Thumbnail]

## 1. Acknowledgements

I would first like to thank Kaggle, the sponsors, and the research team for hosting this competition. Initially, it genuinely felt that the dataset was too small. However, as the competition progressed and the leaderboard experienced massive shake-ups, it became clear that this was in fact somehow necessary for the dataset to be small. This was further concluded especially after I got surrounded by Public LB Toppers on Private LB where #1, #2 and #3 on Public LB got #38, #35, #36 on Private LB (respectively) whereas I went from #49 on Public LB to #37 on Private LB, hence surrounded by toppers. The limited data strongly penalized overfitting and forced participants to confront the true generalization ability of their solutions. In hindsight, this constraint was one of the most valuable aspects of the competition.

I would also like to express my gratitude to Qiyu Liao for being one of the most active and supportive competition hosts I have seen online on such kaggle competitions. Whether it was clarifying doubts, responding to discussions, or encouraging participants when they share findings openly, your presence significantly elevated the collaborative spirit of the competition. 💙

When it comes to Clearing doubts, A special mention also goes to PC Jimmy sir as well. When it came to debugging and clearing doubts, you consistently went above and beyond. Entire notebooks were carefully inspected, every issue was addressed, and explanations were delivered with remarkable clarity and patience, often breaking down complex problems to their simplest form. The fact that this was done selflessly and without any expectation of return is deeply inspiring and speaks volumes about the strength of this community. You are a great encouragement showing how supportive a person can really be.

---

### 📌 Community Contribution Note

Beyond the final solution, I actively shared insights and experimental findings throughout the competition. Two of my discussion posts became the **[2nd highest](https://www.kaggle.com/competitions/csiro-biomass/discussion/650736)** and **[5th highest](https://www.kaggle.com/competitions/csiro-biomass/discussion/669012) voted discussions** in the competition. These focused primarily on clarifying modeling ambiguities, particularly around the **Dead Matter target**, and on proposing practical strategies to handle its inherent inconsistencies. I believe these discussions helped surface important issues early and encouraged more robust solution design across the community.

---

## 2. Coming to the Solution now: Backbone Selection and Rationale

I experimented with several strong vision backbones, including DINOv2, DINOv3, and EVA02. Among these, EVA02 was particularly interesting during early experimentation because it consistently achieved higher cross-validation scores with less effort compared to both DINOv2 and DINOv3.

However, despite its promising cross-validation behavior, EVA02 consistently underperformed DINOv3 on the leaderboard. Moreover, models based on DINOv3 exhibited a noticeably smaller gap between training and validation losses. This discrepancy strongly suggested that DINOv3 was learning representations that generalized better beyond the training distribution.

The explanation is largely rooted in the way DINOv3 is trained. Its self-supervised learning objective, combined with its ability to retain rich second-order statistics through Gram matrix information, makes it exceptionally well-suited for dense prediction and regression-style tasks. In the context of this competition, where subtle spatial cues matter more than explicit semantic categories, this property turned out to be crucial.

---

## 3. Model Architecture

#### 📌 Please find the architecture implementation [here](https://www.kaggle.com/code/komilparmar/multicropvptinov3-implementation)

### 3.1 Visual Prompt Tuning with Task Tokens

After experimenting with many architectural variants, the final model was built around Visual Prompt Tuning combined with DETR-style query tokens, which I refer to here as task tokens.

In addition to the standard image patch embeddings, I appended two types of learnable tokens to the input of DINOv3. The first type consisted of four task tokens, one for each target variable, with the Dead target being derived. The second type consisted of sixteen prompt tokens intended to guide the backbone toward task-relevant representations.

During the forward pass, all tokens were processed jointly by the backbone. From the output, I extracted the task token embedding corresponding to each target, along with global average pooled and standard deviation pooled features from the patch tokens. These representations were concatenated and passed through target-specific heads to produce the final predictions. Note that each Target head only received its corresponding Task token with GAP and STD, not the other task tokens.

[Architecture]

---

### 3.2 Preventing Token Collapse with Orthogonality Loss

While monitoring training dynamics, I observed that both task tokens and prompt tokens gradually converged toward very similar representations. This collapse significantly reduced the effectiveness of having multiple tokens in the first place.

To counteract this, I introduced an explicit orthogonality loss that encouraged tokens to remain decorrelated. This ensured that each token focused on capturing distinct aspects of the input rather than redundantly encoding the same information. The addition of this constraint proved critical for stable convergence and improved robustness. You can assume that it performs the same role as that of multi head in MHSA (Multi Head Self Attention).

---

## 4. Self Prompt Tuning Initialization

To further improve convergence, I adopted the idea of Self Prompt Tuning.

For each target variable, I selected the ten training samples with the highest target values. From these samples, I extracted the CLS token embeddings and computed their mean. This mean representation was then used to initialize the corresponding task token. The same procedure was applied to initialize the prompt tokens, where the sixteen prompt tokens were divided into four groups (encouraging 1 group for each target).

However, this initialization strategy naturally caused a sharp increase in orthogonality loss because all tokens started from correlated representations. To resolve this, I applied Gram Schmidt orthogonalization at initialization time, ensuring that all tokens were mutually orthogonal before training began. This allowed the benefits of informed initialization without sacrificing token diversity.

---

## 5. Cross-Validation Strategy and Early Generalization Signal

Within the first month of the competition, I realized that the choice of cross-validation strategy would dominate everything else. I shared this insight early on in detail, including implementation specifics, in what later became the [most upvoted discussion](https://www.kaggle.com/competitions/csiro-biomass/discussion/615401) of the competition. However, the strategy was simply ignored by most.

The initial strategy involved Monte Carlo seed searching combined with GroupKFold splits over both State and Sampling Date, motivated by early speculation that certain states might be entirely absent from the test set. Later clarifications from the competition hosts confirmed that all states were present in the test data, while only specific sites or locations might be unseen. Based on this, I simplified the strategy to state-stratified GroupKFold splits over sampling dates only.

Even at an early leaderboard score of around 0.63, the alignment between cross-validation and leaderboard performance was striking. The mean cross-validation R squared score closely matched the leaderboard score, with only minor variance across folds. This alignment persisted throughout the competition and served as a strong signal that the modeling choices were correctly targeting generalizable patterns rather than exploiting validation artifacts.

---

## 6. Full Dataset Training Without Validation

As the competition progressed, training times increased substantially, while the dataset itself remained extremely small at only 357 images. At the same time, I wanted to exploit a well-known idea popularized by Chris Deotte sir, where models are retrained on the full dataset after estimating the optimal training duration from cross-validation.

Although this approach is most commonly applied to tabular models such as XGBoost, the underlying principle remains applicable here. I trained models on full dataset without any validation until they reached the mean training loss observed during cross-validation. This approach consistently produced slightly better leaderboard performance compared to ensembling cross-validation models, reinforcing the idea that data efficiency mattered more than validation monitoring in this setting.

---

## 7. From Two-Stream Concat to Mean Aggregation

I experimented extensively with two-stream architectures where the input image was split into two halves. Initially, I concatenated embeddings from each half and passed them jointly to the prediction head. While this worked reasonably well, it increased model complexity and sensitivity to noise.

I then moved to a two-stream summation approach, where each half independently produced a prediction and the results were summed. This formulation, however, suffered from a fundamental limitation. There was no supervision signal indicating which half contributed more to the error, and contrastive losses were not directly applicable. Attempts to resolve this using pseudo-labels generated from a strong ensemble did not yield improvements.

The breakthrough came when switching from summation to mean aggregation. In this setup, each stream was forced to predict the biomass of the entire region using only half of the available information. This intentional information bottleneck dramatically improved robustness and significantly reduced the generalization gap between training and validation losses.

[aggregation]

---

## 8. Random Cropping and Information Bottlenecking

At this point, a natural question arose. If the model could already predict the biomass of an entire region while observing only half of the image, how much visual information was it truly relying on?

To investigate this, I trained the model using random crops of size 448 by 448 sampled from the original 1000 by 2000 images. This restricted the model to roughly ten percent of the total image area per forward pass. Despite this severe information bottleneck, the model continued to perform well and achieved leaderboard scores above 0.6, indicating that it was learning global and textural cues rather than memorizing local patterns.

I then extended this idea using a multi-crop training strategy. Multiple random crops were sampled per image, predictions were made independently, and their mean was compared against the ground truth to compute the loss. With two crops, an important equilibrium emerged. The training, validation, and leaderboard R squared scores all converged to approximately 0.72. This alignment suggested that the model had effectively stopped overfitting, making validation redundant. At this stage, I transitioned to training on the full dataset without a validation split.

The behavior became surprising as the number of crops increased. The training R squared score rose rapidly, with 3 crops I got a Train R2 of 0.85, yet the validation score remained 0.72. Wait, what? Even when using eight random crops, covering nearly the entire image, the validation performance stayed fixed at around 0.72. This implied that while the model was learning noise, it was not unlearning the truly generalizable features.

When these models were trained on the full dataset and evaluated on the leaderboard, a clear pattern emerged.

Train R squared to Leaderboard Score Relationship

| Train R² | Leaderboard Score |
|----------|-------------------|
| 0.72     | 0.72              |
| 0.80     | 0.72              |
| 0.85     | 0.73              |
| 0.90     | 0.74              |
| 0.92     | 0.75              |
| 0.95     | 0.75              |

Empirically, every increase of roughly 0.05 in training R squared translated to an improvement of about 0.01 on the leaderboard. This counterintuitive effect ultimately shaped my final strategy. Rather than aggressively preventing overfitting, I allowed it in a controlled manner, trusting that the general representations learned through severe information bottlenecking and self-supervised pretraining would remain stable. At this stage, it is worth to again note that I did not even touch the backbone at all.

[Crops]

---

## 9. Inference and Ensembling

During inference, each image was divided into tiles of size 500 by 500, hence extracting 8 tiles from 1000x2000 image, from which a central 448 by 448 crop was extracted. Test-time augmentation was limited to simple geometric flips, including horizontal, vertical, and combined flips. No post-processing was applied.

For the final submission, I ensembled multiple checkpoints obtained from the same training run presented above. I also experimented with exponential moving average and stochastic weight averaging. While these techniques improved convergence smoothness, they did not meaningfully raise the performance ceiling.

---

## 10. Conclusion

This solution is the result of continuous experimentation, many failed ideas, and a strong reliance on community-driven learning. I benefited immensely from open discussions, shared insights, and collective debugging. This write-up is one small attempt to give something back to the community.

I strongly encourage questions and discussions. The best ideas in this competition emerged through shared curiosity and healthy skepticism, and I hope this contributes to that spirit.
