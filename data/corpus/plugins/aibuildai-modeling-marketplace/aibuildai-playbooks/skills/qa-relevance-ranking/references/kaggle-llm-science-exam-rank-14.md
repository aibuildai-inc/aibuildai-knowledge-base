# 14th place solution: An attempt to reverse-engineer the dataset

Competition: kaggle-llm-science-exam
Rank: #14
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446484

Unlike many other competitions, I believe in this contest emphasis was placed on the quality and preprocessing of the data rather than complexity of the model architecture. Since we only had 200 rows of initial data, data gathering became a pivotal part of a solution. Here are some key points of my approach:
## Manual labeling of train.csv
The starting point was the train.csv. I manually found the origin page of each question. Although I don't believe it was 100% accurate, it served its purpose.
**Key Discovery:** A significant observation during this phase was that many questions were generated from the same categories. Some origin pages were from non-scientific articles like "Droste effect" and "Triskelion" from "Category:Symmetry". Yet, the majority of other articles from this category, like "Spontaneous symmetry breaking", were related to science. This led to my hypothesis that **the competition dataset was possibly generated from science categories** rather than specific science pages. Moreover, this seemed like an automated extraction rather than a manual selection.  
## Constructing a category-centric dataset
With the newfound understanding of categories playing a significant role, I focused on constructing a category-centric dataset:
- **Adjacent by Category (AC) Dataset:** I extracted all categories from the manually labeled csv and fetched every page within these categories. The pages were then split into segments, approximately 1500 characters each.  
- **Training Dataset Creation:** Using gpt3.5, I created 5 multiple-choice questions, each having 5 options, for every 1500-character segment. This generated a massive dataset of about 750k segment-question-answer triplets, which became the one and only dataset of my model training. [Link](https://www.kaggle.com/datasets/chubasik/kaggle-llm-science-exam-gpt-ac-dataset/)
### Dataset Variations
To expand the available data, I introduced several versions of the AC dataset. These datasets were only used during model inference/validation, not the training part:
- **Grandparent (GP) Dataset:** Starting from the grandparent categories (i.e parent categories of parent categories) of pages in the labeled train.csv, I fetched every child and grandchild page within these categories.  
- **Natural Sciences (NS) Dataset:** A graph search on "Category:Natural_sciences" with a 5-level cutoff covered 193 out of 200 pages from train.csv. To include the remaining 7 pages I added pages from other categories with varying cutoff levels ("Category:Mathematics", "Category:Measuring_instruments", "Category:Crystals", "Category:History_of_science_by_discipline").  
- **GP Plus 1 (GPP1) Dataset:** An enhanced version of the GP dataset, this included child categories of the leaf categories. It was also heavily modified by manually removing hundreds of unrelated categories, e.g. "Category:Soundtrack_albums_by_year".  
- **AC-750 Dataset:** Similar to the AC dataset but with segments of 750 characters.  
  
(The dataset nomenclature was improvised and it later stuck)  
## Article search using vector and TFIDF search  
With the datasets ready, I utilized a hybrid search strategy. A combination of vector (with e5 and bge models finetuned on AC) and tfidf similarities were used to identify the most relevant excerpts corresponding to the questions.
## Final Model Ensemble
My final solution was an ensemble of three Debertas(ForMultipleChoice) with some insignificant differences, like (1) the number of output layers, (2) the order of context-question-option in their inputs, (3) the model awareness of other options. Each of these models was enhanced with 4 distinct context augmentations - GP, NS, GPP1, AC-750.  

In conclusion, this competition was a great learning experience. By observing patterns, recognizing the significance of categories, and implementing diverse strategies, I managed to build a solution that was robust enough to survive the transition to private leaderboard.
