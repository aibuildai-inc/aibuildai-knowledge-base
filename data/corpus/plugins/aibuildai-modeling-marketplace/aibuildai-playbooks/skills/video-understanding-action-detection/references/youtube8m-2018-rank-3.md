# 3rd Place Solution Sharing: NeXtVLAD

Competition: youtube8m-2018
Rank: #3
Source: https://www.kaggle.com/c/youtube8m-2018/discussion/63223

Firstly, I would like to thank the Kaggle and Google team for hosting this challenge and congratulate every participant for completing such a challenging competition. 

During the competition, I devote myself to design a new network architecture to aggregate the frame-level features. Inspired by the work of ResNeXt, the final model, NeXtVLAD is found to be both effective and parameter efficient.  Briefly speaking, the basic idea is to decompose a high-dimensional feature into a group of relatively low-dimensional vectors with attention before applying NetVLAD aggregation over time.
A single NeXtVLAD model with less than 80M parameters achieves a GAP score of 0.87846 in private leaderboard. A mixture of 3 NeXtVLAD models results in 0.88722.

The code is publicly available at: 

https://github.com/linrongc/youtube-8m

Submitted Paper: 

https://github.com/linrongc/youtube-8m/blob/master/eccv2018submission.pdf

Presentation Slides: 

https://github.com/linrongc/youtube-8m/blob/master/ECCV2018_phoenix_lin_presentation.pdf

Time is really tight to write down all the details and clean up the code in just one week, especially for an individual participant. I will keep updating the guideline and paper.
Please let me know if you have any questions!
