# 6th place solution End to End Sequence to Sequence with sliding window.

Competition: rsna-intracranial-hemorrhage-detection
Rank: #6
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117301

Congrats to all participants and the winners, and myself, I must say, for becoming a Kaggle GRANDMASTER!  Catching that elusive fifth Gold medal that I have been chasing for some time now.


Down to business. We utilized an end to end sequence to sequence model. A sliding window approach was used to select a fixed window of “n” slices from the ct volume. The FIG above shows the architecture. The architecture made things nice and simple, training was end to end, no data shuffling and gymnastics. Prediction on the slices was done by the LSTM at each time step. This conveniently also enabled some nice test time augmentation (TTA) with the sliding window approach.
