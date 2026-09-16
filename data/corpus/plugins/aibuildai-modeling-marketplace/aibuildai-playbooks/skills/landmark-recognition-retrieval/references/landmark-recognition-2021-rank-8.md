# 8th place solution

Competition: landmark-recognition-2021
Rank: #8
Source: https://www.kaggle.com/c/landmark-recognition-2021/discussion/276636

Again, thanks to Kaggle and the hosts for hosting an interesting competition. Congratulations to all the winners. Special thanks to all my teammates @takuok and @tascj0 . Fortunately, we got the gold medal of Recognition competition, too. We show the details of our models in [Retrieval solution post](https://www.kaggle.com/c/landmark-retrieval-2021/discussion/276632).
Here, we show the unique part of Recognition challenge.

# Tips in Recognition.

We used the same two strategies as the last year's 1st team used. While submitting, we used pre-computed embedding of train4.1m data, penalized these embeddings that are similar to the non-landmark images. These methods are again very robust this year and critical for the gold medal.
This approach boost the score for a single model from **public/private 0.37843/0.36045** to **0.45724/0.43252**.
Final score is **ensemble of beit, swin, b4, b6, xcit, v2m, b5**. See the details about models in [Retrieval post](https://www.kaggle.com/c/landmark-retrieval-2021/discussion/276632).

# Acknowledge
takuoko is a member of Z by HP & NVIDIA Data Science Global Ambassadors.
Special Thanks to Z by HP & NVIDIA for sponsoring me a Z8G4 Workstation with dual RTX6000 GPU and a ZBook with RTX5000 GPU.
This competition has the big dataset.
So I tried pytorch's DDP parallel training on my dual RTX6000 GPUs and it helped a lot.
