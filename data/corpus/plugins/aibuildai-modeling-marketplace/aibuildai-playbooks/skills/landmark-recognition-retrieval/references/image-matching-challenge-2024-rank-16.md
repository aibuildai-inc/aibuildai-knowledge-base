# [Prize Eligible] 16th place solution  - ALIKED4Khh+DISKh+SIFT

Competition: image-matching-challenge-2024
Rank: #16
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/509883

Thanks for the IMC 2024, always a fun and good area for testing and experiment codes and ideas, benchmark against the old ones! Regardless of the outcome, the common for all competitions are the knowledge one gets, valuable knowledge and insights.

## **The path to the final solution**

First and foremost, the most important, from the past three IMC reading the papers and codes until “solution level of consciousness”, find the challenge, the solutions and the final summaries, what worked and what didn’t.

## **Three things that made the base for the IMC 2024**

Colmap randomness – be aware if it, fix it if needed during validation.

Prize-eligibility – many top solutions use combination of SP-SG, but for the prize and main host challenge use the correct license. Filter out the non-Prize Eligible and collect the best code and ideas with correct licenses and try developing them further.

Validation set – try finding some correlations besides the public leaderboard while testing the ideas.

## **Validation set**

From reading the past Competition Recap from the host one could see in the Per-dataset results that the Urban dataset was the hardest dataset to find a solution for and also with team scores similar to IMC 2024 public dataset score and also a comment from the host “No single team tried to work on this part” – one could have this guessing in mind that maybe the host wanted this to happened this time.

Anyways, the private test set is unknown so best is to collect a diverse validation set, for this I used past competition and the IMC2020 PhotoTourism dataset besides the present competition dataset.

## **Prize Eligible Solutions**

I started with using the two past top codes, the 1st and 2nd Prize Eligible winners, try developing and validate them in parallel for weighted risk-reward-work to a final submission.

Links to their awesome writeups below

https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/427143
https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417045

## **AffNetHardNet8 + AdaLAM solution**

I first fixed the randomness before further testing.

From validation I could see that a mix of image sizes and Harris Corner model worked best on the validation set and LB.
The final setup with a good general score was with KeyNet with low image size, GFTT and DoG with medium and a mix of Harris Corner with different larger images sizes, too also increase the ransac thresholds helped.

Many solutions without randomness fix had better scores, but I can’t guarantee them 😉

I had idea to implement NN matching from the other solution and also try codes like MatchFormer and QuadTreeAttention from my IMC 2022 21st solution https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328896 but the time wasn’t there.

## **ALIKED4Khh+DISKh+SIFT solution – my best Public and Private score .177459/.74038**

It uses for example lightglue, netvlad retrieval, pixsfm, rotation matching localize unregistered images. You can read all about the details here https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/427143

From reading the past top solution writeup, they did some after submission testing and validation, and those values was a good setup when changing the default setup. But while tuned by validation check I saw that even higher value and detection threshold for ALIKED worked even better and also with resize force.

The plan was also to implement the new open version of superpoint, which I did to 50% with extraction of features but didn’t have the final time for the final sp-lightglue code.

## **Other**

I did some submission with a combination of above, using different setup for different scenes, as the scores shifted per validation scene, it went well but as the private set scenes are unknow, I didn’t gamble on it.

----------------
