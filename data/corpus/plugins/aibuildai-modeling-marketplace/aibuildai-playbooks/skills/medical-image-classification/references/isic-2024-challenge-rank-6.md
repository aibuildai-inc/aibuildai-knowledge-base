# 6th palce solution

Competition: isic-2024-challenge
Rank: #6
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532868

I'd like to express my honor in having the opportunity to share my solution with you. Before diving into that, I would like to say a few words. I first encountered Kaggle when I was a freshman in college. Now, having worked for two years, it has been nearly nine years since then. Throughout this time, I have always aspired to earn a gold medal, but I never participated in a competition with a serious commitment. The main reason for this was the belief that I didn’t have enough time. Now, as I have started working, my leisure time has become even scarcer. However, I have come to realize that unless I dedicate enough time to study outstanding solutions and conduct experiments, I will never achieve the goal of winning a gold medal.

So, in the past two months, I have sometimes woken up an hour early to work on competitions, tried to squeeze out an hour of my lunch break to do the same, and occasionally spent an extra two hours after work in the evenings. I am fortunate that this is my first time putting my full effort into a competition and that I achieved a good result. I am also very grateful to the excellent participants on the Kaggle platform. It is because of your presence that Kaggle has become so captivating.

Now, here is my solution.

Overall Solution: Image Model as Features + LGB/CBT
Image + Meta Feature Model
Like most participants, I first trained an image model. The overall framework is identical to the code of the [first-place winner in 2020.](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412) Thanks to Haqishen and his teammates, I made almost no changes to the code logic and continued to train the model using the image + meta features approach.

For the data and cross-validation methods, I used the [Triple Stratified Leak-Free KFold CV data](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/165526) that Cdeotte previously shared. This data is a mix of records from 2020, 2019, 2018, and 2017, to which I added the 2024 data. I noticed that some patient IDs appear in both the 2020 and 2024 datasets, so I re-divided the data to ensure that they were assigned to the same fold. If you are interested in how the 2024 dataset was divided for cross-validation, you can check out this [kernel](https://www.kaggle.com/code/mnk812/20240717-get-duplicate-id/notebook).

During my experiments, I found that due to the small size of the 2024 images, using image sizes of 128 and 192 yielded the best results. I compared several models from the EfficientNet series and found that EB1 with 128 size and EB3 with 192 size performed the best, so these are the models I finally used. The CV score for EB1+128 was 0.165 and LB was 0.165, while the CV for EB3+192 was 0.168 and LB was 0.158.

Another critical point is that, inspired by Haqishen's solution, I changed the binary classification to a nine-class classification. Through data analysis, I found that by modifying iddx_3, I could achieve this goal.

You can refer to this [kernel](https://www.kaggle.com/code/mnk812/2024908-train-image-meta/notebook) for the modifications mentioned above. Thanks again to Haqishen and Cdeotte.

LGB/CBT Model
By introducing the prediction results of the image model as features, I didn't do much digging into the meta features because I found this excellent[ kernel](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data). The feature construction in this kernel is very logical. Thanks to greysky for sharing. You can refer to this [kernel ](https://www.kaggle.com/mnk812/20240823-merge-trainv1)to find my code.

Ensemble
The correlation between CV and LB of my models is consistently high, which makes me very happy. I merged two seeds, each containing one LGB and one CBT model, weighted by CV results. You can refer to this [kernel](https://www.kaggle.com/code/mnk812/20240904-fold-merge-predict/notebook). Additionally, since the EB3+192 model has particularly long prediction times, in another solution, I used this model to re-predict the top 70,000 samples with the highest predicted probabilities from the LGB model for weighted averaging. You can refer to this [kernel](https://www.kaggle.com/code/mnk812/20240905-fold2-merge-predict/notebook).

Conclusion
My best result is 0.173([this kernel](https://www.kaggle.com/code/mnk812/20240813-mergev2-test/notebook)), using the EB3+192 image model, but this model gave me an LB of 0.158 at that time, which was inconsistent with the CV results. So I hesitated to use this model. However, the results show that it wasn't overfitting, and EB3+192 indeed performed better on the PB. So, trust your CV; it's really, really hard to achieve.

Finally, I would like to pay tribute to the Kagglers who continually share and help solve problems during the competition. It is because of your presence that Kaggle is so attractive, and I have learned a lot of knowledge. Thank you.
