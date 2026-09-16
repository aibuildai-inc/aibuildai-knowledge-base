# 3rd place memo

Competition: LANL-Earthquake-Prediction
Rank: #3
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94459#latest-565843

Thanks to organizers for hosting such an interesting competition, and congratulations to all the top teams.

Already mentioned in the [discussion](https://www.kaggle.com/c/LANL-Earthquake-Prediction) , we expected test from p4677.
So we can roughly estimate the test ttf.
Based on this information, We adjusted the ttf of train.
Green line(I will call it best y) is 12.625.

By training based on the above adjusted ttf, you can get a good private score.
For example, RNN with only 6 features and simple GRU, you can get 2.33 on privateLB.
Actually, lightgbm with many features is slightly better, so we mainly used it.

How to find best y?
For example, consider the situation:
・train data : 1~11 earthquakes of train
・test data : 12,13,14,15 earthquakes of train
The best value can be calculated as follows.


    max_ttf_list = [8.828100, 8.566000, 14.751800, 9.459500]
    chunk_length_list = [33988602, 32976890, 56791029, 36417529]
    best_score = None
    for y in np.linspace(5,15,1000): # search range
        error_list = []
        segment_list = []
        for max_ttf, chunk_length in zip(max_ttf_list, chunk_length_list):
            slope = max_ttf/chunk_length*150000
            segument_num = chunk_length//150000
            ttf_true = [max_ttf-slope*i for i in range(segument_num)]
            ttf_adjusted = [y-(y/segument_num)*i for i in range(segument_num)]
            error = abs(np.array(ttf_true)-np.array(ttf_adjusted)).sum()
            error_list.append(error)
            segment_list.append(segument_num)
        score = sum(error_list)/sum(segment_list)
        if best_score is None:
            best_score = score
        elif best_score &gt; score:
            best_score = score
            best_y = y
    print(best_y)


If training by default ttf, MAE of test data is 2.251.
If training by best y and modified ttf, MAE of test data is 1.948.

It may be said that this competition is a leak, but it was interesting to think about creating a better model on that premise.

We would like to say thanks to everyone.
That's all. Thank you for your reading.
