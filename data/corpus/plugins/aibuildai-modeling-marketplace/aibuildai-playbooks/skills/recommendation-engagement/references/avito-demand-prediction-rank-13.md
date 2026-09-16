# The last gold solution

Competition: avito-demand-prediction
Rank: #13
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/59886

First of all, thanks to Avito and Kaggle for providing this great competition. No leakage, very representative testing set, countless solutions. I believe every one enjoyed the game.

Besides, thanks to all community contributors, like Dieter and Peter Hurford, who published helpful kernels and discussions.

Features:
Most of the ideas came from public kernels and discussions: tf-idf, CountVectorizor (char level), image meta features, self-trained W2V, ads time from Benjamin Minixhofer's aggregated features.

What I did additionally is clustering ads titles based on sub-categories. For example:

    sub_category = df[(df['category_name'] == 'something') &amp; (df['param_1'] == 'something') &amp; (df['param_2'] == 'something') &amp; (df['param_3'] == 'something')]
    tfidf = TfidfVectorizor()
    tfidf_vec = tfidf.fit_transform(sub_category['title'])
    kmean_cluster = KMmean()
    sub_category['title_cluster'] = kmean_cluster.fit_predict(tfidf_vec)

I did such calculation for all sub-catgories. After doing this, in the subcategory of iphone, all iphone 5s 32gb were in one group and iphone 7 128 GB were in another group. Then it was a good time to compare the prices and other features. So I groupby this 'title_cluster' and calculated aggregated features like price rank (which gave a big improvement), mean price, count and so on.

I also used regex to manually extract numbers out of the title for properties (area, room numbers, floor) and automobile (vehicle years) ads. Then calculated features such as price per room, price per sqm.

Other than that, since traditional ML models such as LightGBM are not good at handling unstructured data like text and images, I used NN (biLSTM for texts and a simple 4 layers CNN for image) to generate vector representations. Image vectors directly came from a NN that were used to predict the deal_probability. But when I tried the same strategy for texts, the generated text vector dramatically made my LightGBM overfitted. Then I tried another method: only using title and description W2V in a biLSTM NN model with MSE loss and predicting everything else: price, item_seq_number, city, region, user_id, parent_category_name, category_name, param_1, param_2, param_3. All categorical features are onehot encoded after removing low frequent entities. Therefore, X = title W2V + description W2V. y = a few hundred columns table.

Models:
Three layer of stacking. first layer: 11 Lightgbms, 6NNs. second layer: 3 LightGBM, 2 Ridge. third layer is just a Ridge and a LighGBM with linear average. I am a newbie of NN (my oof NN scores were very unstable) and hope to learn more NN strategies from other teams.

Thanks
