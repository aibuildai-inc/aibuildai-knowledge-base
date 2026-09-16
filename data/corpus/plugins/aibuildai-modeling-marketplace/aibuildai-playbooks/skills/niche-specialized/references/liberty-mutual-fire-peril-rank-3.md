# After shock, Let's talk about solutions

Competition: liberty-mutual-fire-peril
Rank: #3
Source: https://www.kaggle.com/c/liberty-mutual-fire-peril/discussion/10194#53012

<p>I didn't have much time to work on this competition, so I kind of brute forced my way through it, without deep exploratory studies of the features.</p>
<p>My submission uses 1000 individual estimators in total. Half are GBMs and half are Extra Trees. I heavily undersampled the data. Each estimator deals with approx 10K data points.</p>
<p>As for the features, half of the estimators deal with only the contract related data. The other half uses sklearn's feature selection to pick top 150 features from contract + crime + geodem + weather. Note that each estimator runs its own feature selection based on its 10K data points.</p>
<p>There's one little trick I used, which I guess others have also done. Instead of predicting the losses directly, I took the logarithm, and predicted on that.</p>
<p>The total training + predicting time on my several years old laptop is around 5-6 hours.</p>
