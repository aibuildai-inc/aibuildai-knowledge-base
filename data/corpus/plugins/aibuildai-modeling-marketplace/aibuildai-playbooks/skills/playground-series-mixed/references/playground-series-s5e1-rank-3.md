# 3rd place solution

Competition: playground-series-s5e1
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s5e1/discussion/560535

First of all, let me thank the organizers of the competition and all the participants for interesting discussions!

In this competition, my solution was based on consecutive building of a multiplicative model, its fine-tuning and cross-validation. I was lucky to get the 1st LB position very soon, and my efforts were focused on how to prevent overfitting.

**[You can check my final solution here](https://www.kaggle.com/code/kdmitrie/pgs501-3rd-place-solution)**

# 1. The basic model
I have **[made the basic model public](https://www.kaggle.com/code/kdmitrie/pgs501-model-2-additional-country-doy-factor)**, and it reached the public score of about 0.05, that outperformed most of public notebooks without ensembling. The model is fully described in the notebook. In short, it incorporates the following factors:
- GDP per capita;
- Store ratio;
- Country ratio;
- Periodic product factor;
- Day-of-week factor;
- Periodic day of year factor;
- Periodic date factor;
- Country-dependent day-of-year factor.

# 2. Holidays
The basic model takes the holidays into account by averaging the sales across the years. This is not correct since the exact date of many holidays may differ year to year.

To overcome this difficulty, `holidays` library was used. However, although this library is cool, it is not perfect. It doesn’t include several holidays in Kenya (Festival of breaking the fast, Moi Day, Feast of the Sacrifice); uses different names for the same holiday (Kenyatta Day and Mashujaa Day); and needs a care when two holidays are in the same day. Moreover, it outputs ‘normal’ as well as ‘observed’ holidays. My experiments showed that it is better to consider these two types of holidays as separate holidays. As a result of using this library, each holiday in each country is represented a separate column in the dataframe, where ones are put in the holiday dates and zeros are put elsewhere.

It [was noticed]( https://www.kaggle.com/competitions/playground-series-s5e1/discussion/559828), that the ‘holiday effect’ takes place a few days later than the actual holiday. My experiments showed, that it could be described with a simple Gaussian curve: 
$$H(t) = \exp\left\(- \dfrac{(d-d_h-d_0)^2}{2\sigma_0^2} \right\),$$
where \\(d\\) is the current date; \\(d_h\\) is the date of the holiday; \\(d_0=4.5\\) is a response shift, and \\(\sigma_0=2\\) is the width of the Gaussian curve. The whole response is calculated as a convolution of \\(H(t)\\) with the data from the corresponding holiday column and its multiplication by amplitude factor, that needs to be determined.

# 3. Strange 1.06 multiplicator
The submission score gets sufficiently better being multiplied by a factor of about 1.06, credit to @cabaxiom for discovering this. It could be improved even further if the predictions for Kenya are also multiplied by about 1.03. This was very strange for me since it was hard to describe this factor, and using it sufficiently decreased the score during CV. I think this factor and its understanding is the key to the competition. 

Finally, after accounting for all the factors I discovered that the sales depend on time in a complex way, and we need to make a prediction (see the figure below). My efforts to explain it by some kind of economic indicators or by features already existent in the dataset failed. The pattern is uncertain, because it is not linear due to a wave in 2010-2012. Although the linear continuation over 2017 explains the 1.06 and 1.03 factors quite precise, we can’t be sure that this linear growth exists in 2018-2019. So, I decided to make my first final submission under the hypothesis of linear trend, while the second assumes it to be constant after 2018-01-01.



**Looking at the private score, it was a good decision: the second submission achieves much better score!**

- To describe the mentioned trend, I used a ReLU function: \\( \text{trend}(d) = 1 + s \cdot\text{ReLU}(d-d_1)\\), where \\(s\\) and \\(d_1\\) are the slope and shift parameters, correspondingly.
- To deal with the differences between Kenya and other countries, I trained an additional model on the Kenya’s data only.

# 4. Parameters optimization
In the basic model, the parameters were optimized using sequential regressions. This seems to be not optimal, and it is better to optimize them simultaneously. Moreover, the MAPE metric used in the competition differs from the MSE.

So, instead of this, I’ve built a `predict` function:
$$
\text{predict}(\hat X, \vec\alpha) = \alpha_0\prod\limits_{n=1}^N(\hat 1 + \hat X_n \vec\alpha_n),
$$
where the whole dataframe \\(\hat X\\) is divided into \\(N\\) sets of columns \\(\hat X_n\\): \\(\hat X = \[\hat X_1, \hat X_2, \dots, \hat X_N \]\\), and \\(\hat 1\\) is a column-vector of ones.

At the beginning, all the parameters are initialized with zeros or something reasonable. Then the training is performed using one or another set of columns. Finally, the training is performed using all columns. This is done for all countries and for Kenya only.

I used simple but powerful `minimize` from `scipy.optimize` to perform the optimization. This allowed me to quickly experiment with many factors and functional dependencies without the need to create the entire infrastructure necessary for DL frameworks.

The 6-fold CV was performed on the per-year basis, and the final model was trained on the whole data.
 


# 5. Drop the data
I have discovered that if we drop the 2010-2012 data for Kenya, both local CV and public LB score improve significantly. So I did this in my final model. 

Unfortunately, **it was a good idea for improving public score, but it was bad for the private score**. [The notebook without data drop](https://www.kaggle.com/code/kdmitrie/pgs501-separate-countries-without-data-drop) achieves the private score of 0.04369, that corresponds to the 1st place.

# 6. What didn’t work for me in this competition
- AutoML;
- DL (I can’t wait to see Chris’ solution!);
- More complex holiday response functions as well as efforts to change the \\(d_0\\) and \\(\sigma_0\\);
- Additional economic factors except for GDP per capita;
- ARIMA-like models.

# 7. Human or AI?
Currently, I don't know, what did @georgekoussa used in the competition. 
 
However, @cdeotte was so kind to share his DL notebooks. For me, it was particularly interesting, which solution would be better in the competition of such a type, that a human can perform the feature extraction 'by hand'. Okay, we can conclude, that both solutions are approximately at the same level. However, I believe, DL approach would be much better in more complicated tasks.

**[You can check my final solution here](https://www.kaggle.com/code/kdmitrie/pgs501-3rd-place-solution)**
