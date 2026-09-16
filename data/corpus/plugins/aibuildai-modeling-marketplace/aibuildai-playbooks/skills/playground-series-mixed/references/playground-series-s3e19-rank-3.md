# 3rd Place Solution: EDA is the key

Competition: playground-series-s3e19
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e19/discussion/428682

I'm very happy to get 3rd place on my first kaggle competition. Maybe I will write a notebook later, but it depends. I want to read other solutions first. Thanks @paddykb, @kdmitrie and all other people who contributed ideas.

# EDA and data processing
I think EDA is the most important part for this competition, and probably for most such competitions. 
## Correcting the trend

Based on the plot of daily total sales, it's obvious that trend in 2020 is abnormal. 
My method to correct the trend:
1.  Calculate daily fraction of each of the 75 categories.
2. Use STL decomposition to extract the trend from daily total sales series.
3. Replace the trend from 2020-02-15 to 2020-11-15 with the average trend of the other years.
4. Restore the daily total sales series.
5. Restore the series of each category based on its fraction.

## Modelling the baselines
Baseline for total daily sales seems to be discrete on each year. The baseline of each category can be modeled as:
$$B_{y, p, c, s} = B_y F_{y, p} F_{y, c} F_{y, s}$$
where *B* is baseline level, *F* is fraction, *y* is year, *p* is product, *c* is country, *s* is store. The subscript denotes the corresponding combined categories. So to find baselines of 2022 one needs to find all these values with *y=2022*.
### Fraction of country by year

Country fractions fluctuate with time. It turned out that these fractions are correlated with GDP per capita.  However for unknown reason all countries in 2022 have the same fraction, i.e., 0.2. Probably due to data construction error.
$$F_{y, c} = f_{y, c}; F_{2022, c} = 0.2$$
where *f* denotes empirical fraction and *y = 2017, 2018 ... 2021*.

### Fraction of product by year

Product fractions are roughly constant through years. There seems to be some seasonality of two-year period. But I'd rather capture that by temporal embedding. Here I set them constant as weighted average:
$$F_{2022, p} = F_{y, p} = (2(f_{2017, p} + f_{2019, p} + f_{2021, p}) + 3(f_{2018, p} + f_{2020, p}))/12$$

### Fraction of store by year

Store fractions seems to be constant through years. The fluctuation is negligible.
$$F_{2022, s} = F_{y, s} = f_{s}$$

### Baseline of total daily sales.
I used median as baseline. For 2022 I left it unknown and determined it by LB probing at the end.
$$B_y = median_y; B_{2022} \approx 18900$$

## EDA by category
### Standardization
With the baseline model, standardization can be done directly:
$$S_{y, p, c, s, d} =  N_{y, p, c, s, d} /  F_{y, p}  /  F_{y, c} / F_{y, s} / B_y - 1$$
where *N*  is num of sold, *S* is standardized num of sold, *d* is date.
### Subplots of daily means (after standardization) by category
#### By country

Holidays behave differently among countries, especially for Japan. 

#### By product

Seasonalities are different among products. Note that, though weak, some products have seasonalities of two years instead of one year.

#### By store

All stores seem to behave the same.

## Sum the store sales
Reason to sum the store sales:
1. Based on the analysis above, store fractions seem to be constant. And there is no difference of seasonality and holiday effect among stores.
2. For some categories, e.g. Using LLMs to Win Friends and Influence People - Argentina - Kaggle Learn,  the num of sold is too small and it oscillates between several integers. Such series may affect model training.


So I used the sum of daily sales by product and country for training and validation. Similar standardization applied:
$$S_{y, p, c, d} =  N_{y, p, c, d} /  F_{y, p}  /  F_{y, c} / B_y - 1$$
For prediction, convert this intermediate value back to num of sold: 
$$N_{y, p, c, s, d} =  (S_{y, p, c, d} + 1) B_y F_{y, p} F_{y, c} F_{y, s} $$
I put all model parameters to an excel table. See attachment.

#Training and prediction
## Pipeline
Since seasonalities are different among products, I trained each product separately. So there are 5 models. 
Training data is split by year to 5 folds, followed by cross validation.
## Feature engineering
Since private score is based on the last 75% of 2022,  lag features are not needed.
### Temporal embedding
- For week seasonality, I used Fourier terms up to order 3 .
- For two-year seasonality, I used Fourier terms up to order 5 .
### Holiday and date related features
I generated two categorical features:
- c_holiday: country + corresponding holidays
- c_year_date: country + 365 dates of a year

These categorical features are then converted to numeric levels by target encoding. I used linear mixed model with *S ~ temporal variables* grouped by the categorical variable. Of course, encoding is done separately for each product. To avoid overfitting, encoding is done with 4~5 fold OOS predictions.
See:
https://www.statsmodels.org/stable/mixed_linear.html
https://arxiv.org/pdf/2104.00629.pdf
I didn't do feature selection
##Modeling
I used lightbm with linear tree and dart. I didn't do much hyper-parameter tuning except I set `num_leaves` to a small value (~5). The optimal `num_iterations` is determined by the cross-validation RMSE. The final model is trained on the whole training set with optimal `num_iterations`.
I didn't do any ensmeble.
##Result
The validation SMAPE is 4.87.
Feature importance:

As mentioned above, baseline of total daily sales of 2022 is determined by LB probing. This is simply done by:
1.  Randomly guess a few values and get the scores
2. Use a parabola to fit.
3. Try the minimum point.
4. Add the result and fit again. 
5. Repeat 3-4 one or two times.

**That's it. Hope it helps.**
