# Public 531st -> Private 10th: The Complete Solution Code

Competition: godaddy-microbusiness-density-forecasting
Rank: #10
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/418770

# **GoDaddy - Microbusiness Density Forecasting**

The solution is 3 months ago, so if there is anything missing, please let me know.

The objective of the competition is to predict monthly **microbusiness density** in **3315 cfips** (County FIPS) in the United States. The code implementation incorporates several special features and techniques to improve the accuracy of the predictions.

## **Approach**

### **1. Predicting 3315 Time Series**
This approach predicts the microbusiness density for each of the **3315 cfips** individually. This allows for more granular and accurate predictions tailored to specific regions.

### **2. Technical Indicators**
The code utilizes popular technical indicators such as **Exponential Moving Average (EMA)**, **Momentum (MOM)**, and **Relative Strength Index (RSI)**. These indicators capture underlying trends, momentum, and market conditions, enhancing the predictive power of the model.

### **3. External Datasets**
In addition to the microbusiness density data, the model incorporates several external datasets. These datasets include information such as **unemployment data**, **earnings**, **rent**, **DSG10**, **tax rate**, **housing price**, and **population estimates**. By incorporating these relevant external factors, the model can capture the influence of broader economic and demographic factors on microbusiness density.

### **4. Optuna for Hyperparameter Optimization**
**Optuna** is used to minimize the **SMAPE** in the **Catboost** model.

### **5. Catboost Model with Cross Validation**
The code employs the **Catboost model**. Also used **Cross Validation (CV)** to generalize well to unseen data.

### **6. Multiple Model Training**
To predict multiple future time periods, five separate models are trained. Each model is designed to predict a specific time horizon, such as **t+1 month**, **t+2 months**, and so on.

### **7. External Dataset Addition**
The code "**6-external-datasets.ipynb**" is found on Kaggle and is provided to incorporate additional external datasets into the prediction model.

## **Limitations**

- **Limited Historical Data**: The current approach utilizes only the past three months of data to predict future microbusiness density. Exploring longer intervals of historical data could provide additional context and potentially improve the forecasting accuracy.

## **Code Files**

- **6-external-datasets.ipynb**: add additional external datasets to the prediction model. It can be found on Kaggle and is intended to enhance the feature set.

- **kaggle_competition_microbusiness.ipynb**: contains the full model training code for microbusiness density prediction. It encompasses data preprocessing, model training using **Catboost** and **Optuna**, and evaluation using **SMAPE**.

**Github link**: [https://github.com/ttterence927/kaggle_competition_microbusiness/](https://github.com/ttterence927/kaggle_competition_microbusiness/)

## **Show Your Support**

If you find this code implementation valuable or interesting, please consider giving it a star on **GitHub**.
