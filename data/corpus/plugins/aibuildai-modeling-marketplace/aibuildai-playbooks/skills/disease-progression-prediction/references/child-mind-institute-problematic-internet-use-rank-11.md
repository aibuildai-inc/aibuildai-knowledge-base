# 11th Place Solution for the Child Mind Institute — Problematic Internet Use Competition

Competition: child-mind-institute-problematic-internet-use
Rank: #11
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/553030

We are thrilled to achieve a gold medal in our first Kaggle competition and deeply grateful to everyone at Kaggle for all that you’ve shared, which greatly contributed to our learning journey. In this write-up, we’re excited to share our solution and key takeaways.

# Context
- Business context: https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview
- Data context: https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data

# Overview of the Approach
Our model builds on this notebook: https://www.kaggle.com/code/laiyunghwei/lb-0-493. We used three gradient boosting models-LightGBM, XGBoost, and CatBoost-and combined them using a voting regressor with equal weights.

# Details of the Submission
## Exploratory Data Analysis (EDA)
### Missing values
Missing data was a significant challenge. Nearly all features, except demographic ones, contained missing values, with about 10 features having missing rates exceeding 70%. The target variable also had approximately 30% missing data.

Additionally, features from the same instrument had similar missing rates, and the missing values were randomly distributed, allowing us to handle them through either direct removal or imputation.

### Outliers 
We identified multiple outliers after a detailed examination of each feature:

| Feature | Outlier Standard | Number of Outliers |
| --- | --- | --- |
| `CGAS-CGAS_Score` | Extremely high value (e.g., 999) | 1 |
| `Physical-Weight` | Invalid weight value (e.g., 0) | 61 |
| BIA related features (such as `BIA-BIA_TBW`, `BIA-BIA_ICW`) | Values significantly higher or lower compared to the normal range | 2 |
| `Physical-HeartRate` | Abnormally low heart rate (<30 bpm) | 1 |
| Total Body Water percentage (`BIA-BIA_TBW`/`Physical-Weight`) | <20% or >100% | 10 |

### Highly correlated features
A correlation heatmap revealed that features from the same instrument were often highly correlated (e.g., Bio-electric Impedance Analysis (BIA) features had correlations > 0.9).

Additionally, some features, like BMI, could be derived from others (BMI = FFMI + FMI). Reducing such redundancy was crucial for simplifying gradient boosting models.

### Correlation with age
Physical features like height and weight exhibited strong correlations with age. By normalizing these features (e.g., height/age, weight/age), we can extract more meaningful information. And age-based regression seems a reasonable method for imputing missing values.

### Time series data
Time series data had high missing rates and minimal impact on prediction accuracy, so we ultimately excluded it from our solution.

## Data Preprocessing
### Handling outliers
We removed entries with outlier values (explained in the EDA part) to avoid skewing the model.

### Feature engineering and selection
We used feature engineering to extract more valuable insights and remove duplicate information, and dropped features with too many missing values, redundancy, or strong correlations. The final set of selected features included:

| No. | Feature | Explanation
| --- | --- | --- |
| 1 | `Basic_Demos-Age` | Participant's age |
| 2 | `Basic_Demos-Sex` | Participant's sex |
| 3 | `CGAS-CGAS_Score` | Children's Global Assessment Scale (CGAS) score |
| 4 | `Physical-Height_per_Age` | `Physical-Height`/`Basic_Demos-Age` | 
| 5 | `Physical-Weight_per_Age` | `Physical-Weight`/`Basic_Demos-Age` |
| 6 | `FGC_Zone_Total` | `FGC_CU_Zone`+`FGC_SRL_Zone`+`FGC_SRR_Zone`+`FGC_TL_Zone` |
| 7 | `BIA-BIA_Activity_Level_num` | Activity level |
| 8 | `BIA-BIA_BMR` | Basal Metabolic Rate |
| 9 | `BIA-BIA_DEE` | Daily Energy Expenditure |
| 10 |`BIA-BIA_FFMI` | Fat-Free Mass Index | 
| 11 |`BIA-BIA_FMI` | Fat Mass Index |
| 12 |`BIA-BIA_Frame_num` | Frame size |
| 13 |`BIA-BIA_SMM` | Skeletal Muscle Mass | 
| 14 | `BIA-BIA_TBW` | Total Body Water volume |
| 15 | `SDS-SDS_Total_T` | Sleep Disturbance Scale total score |
| 16 | `PreInt_EduHx-computerinternet_hoursday` | Average hours per day spent on computer/internet usage |
| 17 | `PAQ_Total` | Physical Activity Questionnaire score, combined by `PAQ_A-PAQ_A_Total` and `PAQ_C-PAQ_C_Total`  |

### Imputation
We evaluated two imputation methods: age-based regression and a hybrid approach combining K-Nearest Neighbors (KNN) for continuous variables and Random Forest for categorical variables. Both methods are reasonable depending on the data characteristics.

In this project, most features are closely related to physical development, making linear regression with age a suitable choice for imputing missing values. This approach ensures that the imputed values align with the observed data distribution.

The hybrid method leverages the assumption that continuous features follow local similarity patterns, which KNN can effectively capture, and categorical features are sufficiently correlated with other features, allowing Random Forest to provide accurate predictions.

After testing both methods, age-based regression was selected as it demonstrated better performance for this dataset.

## Modeling and Prediction
We tested three methods: 1) Ridge Regression; 2) Multilayer Perceptron (MLP); 3) Gradient Boosting. Gradient boosting consistently outperformed the other methods with a stronger Quadratic Weighted Kappa (QWK) score, so we stick to this method. We optimized hyperparameters using Optuna and validated results with 5-fold cross-validation.

#Key Takeaways
This competition was not only a test of technical skills but also an exercise in handling real-world data challenges. While proud of our results, we see room for improvement and valuable lessons for future projects. 
- **Feature engineering and selection matters**: Due to significant data noise, carefully removing redundant or irrelevant features can greatly improve the performance of gradient boosting models.
- **Data imbalance is also a limitation**: The target variable (SII) was highly imbalanced, with about 60% of values being 0 and 85% less than 2. This imbalance complicated the task of establishing clear relationships between features and the target variable.
- **Physical data alone may be insufficient**: Physical data alone may not fully explain problematic internet use (PIU). For example, some children with high SII scores displayed strong fitness metrics, which, while unexpected, is reasonable. Including additional data, such as behavioral patterns, may provide more insights.
- **Need for objective measurements**: During the competition, we couldn’t help but question the reliability of using Parent-Child Internet Addiction Test (PCIAT) results as a measure of PIU. Since it’s based on parent-reported answers, it can be subjective and potentially inaccurate. We think it might be possible to develop a more objective way to measure PIU. For example, the feature `PreInt_EduHx_computerinternet_hoursday`, which directly tracks computer usage time, seems like a more reliable indicator.

# Sources
- https://www.kaggle.com/code/laiyunghwei/lb-0-493
- https://www.sciencedirect.com/science/article/pii/S1386505624001047
