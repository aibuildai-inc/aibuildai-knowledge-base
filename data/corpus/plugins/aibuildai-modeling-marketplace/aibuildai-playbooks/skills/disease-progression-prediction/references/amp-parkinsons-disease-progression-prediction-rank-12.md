# 12th Place Solution

Competition: amp-parkinsons-disease-progression-prediction
Rank: #12
Source: https://www.kaggle.com/c/amp-parkinsons-disease-progression-prediction/discussion/411394

Before anything else, I want to express my deep gratitude to my team members @vitalykudelya @yukisawamura @yukisawamura @salaryman who contributed their time, efforts, and expertise in this competition.  Next, we'd like to extend our thanks to the Kaggle community and the hosts. Thank you for providing a challenging dataset and a great learning opportunity.

# **Overview of our solution**
[Overview]

Our solution is based on user grouping, trend calculation for each group, user group prediction (classification), and trend assignment for each group.
You can check our solution code [here](https://www.kaggle.com/maruichi01/11th-place-solution). It takes super long time to be scored beacuse my pipeline is not good😓

## **1. Grouping**
During our exploratory data analysis (EDA), we noticed significant differences in the target trends among users based on the presence or absence of Medication information (across all 'visit_month' data, regardless of being 'On' or 'Off').  It is basis of our pipeline.

## **2. Trend Calculation**
Based on the groups' information, we created three types of trends: those without Medication information, those with Medication information, and overall trends. These trends were adopted from the '[Only_Trend](https://www.kaggle.com/code/vitalykudelya/only-trends)' and '[Protein Shift](https://www.kaggle.com/code/vitalykudelya/p05060-protein-npx-groups-trend-silver-medal)' notebooks. Particularly, the protein 'P05060' greatly contributed to improving the Public LB score.

## **3. Feature Engineering + Group Classification**
We implemented binary classification using LightGBM to predict whether a user belongs to a group with Medication information or not. The prediction was made based on 'visit_month' related information, and Protein and Peptide data.
For our final submission, to increase the robustness of the model, we utilized 10-fold cross-validation (CV) and random seed averaging.

## **4. Using Binary Classification Results for Trend Mapping**
Based on the results of the binary classification, we mapped the trends of each group. For our final submission, if the prediction from binary classification was above 0.75, we assumed it belonged to the group with Medication information. If it was below 0.15, we assumed it belonged to the group without Medication information. For all other cases, we used the overall trend.

As the dataset for this competition was small, both CV and LB scores were unstable. We struggled until the end to decide which submission to choose. If you have any questions or feedback, please feel free to comment. We look forward to learning with all of you in the community.
