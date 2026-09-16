# Fine-Tuning of TransPolymer for Polymer Property Prediction

Competition: neurips-open-polymer-prediction-2025
Rank: #39
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/fine-tuning-of-transpolymer-for-polymer-property-p

# NeurIPS - Open Polymer Prediction 2025: Silver Medal Solution
**Team:** Solo
**Members:** Fernando Villafuerte ([fjoaquin.villafuerte@gmail.com](mailto:fjoaquin.villafuerte@gmail.com))

# Context

- Business Context: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/overview
- Data Context: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data

# Overview of the Approach

##Competition Description##

This competition focused on predicting five key polymer properties from SMILES representations, including:
1. Glass Transition Temperature (Tg, degrees C)
2. Thermal Conductivity (Tc, W/(m*K))
3. Radius of Gyration (Rg, Angstroms)
4. Density (g/cc)
5. Free Fractional Volume (FFV)


##Model and Algorithm Used##

I employed LoRA fine-tuning of the [TransPolymer](https://github.com/ChangwenXu98/TransPolymer) transformer-based language model for property prediction from SMILES strings. LoRA was deployed using PEFT and the HuggingFace Trainer, the implementation of which can be seen in the DownStream.py script in the GitHub repository included under Project Links. 


##Data Preprocessing##

**Source:** Primary and supplementary data provided by competition organizers.

**Augmentation:** Used RDKit for SMILES string augmentation. Employed standard enumeration, kekulization, and stereo enumeration strategies for augmentation (See the data_prep.py script in the linked GitHub repository).

**Splitting:** From the augmented dataset, created five separate pairs of downstream training and test sets for each of the competition's target properties (Tg, Tc, Rg, FFV, and density). The split between test and training data for each set was 0.9/0.1 (See the data_prep.py script in the linked GitHub repository).

**Normalization:** Data for Tc, Rg, FFV, and density were not normalized. For Tg, data were rescaled to Kelvin and both max-value normalization and z-normalization were tested; optimal performance achieved with max normalization (See z-normalization.py and normalization.py in the linked GitHub repository). 

**Format of training and test datasets:**  
    | SMILES         | property |
    |----------------|----------|
    | CC(=O)OC1=CC=... |   0.123 |

## Validation Strategy

As noted above in Data Preprocessing, validation was performed against a test set for each target property, which constituted 10% of the total data for that property. The HuggingFace trainer was set up to train for 10 epochs, with validation performed on the test set at the end of each epoch. Training was programmed to terminate if no improvement was observed in the validation loss after two consecutive epochs. LoRA training parameters used were standard. Training parameters are set in the config_finetune.yaml file and training performed with the DownStream_LoRA.py script included in the linked GitHub repository. 

Effectiveness of the training was assessed by performing inference using each of the five LoRA-fine-tuned model checkpoints for each target property, assessing the Pearson correlation between the predicted and true values for each target property, and producing the corresponding scatter plot. A reference line indicating the ideal case where predicted and true properties are perfectly correlated was plotted for reference. Agreement between the line of perfect correlation and the scatter plot, in addition to a high correlation score, were used to assess the predictive ability of each model checkpoint. See the correlation.ipynb notebook file in the linked GitHub repository for reference. 


# Details of the Submission

## Results

Fine-tuning with standard LoRA parameters produced model checkpoints with reasonable ability to predict Tc, Rg, density, and FFV. Tg, however, presented some issues. Training on Tg data with no normalization produced fine-tuned models without the ability to predict Tg beyond a certain threshold, with poor Pearson correlation and poor agreement with the line of perfect correlation, indicating a low likelihood of accurate prediction of Tg for unseen SMILES data. First rescaling the Tg data to Kelvin and then normalizing it by the largest value across both the training and test data sets substantially improves the predictive ability of the fine-tuned model, as seen in the figure below. 

[Effect of Tg normalization on Pearson correlation]

Rescaling to Kelvin and normalization of Tg data with the largest value across both training and test datasets was then compared to z-normalization of Kelvin-rescaled Tg data. Though the Pearson correlation between predicted and true values was slightly higher for a model trained on z-normalized Tg data, the scatter plot of true and predicted values did not align well with the line of perfect correlation, suggesting that the predictive ability of this model would be poorer than that of a model trained on Tg data normalized by the largest Tg value, as suggested by the figure below. 

[Normalization of Tg by largest value vs. Z-normalization]

- **Leaderboard Placement:** Silver medalist
- **Best Public Score:** 0.068
- **Best Private Score:** 0.087


## Key Insights & Lessons Learned

- LoRA fine-tuning on large transformer models is highly effective for polymer property prediction from SMILES data.
- Careful normalization, especially for Tg, significantly boosts performance.
- Simple architectures with good data preprocessing can perform well. 


## Acknowledgments

- Pre-trained model from [TransPolymer](https://github.com/ChangwenXu98/TransPolymer)
- LoRA methodology inspired by Hu et al. (2021)
- Competition organizers, Notre Dame University, and Kaggle community.


## References

1. Xu, C., Wang, Y., & Barati Farimani, A. (2023). TransPolymer: a Transformer-based language model for polymer property predictions. _npj Computational Materials, 9_(1), 64.
2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.



# Appendix

## Relevant Files

1. **config_finetune.yaml**: Configuration file for LoRA fine-tuning
2. **data_prep.py**: Data augmentation script
2. **DownStream_LoRA.py**: Fine-tuning script
3. **inference.ipynb**: Notebook for running inference from model checkpoints
4. **correlation.ipynb**: Notebook for assessing Pearson correlation between predicted and true test values
5. **normalization.py**: File for normalizing Tg data by largest Tg value
6. **z_normalization.py**: File for z-normalization of Tg data. 

## Setup

Create a python environment:
```bash
python -m venv your_environment_name
```

Clone the repository:
```bash
git clone https://github.com/fvillafu125/polymers_ftw.git
cd polymers_ftw
```

Install dependencies:
```bash
pip install -r requirements.txt
```
Or install packages manually:
```bash
pip install torch rdkit pandas ...
```

## Fine-tuning

1. Edit the training parameters in the config_finetune.yaml file.
2. Run the following command:
```bash
python DownStream_LoRA.py
```
Note that the default loss function for downstream training is MSE.

## Inference

Inference can be performed with the included notebook, inference.ipynb.

## Model Checkpoints

The pre-trained TransPolymer model checkpoint is included in ckpt/pretrain.pt. The LoRA fine-tuned model checkpoints are included in ckpt/neurips.pt.

## Data Visualization

The provided correlation.ipynb notebook allows for visualization of Pearson correlation between actual test and predicted properties. The loss_plot.ipynb notebook allows for plotting training and test loss.
