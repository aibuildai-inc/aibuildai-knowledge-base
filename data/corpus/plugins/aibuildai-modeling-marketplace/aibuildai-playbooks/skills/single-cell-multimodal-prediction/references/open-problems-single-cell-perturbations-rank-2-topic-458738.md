# 2nd Place Solution for the Open Problems – Single-Cell Perturbations

Competition: open-problems-single-cell-perturbations
Rank: #2
Source: https://www.kaggle.com/c/open-problems-single-cell-perturbations/discussion/458738

I am thrilled to finally unveil my solution to this competition!
## Context
- https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview
- https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data
## Overview of the approach
My optimal model emerged as a composite of four models, where three were employed as a normalized sum, and the fourth acted as an amplifier, showcasing discernible impact, potentially on labels with alternating signs:

- weight_df1: 0.5 (utilizing std, mean, and clustering sampling, yielding 0.551)
- weight_df2: 0.25 (excluding uncommon elements, resulting in 0.559)
- weight_df3: 0.25 (leveraging clustering sampling, achieving 0.575)

- weight_df4: 0.3 (incorporating mean, random sampling, and excluding std, attaining 0.554)
The resultant model is expressed as the weighted sum of these components: resulted_model = weight_df1 * df1 + weight_df2 * df2 + weight_df3 * df3 + weight_df4 * df4. All models were trained using the same transformer architecture, albeit with varying dimensional models (with the optimal dimension being 128).

**Data preprocessing and feature selection**
To train deep learning models on categorical labels, I employed one-hot encoding transformation. To address high and low bias labels, I utilized target encoding by calculating the mean and standard deviation for each cell type and SM name. During experimentation, I compared models with and without uncommon columns and discovered that including uncommon columns significantly improved the training of encoding layers for mean and standard deviation feature vectors, resulting in noticeable performance enhancement.
**EDA**

In my feature exploration, I performed truncated Singular Value Decomposition (SVD) specifically on the target variables. Despite experimenting with different SVD sizes, the models generated from this approach consistently fell short of replicating the performance observed in the full targets regression. Notably, as part of this analysis, I identified certain targets with significant standard deviation values.

To leverage this insight, I strategically enriched the feature set by introducing the standard deviation as a dedicated feature for each cell type and SM name. This involved concatenating these standard deviation values into a unified vector format (std_cell_type, std_sm_name).

Furthermore, as I delved into the dataset, I conducted a comprehensive examination, revealing a remarkable cleanliness characterized by the absence of NaN values and duplicates. This meticulous exploration not only extended to model training but also encompassed a nuanced analysis of the target variables, contributing to the informed enhancement of the feature set for improved model performance.


**Sampling strategy**
In devising an effective sampling strategy for partitioning the data into training and validation sets, I employed a sophisticated approach rooted in the clusters derived from a K-Means clustering analysis on the target values. The rationale behind this strategy lies in the premise that data points within similar clusters share inherent patterns or characteristics, thus ensuring a more nuanced and representative split.

For each distinct cluster identified through K-Means, I executed the data partitioning using the train_test_split function from the scikit-learn library. This method facilitated a controlled allocation of data points to both the training and validation sets, ensuring that the models were exposed to a diverse yet stratified representation of the data.

The validation percentage, a crucial parameter in this process, was meticulously chosen to strike a balance between model robustness and computational efficiency. By experimenting with validation percentages ranging from 0.1 to 0.2, I systematically assessed the impact on model performance. Ultimately, for the optimal model configuration, a validation percentage of 0.1 was deemed most effective. This decision was guided by a careful consideration of the trade-off between the need for a sufficiently large training set and the importance of a robust validation set to gauge model generalizability.

In summary, this sampling strategy, grounded in cluster-based data splitting, not only acknowledges the underlying structure within the target values but also optimizes the partitioning process to enhance the model's ability to capture diverse patterns and generalize effectively to unseen data.

**Modeling**

This is my best architecture
```python
 class CustomTransformer_v3(nn.Module):  # mean + std
    def __init__(self, num_features, num_labels, d_model=128, num_heads=8, num_layers=6, dropout=0.3):
        super(CustomTransformer_v3, self).__init__()
        self.num_target_encodings = 18211 * 4
        self.num_sparse_features = num_features - self.num_target_encodings

        self.sparse_feature_embedding = nn.Linear(self.num_sparse_features, d_model)
        self.target_encoding_embedding = nn.Linear(self.num_target_encodings, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.concatenation_layer = nn.Linear(2 * d_model, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, activation=nn.GELU(),
                                       batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        sparse_features = x[:, :self.num_sparse_features]
        target_encodings = x[:, self.num_sparse_features:]

        sparse_features = self.sparse_feature_embedding(sparse_features)
        target_encodings = self.target_encoding_embedding(target_encodings)

        combined_features = torch.cat((sparse_features, target_encodings), dim=1)
        combined_features = self.concatenation_layer(combined_features)
        combined_features = self.norm(combined_features)

        x = self.transformer(combined_features)
        x = self.norm(x)

        x = self.fc(x)
        return x

```
I modularized my model into two distinct segments to efficiently handle the complexity of the input data:

Sparse Feature Encoding:

For encoding sparse features, I utilized an embedding layer tailored to handle the sparsity inherent in certain feature types.
Considering the nature of sparse features, I initially explored utilizing the nn.Embedding layer. However, due to computational constraints on my laptop GPU, I opted for an alternative approach using a linear layer (nn.Linear) to convert sparse features into a dense representation.
Target Encoding Feature Encoding (Dense Features):

Concurrently, I addressed the encoding of target encodings, which are inherently dense features.
To accomplish this, I employed a separate linear layer (nn.Linear) designed specifically for target encodings, ensuring an effective transformation into a meaningful latent space.
Following these individual encodings, I concatenated the resulting embeddings into a unified latent vector, fostering a comprehensive representation of both sparse and dense feature information. To further enhance the model's capacity to capture nuanced patterns, I employed normalization (nn.LayerNorm) on the concatenated features.

Considering the computational challenges posed by sparse feature embedding using nn.Embedding, I strategically utilized nn.Linear for efficient processing on my laptop GPU.

In addition, I introduced a nuanced approach for encoding dense features by breaking them into four distinct encoding layers, each dedicated to capturing nuanced patterns related to mean and standard deviation for both 'sm_name' and 'cell_type'.

Furthermore, the model architecture leveraged the state-of-the-art Lion optimizer, recognized for its efficacy in optimizing transformer-based models. The specific model implementation, as exemplified in the CustomTransformer_v3 class, showcases a transformer architecture with multiple layers, heads, and dropout for optimal learning.

In summary, this modularized and intricately designed model not only optimizes computational efficiency but also demonstrates a keen understanding of the unique characteristics of sparse and dense features, contributing to the overall effectiveness of the learning process.


**Hyperparameters**
The model training regimen was characterized by a judicious selection of hyperparameters, featuring an initial learning rate of 1e-5 and a weight decay of 1e-4, the latter serving as an additional regularization mechanism, particularly for dropout layers within the model architecture. This meticulous choice of hyperparameters aimed at striking a balance between effective training and prevention of overfitting.

To dynamically adjust the learning rate during training, a learning rate scheduler of the ReduceLROnPlateau type was employed. The scheduler, configured with a mode of "min" and a reduction factor of 0.9999, adeptly adapted the learning rate in response to plateaus in model performance, ensuring an efficient convergence towards optimal results. The patience parameter, set to 500, determined the number of epochs with no improvement before triggering a reduction in the learning rate.

The training process spanned 20,000 epochs, incorporating an early stopping mechanism triggered after 5,000 epochs of stagnation. This comprehensive training strategy underscored a thoughtful balance between fine-tuning the model's parameters and preventing overfitting, contributing to the robustness and efficiency of the learning process.

**Loss function**

While Mean Absolute Error (MAE) and Mean Squared Error (MSE) individually prove effective for gauging bias or variance, it's crucial to recognize their distinct roles in assessing model performance. To strike a balance between these metrics, I opted for Huber loss, amalgamating the strengths of both MAE and MSE. Despite utilizing the Mrrmse metric, the ultimate model selection hinged on the performance demonstrated by the optimal loss function on the validation set.

## Preventing overfitting
In addressing the challenges posed by the limited size of the dataset, a critical consideration was ensuring the model's ability to effectively regularize. To achieve this, I incorporated Dropout layers within the transformer architecture and applied weight decay for L2 penalty on the model weights. Additionally, to mitigate the impact of exploding gradients, I implemented gradient norm clipping with a maximum norm of 1.

## Validation Strategy
In pursuit of an optimal model configuration, I adopted a robust validation strategy by training with different seeds and employing k-fold cross-validation. The diverse set of performance metrics for each fold included:

- Achieving a score of 0.551 through the incorporation of standard deviation, mean, and clustering sampling.
- Attaining a score of 0.559 by excluding uncommon elements from the dataset.
- Realizing a score of 0.575 through clustering sampling.
- Securing a score of 0.554 by integrating mean, random sampling, and excluding standard deviation from the features.
This validation approach not only facilitated the identification of the best-performing model but also guided the fine-tuning of hyperparameters to optimize overall model performance.
## Sources
- https://github.com/lucidrains/lion-pytorch
- https://www.kaggle.com/code/ayushs9020/understanding-the-competition-open-problems
- https://www.kaggle.com/code/alexandervc/op2-eda-baseline-s
- https://github.com/Eliorkalfon/single_cell_pb
