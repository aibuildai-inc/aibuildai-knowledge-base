# 7th Place Solution for the Open Problems – Single-Cell Perturbations

Competition: open-problems-single-cell-perturbations
Rank: #7
Source: https://www.kaggle.com/c/open-problems-single-cell-perturbations/discussion/459623

Thank the organizers for hosting this interesting competition and congrats to the Winners.
It was a big surprise for me to see my resolution finally achieve 7th place. 

## Context
https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview
https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data

## Overview of the approach

My approach has two major steps: i) learn embeddings for all cell types, small molecular, and genes,  ii) use embedding as features to train a deep learning model to predict the target while accounting for overfitting. Throughout the competition, I only used a fully connected network with three layers. My best submission is also based on a single FC model.

### Learning embeddings

In this step, the goal is to learn a specific embedding for each cell type, molecular, and gene. I was strongly inspired by [this paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-01977-6), where the authors used deep tensor factorization to learn a dense, information-rich representation for cell type, experimental assay, and genomic position.  

Basically, I used the same approach but played around with the model architectures and parameters, such as the number of latent factors, and the number of dimensions of the network, as well as how to combine the features (e.g., concatenate vs. additive). 

In the end, my mode is the following:
```python
class DeepTensorFactorization(torch.nn.Module):
    def __init__(self, 
                 cell_types, 
                 compounds, 
                 genes, 
                 n_cell_type_factors: int=4, 
                 n_compounds_factors: int=16, 
                 n_gene_factors: int=128,
                 n_hiddens: int=2048,
                 dropout: float=0.1):
        super().__init__()
        
        self.cell_types = cell_types
        self.compounds = compounds
        self.genes = genes
        
        self.n_cell_types = len(cell_types)
        self.n_compounds = len(compounds)
        self.n_genes = len(genes)
        
        self.n_cell_type_factors = n_cell_type_factors
        self.n_compounds_factors = n_compounds_factors
        self.n_gene_factors = n_gene_factors
        
        self.cell_type_embedding = torch.nn.Embedding(self.n_cell_types, self.n_cell_type_factors)
        self.compound_embedding = torch.nn.Embedding(self.n_compounds, self.n_compounds_factors)
        self.gene_embedding = torch.nn.Embedding(self.n_genes, self.n_gene_factors)
        
        self.n_hiddens = n_hiddens
        self.dropout = dropout
        self.n_factors = n_cell_type_factors + n_compounds_factors + n_gene_factors
        
        self.model = nn.Sequential(nn.Linear(self.n_factors, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, 1))
        
    def forward(self, cell_type_indices, compound_indices, gene_indices):
        cell_type_vec = self.cell_type_embedding(cell_type_indices)
        compound_vec = self.compound_embedding(compound_indices)
        gene_vec = self.gene_embedding(gene_indices)
        
        x = torch.concat([cell_type_vec, compound_vec, gene_vec], dim=1)
        x = self.model(x)
        
        return x
```
To train this model, I used all the data from `de_train.parquet` and converted the table as follows:
```python
# convert to long df
df = pd.read_parquet('de_train.parquet')
df = df.sort_values(['cell_type', 'sm_name'])
df = df.drop(['sm_lincs_id', 'SMILES', 'control'], axis=1)
df = pd.melt(df, id_vars=['cell_type', 'sm_name'], var_name='gene', value_name='target')
```
The training data looks like:


Here, I trained the model with 100 epochs without validation, because I will only the embedding layer for Step 2.

To check if the model learns meaningful embedding for cell types, molecules, and genes, I also visualized the embedding with UMAP. For example, below is the 2D UMAP of gene embeddings:



By eyeballing, it looks like there are some structures for the genes.

### Predicting target  

Once I obtained the embeddings, I trained another model to predict the target, and this model was also used to generate the final submission. Again, I used a FC network as follows:
```python
class PerturbNet(torch.nn.Module):
    def __init__(self, 
                 n_input: int=148,
                 n_hiddens: int=2048,
                 dropout: float=0.5):
        super().__init__()
        
        self.n_input = n_input
        self.n_hiddens = n_hiddens
        self.dropout = dropout

        self.model = nn.Sequential(nn.Linear(self.n_input, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, 1))
        
    def forward(self, x):
        x = self.model(x)
        
        return x
```

To prevent overfitting, I used each of the cell types as validation data, which means my model was based on 4-fold cross-validation. For the compounds, I followed the notebook here to select the private test compounds for validation:

```python
for key, cell_type in cell_type_names.items():
    print(cell_type)
    
    # split data for training and validation, here we used the private test compounds for validation
    df_train = df[(df['cell_type'] != key) | ~df['sm_lincs_id'].isin(privte_ids)]
    df_valid = df[(df['cell_type'] == key) & df['sm_lincs_id'].isin(privte_ids)]
    
    df_train = df_train.sort_values(['cell_type', 'sm_name'])
    df_valid = df_valid.sort_values('sm_name')
    
    df_train = convert_to_long_df(df_train)
    df_valid = convert_to_long_df(df_valid)
    
    df_train.to_csv(f'../../results/PerturbNet/splited_data/train_{cell_type}.csv')
    df_valid.to_csv(f'../../results/PerturbNet/splited_data/valid_{cell_type}.csv')
```

For model training, I used the following strategies:
```python
    # Setup loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5)
```

I trained one model by using each of the cell types as validation data, and for the final submission, I just averaged the predictions.

## What didn't work for me

During the competition, I spent a lot of time including the features based on prior knowledge, such as single-cell data, molecular structure embedding, and gene embedding for other large models, such as geneFormer. However, they all didn't work out. So my final model was just based on the features learned from Step 1.


## Source
https://github.com/lzj1769/7th_place_solution_Single-Cell-Perturbations
https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-01977-6
https://github.com/jmschrei/avocado
