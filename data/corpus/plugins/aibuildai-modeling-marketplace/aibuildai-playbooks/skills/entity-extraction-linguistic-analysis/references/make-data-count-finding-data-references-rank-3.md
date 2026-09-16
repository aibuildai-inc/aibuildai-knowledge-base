# 3rd Place Solution

Competition: make-data-count-finding-data-references
Rank: #3
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/3rd-place-solution

Thanks to Kaggle and the hosts for hosting the competition, it proved much more challenging than we initially anticipated.

Our solution is comprised of mainly two steps:

- Citation retrieval via regex + postprocessing to remove FPs
- Type classification via a 6-fold Deberta-v3 ensemble + heuristics 


### Citation Retrieval

Similarly to other teams, our retrieval process here is pretty straightforward. We used simple regexes and rules to extract candidate citations for both DOI and accession IDs and then compared against the 2024 DataCite annual dump and a few additional entries from the mined terms corpus by EUPMC. Additionally, we extracted paper metadata information from the Crossref 2025 dump as well.

This simple method was enough to get almost 100% recall for the DOIs while having minimal false positives. While we tried different ways, this proved to be the most reliable in the end. Besides that, we also applied some small heuristics based on what we observed in the training set and LB probing.


### Postprocessing and heuristics

If we could not find a citation in the PDF parsed text, we would also search the XML for both accessions and DOIs. Otherwise, we simply ignored the XML and relied only on PDF.

We also had to ignore a few different accession and DOIs that were always false positives, such as:
- `figshare` DOIs
- `GCA_` accessions, which was a major source of false positives
- `HGNC:*`, `rs*` and a few different types

Finally, if multiple accessions and DOIs are found in the same article we found that ignoring the DOI gave a boost on LB. 


### Dataset type classification

At the very beginning we knew that type classification was one of the most important parts of this competition. The reason is that, since we use the F1 score as the metric, a FN or a FP in the retrieval part count only as 1 single error. However, if we get a TP sample but misclassify its type we would get 2 errors: 1 FN for the missing correct type and 1 FP for the wrong predicted type.

Our solution here consists in mainly two steps: a 6-fold Deberta-v3 ensemble and some heuristics.

### Heuristics

Similarly to other teams, we first used a few rules that proved to work on both LB and CV:

- All `dryad` DOIs and `SAMN` accessions are primary
- If a DOI title or authors list are similar to the paper (via string edit distance), the dataset is primary
- If an accession is cited more than 5 times in EUPMC it is secondary (simple probability check)
- If a DOI has 5 or more citations, it is secondary (simple probability check)

The remaining citations were then classified using a Deberta-v3 Large ensemble.

### Training details

We created a 6-fold `StratifiedGroupKFold` (stratified by `type` and grouped by `article_id`) and trained one deberta-v3 large on each of those fold using a binary classification setup (classification head on top). The following features were used to generate the training prompts:
- The context around the citations on the paper (1k characters)
- The first 500 characters of the article text
- Paper and dataset titles whenever available

We also tried to fine-tune many LLMs like Qwen2.5 7B with classification heads but this proved very unstable throughout the training. The debertas were not only faster but also achieved better results. 

We found two very important tricks to stabilize the training: adding gradient clipping and model/weight EMA. The latter is a very simple technique that consists of having a trainable model via SGD/Adam and a frozen counterpart, which is updated via direct averaging (weighted) of both model's weights after each training step. Using `transformers` lib, it can be easily implemented by inheriting the `Trainer` class like the following:

```python
from typing import Dict
import torch
from ema_pytorch import EMA
import copy

class EMATrainer(Trainer):
    def __init__(self, ema_decay=0.9995, ema_update_every=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize EMA after model is set
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.ema = None
        
    def _setup_ema(self):
        if self.ema is None:
            self.ema = EMA(
                self.model,
                beta=self.ema_decay,
                update_every=self.ema_update_every,
                update_after_step=50  # Start EMA after 50 steps
            )
    
    def training_step(self, model, inputs, num_items_in_batch = None):
        """Override training step to include EMA updates"""
        if self.ema is None:
            self._setup_ema()
            
        # Perform normal training step
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        
        # Update EMA after each step
        self.ema.update()
        
        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluate using EMA model"""
        if self.ema is not None:
            # Temporarily use EMA model for evaluation
            original_model = self.model
            self.model = self.ema.ema_model
            
            results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            
            # Restore original model
            self.model = original_model
            return results
        else:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save EMA models"""
        # Save EMA model
        ema_output_dir = f"{output_dir}/ema_model"
        if self.ema is not None and output_dir is not None:
            self.ema.ema_model.save_pretrained(ema_output_dir)
        else:
            self.model.save_pretrained(ema_output_dir)            
```

These two techniques, albeit simple, made the training curves much smoother despite the small and noisy training dataset.

At inference time, we ran 2 DeBERTa models at a time (one in each T4 GPU) and ensembled the 6 predictions via simple averaging.
This 6-fold ensemble was significantly better than using a 0-shot model in our testing. An additional advantage of having a classification head was the ability to tune threshold at will (either directly or by quantiles).



### Shoutouts


Finally we would like to give a shoutout to the following people:

- @mccocoful for sharing so much stuff since the beginning of the competition, even when being isolated at the top of the LB for so long
- @rdmpage for going through the trouble of annotating and sharing a (better) version of the train labels
- @keakohv for sharing the data citation corpus idea and making the competition a bit more of a fair ground for all teams who were not using it before

In my opinion, people like you guys are the reason Kaggle is so great, and why I was able to learn so much along the years. I don't know of a place where people are so willing to share and help each other out in such a competitive environment.
