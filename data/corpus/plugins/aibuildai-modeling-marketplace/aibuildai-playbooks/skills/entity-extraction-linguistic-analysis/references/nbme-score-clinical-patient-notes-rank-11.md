# 11th Solution: Trust CV

Competition: nbme-score-clinical-patient-notes
Rank: #11
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322804

[source code available](https://github.com/Zacchaeus14/nbme)
Firstly, I'd like to express my gratitude to Kaggle and NBME for organizing such a fantastic competition and congratulate all the winners. I began working on this project approximately 10 days before the deadline, initially aiming for a silver medal; achieving gold was thrilling. My solution, grounded in pure engineering rather than any secret tricks, stemmed from the [top-rated baseline](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train). Due to time constraints, I had to rely entirely on my CV strategy, which was adapted from the baseline (5-fold GroupKFold).

# Tokenizer
The baseline encountered minor issues with tokenizers, which I identified and resolved throughout the process.

## FN in labels
The `create_label` function in the baseline inaccurately labeled sentences ending with a single key phrase. Post-tokenization, these phrases were marked as negative. To address this, I modified the following code:

from
```
                if start_idx == -1:
                    start_idx = end_idx
```
to
```
                if start_idx == -1:
                    start_idx = end_idx - 1
```
## Trimmed offsets
As discussed, roBERTa tokenizers handle offsets with trimmed spaces by default. To address this, we need to load them with `trim_offsets=False`. However, I discovered that for some models (e.g., Electra, Ernie, Albert, Funnel, MPNet), offsets remain trimmed even when `trim_offsets` is set to `False`. I created a small function to manually correct this:
```
def fix_offsets(offset_mapping, text):
    n = len(offset_mapping)
    if n == 0:
        return []
    if n == 1:
        return offset_mapping[0]
    re = [offset_mapping[0]]
    last_e = 0
    for i in range(1, n-1):
        s, e = offset_mapping[i]
        if text[s] != ' ' and s != last_e:
            s = last_e
        last_e = e
        re.append((s, e))
    re.append(offset_mapping[n-1])
    return re
```
## Misalinged annotations
In the baseline's `get_results` function, there's a line that adds 1 to all indices:
```
result = np.where(char_prob > th)[0] + 1
```
This assumes that all tokens start from spaces, which can cause the first character to be lost. To fix this, I removed the "add 1" and instead stripped spaces from both sides:
```
def my_get_results(char_logits, texts, th=0):
    results = []
    for i, char_prob in enumerate(char_logits):
        result = np.where(char_prob > th)[0]
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        temp = []
        for r in result:
            s, e = min(r), max(r)
            while texts[i][s] == ' ':
                s += 1
            while texts[i][e] == ' ':
                e -= 1
            temp.append(f"{s} {e+1}")
        result = temp
        result = ";".join(result)
        results.append(result)
    return results
```
# Model
My models are straightforward, trained using the Hugging Face trainer. With limited time for hyperparameter tuning, I adopted the following settings and kept all other defaults:
* For large models: `lr=1e-5, bs=4, weight_decay=0, warmup_ratio=0.2, epoch=10`. For models that couldn't fit into 24GB RAM (xlarge ones), I used `bs=2, lr=5e-6`.
* For base models: `lr=3e-5, bs=16, weight_decay=0, warmup_ratio=0.2, epoch=10`.

I combined 14 models to create the pseudolabels, with weights determined by Optuna within the range of (0, 1).
| Model | CV | Weight |
| --- | --- | --- |
| 1. roberta-large | 0.8823 | 0.269 |
| 2. muppet-roberta-large | 0.8817 | 0.450 |
| 3. roberta-base | 0.8742 | 0.016 |
| 4. bart-large | 0.8782 | 0.147 |
| 5. bart-large-mnli | 0.8786 | 0.291 |
| 6. deberta-large | 0.8844 | 0.760 |
| 7. deberta-base | 0.8753 | 0.373 |
| 8. deberta-large-mnli | 0.8845 | 0.919 |
| 9. deberta-v3-large | 0.8833 | 0.584 |
| 10. ernie-large | 0.8794 | 0.059 |
| 11. electra-large-discriminator | 0.8809 | 0.020 |
| 12. funnel-large | 0.8797 | 0.953 |
| 13. deberta-v3-base | 0.8773 | 0.491 |
| 14. roberta-large-mnli | 0.8819 | 0.086 |
 
The ensemble was achieved by merely averaging character logits, ensuring that models were not blended across folds to avoid leakage. Each fold model labeled one-fifth of the unlabeled data (approximately 110k). Following pseudolabeling, I trained the initial 14 models for one epoch, in addition to two more deberta-xlarge models for the final prediction. The process was largely unchanged.
| Model | CV |
| --- | --- |
| 1. roberta-large | 0.8912 |
| 2. muppet-roberta-large | 0.8910 |
| 3. roberta-base | 0.8872 |
| 4. bart-large | 0.8893 |
| 5. bart-large-mnli | 0.8901 |
| 6. deberta-large | 0.8930 |
| 7. deberta-base | 0.8911 |
| 8. deberta-large-mnli | 0.8927 |
| 9. deberta-v3-large | 0.8928 |
| 10. ernie-large | 0.8916 |
| 11. electra-large-discriminator | 0.8917 |
| 12. funnel-large | 0.8896 |
| 13. deberta-v3-base | 0.8895 |
| 14. roberta-large-mnli | 0.8911 |
| 15. deberta-xlarge | 0.8948 |
| 16. deberta-v2-xlarge-mnli | 0.8940 |

## Final submissions
### Submission 1: CV 0.8967 Public LB 0.890 Private LB 0.892
| Model | Weight |
| --- | --- |
| 8. deberta-large-mnli | 0.240 |
| 9. deberta-v3-large | 0.609 |
| 15. deberta-xlarge | 0.968 |
| 16. deberta-v2-xlarge-mnli | 0.188 |
### Submission 2: CV 0.8961 Public LB 0.889 Private LB 0.891
| Model | Weight |
| --- | --- |
| 1. roberta-large | 0.339 |
| 7. deberta-base | 0.692 |
| 8. deberta-large-mnli | 0.306 |
| 9. deberta-v3-large | 0.821 |
| 10. ernie-large | 0.645 |

# Conclusion
As I heavily relied on my CV, I felt disheartened when the CV/LB correlation deteriorated after my CV surpassed 0.89. Nevertheless, I had no choice but to trust it, as I had limited submission quota to probe the leaderboard. Special thanks to Prof. Ross at NYU Shanghai for providing me with the necessary GPU resources, which enabled me to train numerous models within a short time frame.
