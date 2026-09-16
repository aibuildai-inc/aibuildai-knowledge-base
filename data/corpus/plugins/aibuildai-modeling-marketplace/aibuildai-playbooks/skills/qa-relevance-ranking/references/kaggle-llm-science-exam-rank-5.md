# 5th place solution: Llama 2 70B meets Sparse & Dense Retrievals from Own parsed wikipedia dataset

Competition: kaggle-llm-science-exam
Rank: #5
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446293

Thank you to the organizers and congratulations to all the participants.

There are really lots of things that can be done from dataset preparation, model training to post-processing (how to utilize trained LLM model), and it was a very tough competition for me. But I enjoyed the competition overall!
I would like to thank my team members @zaburo, @qhapaq49, @charmq, @flowlight for their hard work too. 

The figure shows our retrieval to the model overall pipeline.



~~We wrote dataset & model part at first. The rest part will be updated later.~~ --> Write-up Updated!

## Short summary

 - We successfully run Llama 70B model on kaggle notebook
 - We used BM-25 based sparse retrieval (using [pyserini](https://github.com/castorini/pyserini) library whose backend is [Apache Lucene](https://lucene.apache.org/)) to search all the wikipedia pages.
 - We parsed wikipedia dataset by ourselves to cover most of the pages with number information.


## Dataset creation

### Wikipedia dataset

In the early stage of the competition when I checked the given train.csv and tried to answer manually, I felt this competition is a “retrieval” competition rather than a model training competition. We also noticed some pages or numbers are missing in the publicly available dataset. So I decided to parse our wikipedia dump manually. 

Below are comparison tables of datasets:

| Dataset | Parser | Page | Text info |
| --- | --- | --- | --- |
| Kaggle published dataset by @jjinho <br/> [jjinho/wikipedia-20230701](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) | `wikitextparser` | Some pages are dropped | All templates are dropped.<br/>All the “\n” are dropped (difficult to read by human). |
| Huggingface dataset<br/> - [wikipedia](https://huggingface.co/datasets/wikipedia) <br/> - [graelo/wikipedia](https://huggingface.co/datasets/graelo/wikipedia) | `mwparserfromhell` | Better coverage | All templates are dropped.<br/> `<math>`, `<ref>`, `<table>` tags are dropped. |
| Our dataset | `wikitextparser` + custom template processing | Better coverage | `{val}` and `{math}` templates text are kept (Other templates are dropped).<br/> Keep `<math>` tag, but `<ref>`, `<table>` tags are dropped. |

Huggingface's Wikipedia dataset covers most of the pages but numbers are missing, it seems due to the `mwparserfromhell` parser issue. So, I replaced the parser with `wikitextparser` from the [graelo/wikipedia](https://huggingface.co/datasets/graelo/wikipedia) code base (There was a helpful instruction on how to run the code at the bottom of the dataset description!).
I believe our dataset covers most of the pages with numbers or math expression information.

### QA dataset creation

Many datasets were shared publically during the competition, but we also tried to make it by ourselves as well. Below is the prompt for ChatGPT 3.5 that I used to generate the QA dataset.

```
system_content = """
Forget all the previous instruction and rigorously follow the rule specified by the user.
You are a professional scientist's assistant.
"""

user_content_template_qa = Template(
    """
Please consider 5 choices question and answer of the following TEXT.
The purpose of this question is to check respondent's deep science understanding of the TEXT.
We assume this question is for professional scientists, so consider super difficult question.
You can ask very detailed question, for example check specific sentence's understanding.
It is good practice to randomly choose specific sentence from given TEXT, and make QA based on this specific sentence.
You must make QA based on the fact written in the TEXT.
You may create wrong answers based on the correct answer's information, by modifying some parts of the correct answer.
Your response must be in following format, don't write any other information. 
You must not include "new line" in each Q), 1), 2), 3), 4), 5), and A):
Q) `question text comes here`
1) `answer candidate 1`
2) `answer candidate 2`
3) `answer candidate 3`
4) `answer candidate 4`
5) `answer candidate 5`
A) `answer`

where only 1 `answer candidate` is the correct answer and other 4 choices must be wrong answer.
Note1: I want to make the question very difficult, so please make wrong answer to be not trivial incorrect.
Note2: The answer candidates should be long sentences around 30 words, not the single word.
Note3: `answer` must be 1, 2, 3, 4 or 5. `answer` must not contain any other words.
Note4: Example of the question are "What is ...", "Which of the following statements ...", "What did `the person` do",
and "What was ...".
Note5: Question should be science, technology, engineering and mathematics related topic. 
If the given TEXT is completely difference from science, then just output "skip" instead of QA.


Here is an example of your response, please consider this kind of difficulty when you create Q&A:
Q) Which of the following statements accurately describes the impact of Modified Newtonian Dynamics (MOND) on the observed "missing baryonic mass" discrepancy in galaxy clusters?"
1) MOND is a theory that reduces the observed missing baryonic mass in galaxy clusters by postulating the existence of a new form of matter called "fuzzy dark matter."
2) MOND is a theory that increases the discrepancy between the observed missing baryonic mass in galaxy clusters and the measured velocity dispersions from a factor of around 10 to a factor of about 20.
3) MOND is a theory that explains the missing baryonic mass in galaxy clusters that was previously considered dark matter by demonstrating that the mass is in the form of neutrinos and axions.
4) MOND is a theory that reduces the discrepancy between the observed missing baryonic mass in galaxy clusters and the measured velocity dispersions from a factor of around 10 to a factor of about 2.
5) MOND is a theory that eliminates the observed missing baryonic mass in galaxy clusters by imposing a new mathematical formulation of gravity that does not require the existence of dark matter.
A) 4

Let's start. Here is TEXT: $title\n$text
"""
)

```
I only give ChatGPT 3.5 an abstract and 1 paragraph from specific page as `$text`.

Since we can achieve very high scores like 0.995 for train.csv at the end of the competition. We used our generated 1000 dataset for local validation.

## Retrieval

### Sparse retrieval

I'm not professional to the NLP field. I surveyed existing research, especially related to retrieval-based LLMs. And [ORQA](https://aclanthology.org/P19-1612/) paper affected our approach, it reported that “On datasets where the questioner already knows the answer, a traditional IR system such as BM25 is sufficient.”. Since ChatGPT 3.5 was given the sentence to generate QA, we thought the same applies in this competition too.
So we read [BERTserini](https://aclanthology.org/N19-4013/) paper and decided to use [pyserini](https://github.com/castorini/pyserini) library, which is Python wrapper of [anserini](https://github.com/castorini/anserini) that uses Apache Lucene in the backend. 

[This notebook](https://www.kaggle.com/code/strifonov/pyserini-scibert) by @strifonov was really helpful to run pyserini on kaggle notebook (thank you!).

Apache Lucene was very efficient and convenient. It was easy to index all the wikipedia dataset and search through BM25 score.
We splitted each wikipedia page to the paragraph (by “\n\n”). It only required several hours to index in total of 74M records. In the inference time, searching is accelerated by multi-threading, 200 searches are finished in 2 min or so.



### Dense retrieval

We created 2 types of dense retrieval.

**1. All the Wikipedia contents split by sentence wise block**

We used [instructor-xl](https://huggingface.co/hkunlp/instructor-xl) for the embedding calculation and used [faiss](https://github.com/facebookresearch/faiss) to create index.
The original index is too big (300GB) to run on memory, so we quantized the index (300GB → 10GB) for this approach.

**2. STEM 270k contents split by paragraph wise block**

We used [bge-large-en](https://huggingface.co/BAAI/bge-large-en) for the embedding calculation.
And only used same pages published in [STEM270k](https://www.kaggle.com/datasets/mbanaei/all-paraphs-parsed-expanded) dataset by @mbanaei.

We added “title” information in each block, so that we can obtain correct search result even when “it/he/she” is used to refer to the title.

In total, we used 3 types of retrievals, and combined them as a context to input to the LLM.

## Model (written by @zaburo)

Our team used Llama-2 7B, 13B, 70B and [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1). Since Mistral 7B was achieving better scores compared to Llama-2 of the same size, we used Mistral 7B and Llama-70B in our final submission. Based on the scores from MMLU and the results of our preliminary experiments, we used models without Instruction Tuning, such as Llama-2-70b-hf and Mistral-7B-v0.1.

For each model, we input the problem in the following format and had the model output the probability that the token corresponding to "▁A", "▁B", "▁C", "▁D", "▁E" comes immediately after this sentence. This yielded decent performance even with models without any fine-tuning.
```
{context_0}

Question: {prompt}
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
Answer:
```

Fine-tuning was done using QLoRA. We incorporated bitsandbytes' 4bit quantization and PEFT's LoRAModel into our training pipeline. The hyperparameters of QLoRA were almost exactly the ones in QLoRA's paper, except for the batch size which had to be lowered due to long input and the GPU memory constraint.

We expected that running Llama 70B on Kaggle Notebook would be challenging, so we started working on it from the early stages of the competition. We registered quantized weights (less than 40GB) for each layer as a dataset, and performed inference layer-by-layer. By applying xformers' `memory_efficient_attention` to the attention layers, we were able to keep memory consumption linear even with long contexts, leaving plenty of room in the GPU memory. We used only approximately 6GB of GPU memory for inference.

LLM can considerably change inference results depending on the order of choices, so we used a test time augmentation strategy in which we rotated the order of choices 5 times to average this effect. This TTA was most effective with pretrained models without any retrieval, and although the effect drastically decreased with retrievals or QLoRA, it remained effective to the end.

Naively carrying out TTA inference would increase the execution time 5 times, but rather than inferring the entire set 5 times, we concatenated all patterns of choice order like `{context} {Q} {A B C D E} {B C D E A} … {E A B C D}` and set the `attention_mask` appropriately to reduce the inference of the long common part to only once. By applying this technique, a 70B model with 5 times TTA worked within time for the full test set, and using a single 70B + BM25 Top2 + 5 times TTA resulted in a Private 0.925 score.



## Validation

We noticed MAP@3 is a bit unstable metric, and can fluctuate a bit. Below are our strategy not to overfit to the public LB.

 - We used 3 datasets, train.csv (200), @yalickj [dataset](https://www.kaggle.com/datasets/yalickj/dataset-wiki-new-1) (300), and our own dataset (1000, described above), for the local validation.
    - We did not use these data in the training, even in the final submission.
 - We tried to run various experiments with local dataset validation. We did not submit methods which do not work well in our local dataset. I think it prevents us from adopting various methods that overfit the public LB.
    - For example, a custom query like [this](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/442595#2454966) did not work well in our experiment, so we did not adopt this.
    - Also, we tried embedding model fine tuning, but it did not work well in our local dataset, so we did not submit this too.


## Final Pipeline - Multi Stage Inference (written by @charmq)

We noticed that putting multiple contexts in a prompt at the same time improves the performance of the model. However, because simultaneous input of contexts increases inference time, we used a multi-stage inference pipeline in which easy problems are answered by small models (mistral-7B) and difficult problems are answered by large models (llama2-70B). The final submission consists of three stages as follows:

**1st stage: ensemble of mistral-7B models**
mistral-7B with BM25 top2 contexts + mistral-7B with instructor and bge contexts
Inference on all data

**2nd stage: llama2-70B**
llama2-70B with BM25 top2, instructor and bge contexts
Inference on bottom 40% data with low confidence in the 1st stage

**3rd stage llama2-70B**
llama2-70B with bge contexts (2~3 times longer than bge contexts used in the 1st and 2nd stage)
Inference on bottom 5% data with low confidence in the 1st and 2nd stage

This multi stage pipeline achieved private:0.926, public:0.928.
