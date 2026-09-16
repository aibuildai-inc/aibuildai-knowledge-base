# 24th place : Solution summary

Competition: make-data-count-finding-data-references
Rank: #24
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/25th-place-solution-summary

Firstly, I would like to thank Kaggle for hosting such an amazing competition. This was my first Silver medal and I'm grateful for my teammates @chaneyma, @diptyajitdas, @vaibhav486 and @blackberryisbetter, without whom our team would've never reached this far.

Our solution had the following stages:-

- 1st stage : We extract DOI patterns with some filtering on particular article IDs (e.g : 10.1158/, 10.6084/, 10.5256/) based on train data. At this stage, all valid DOI's obtained are set to Primary. Common DOIs are not dealt with at this stage.
- 2nd stage :-
- Extracted potential DOI / accession ID candidates using regex.
- We dealt with DOIs with were more frequent and had more patterns at this stage (with zenodo, dryad, pasta, pangaea, cranfield, usn and icpsr suffixes)
- Used the following regex (apart from SAMN set to Primary) for accession IDs:-
`REGEX_IDS = (
    r"\b(?:"
    r"CHEMBL\d+|"
    r"[E]-\w{4}-\d+|EMPIAR-\d+|"
    r"ENSBTAG\d+|ENSOARG\d+|ENSMMUT\d+|ENS[FPTG]\d{11}(\.\d+)?|"
    r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
    r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|BX\d{6}|KX\d{6}|K0\d{4}|"
    r"CAB\d{6}|"
    r"((NC|NG|NM|NP|NR|NT|NW)_\d+)(\.\d+)?|"
    r"PRJ[DEN][A-Z]\d+|"
    r"SAM[NDE]\d+|"
    r"G(PL|SM|SE|DS)\d+|"
    r"PDB\s?[1-9][A-Z0-9]{3}|HMDB\d+|"
    r"(SR[RPAX]|STH|ERR|DRR|DRX|DRP|ERP|ERX)\d+|"
    r"(BIOMD|MODEL)\d{10}|BMID\d{12}|"
    r"UPI[A-F0-9]{10}|"
    r"CPX-[0-9]+|"
    r"UPI[A-F0-9]{10}|"
    r"RF\d{5}|"
    r"GU\d{6}|"
    r"EBI\-[0-9]+|"
    r"EGAD\d{11}|"
    r"NA\d{5}|"
    r"NSC\d{5}|"
    r"R-[A-Z]{3}-\d+(-\d+)?(\.\d+)?|REACT_\d+(\.\d+)?|"
    r"phs\d{6}(?:\.v\d{1,2}\.p\d{1,2})?|"
    r"EMD-\d{4,5}|"
    r"MTBLS\d+|"
    r"NZ_[A-Z]{2,4}\d+(\.\d+)?|"
    r"PXD\d+|"
    r"A[PMYF]\d{6}|"
    r"M[TNF]\d{6}|"
    r"URS[0-9A-F]{10}(_\d+)?|"
    r"[1-5]\.(?:10|20|30|40|50|60|70|80|90)\.\d{2,4}\.\d{2,4}|"
    r"CVCL_[A-Z0-9]{4}"
    r")\b"
)`
- Post-filtering on both DOI using MDC corpus v4.1 no eupmc v1.
- Used LLM (qwen2.5 32B instruct awq) to classify whether a text is part of body or references
- Used LLM (qwen2.5 7B instruct awq) to extract abstract, title and author information
- Used LLM (qwen 2.5 32B instruct awq) to classify doi's as primary or secondary. We provided separate prompts based on whether the text is from body or from references. We also used logit restriction.
- Merged all predictions together for submission

What didn't work for us:-

- We made many accession ID experiments. Including the PDB strings (e.g : 1kqp) and IDs like P\d{5}, Q\d{5) captured many but contained a lot of false positives. Unfortunately, our accession ID experiments were done in batch of accessions so at one point, it became hard to track which ones actually helped boost the score 😅
- We also tried to restrict the number of FPs we obtained from above using LLM to determine a valid accession ID given surrounding context and threshold the probability to obtain accession IDs more probable to be valid. But the score was almost breakeven compared to the solution without this method, so we ended up not using it.

I am, in all honesty, a little disappointed at the quality of the training set we received for this competition which kept us guessing sometimes on the validity of our submissions without any guidance apart from public LB. Hopefully, I would be able to avoid such competitions in the future.

That being said, it was all-in-all a great learning experience for me, especially on regex and polars.

Thanks to @yeoyunsianggeremie for sharing the first baseline notebook, based on which our solution was made.
Thanks to @aerdem4 for sharing the first notebook with using logit restriction, based on which our solution was made.
Thanks to @mccocoful for sharing the initial regex solution and providing guidance in multiple discussions, based on which our solution was made.
Thanks to @kawchar85 for suggesting focus on prompt engineering.

Let me know if anyone has questions on the above. This is just a first draft of the solution and I haven't covered very single detail currently and plan to clean up the code and post the detailed solution in some time.
