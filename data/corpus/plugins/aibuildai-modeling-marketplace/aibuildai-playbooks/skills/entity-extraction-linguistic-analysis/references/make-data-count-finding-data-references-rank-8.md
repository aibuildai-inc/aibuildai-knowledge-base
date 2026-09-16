# 8th place solution

Competition: make-data-count-finding-data-references
Rank: #8
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/8th-place-solution

## Brief Summary
### DOIs
- step1. Use regex r'10\s*\.\s*\d{4,9}\s*/\s*\S+' to extract candidate dois from text.
- step2. Filter dois using Data Corpus.
- step3. Classify doi type using a fine-tuned Qwen2.5-32b-instruct model based on: paper's abstract, data section, doi context.（LB score increased by finetuning）

### Accession ids
- step1. Use regex to extract candidate accs from text.
- step2. Filter accs using Europe PMC dataset.
- step3. Classify acc type using Qwen2.5-32b-instruct model based on: acc context, acc table information(caption,footnote,headers,3 lines before and after the acc), paper's abstract, data section.（LB score increased by prompt engineering）


acc regex:
```python
REGEX_ACC_IDS = (
        r"\b(?:"
        r"E-[A-Z][A-Z][A-Z][A-Z]-[0-9]+|"  #arrayexpress
        r"(?:BIOMD|MODEL)\d{10}|BMID\d{12}|"  #biomodels
        r"SAM[NED][A-Z]?[0-9]+|"  #biosample
        r"(?:[0-9][a-zA-Z0-9]{4}[0-9]{2})|(?:[1-4]\.[0-9]+\.[0-9]+\.[0-9]+)|" #cath
        r"CVCL_[a-zA-Z0-9]{4}|"  #Cellosaurus
        r"CHEMBL\d+|"  #chembl
        r"CPX-[0-9]+|"  #complexportal
        r"SRS\d{6}|"  #metagenomics
        r"EMPIAR-\d{5}|" #empiar
        r"[eE][nN][sS][a-zA-Z]*[gGptPT]\d{11}|" #ensembl
        # r"HGNC:[0-9]+|" #hgnc
        r"(?:HPA|hpa|CAB|cab)\d{6}|"  # HPA
        r"[iI][pP][rR]\d{6}|" #interpro
        r"PF(AM)?\d{5}|"  #pfam 
        r"(R)?PXD\d{6}|"   #pride, pxd
        r"([a-nr-zA-NR-Z][0-9][a-zA-Z][a-zA-Z0-9][a-zA-Z0-9][0-9]|[opqOPQ][0-9][a-zA-Z0-9]{3}[0-9])([-][0-9]+)?|"
        r"UPI[A-F0-9]{10}|" #uniprot
        r"(AC|AP|NC|NG|NM|NP|NR|NT|NW|NZ|XM|XP|XR|YP|ZP|NS)_([A-Z]{4})*\d{6}(\d{3})?([.][0-9]+)?|" #refseq
        r"[rs]s[0-9][0-9]?[0-9]?[0-9]?[0-9]?[0-9]?[0-9]?[0-9]?[0-9]?|" #refsnp
        r"PRJ[DEN][A-Z][0-9]+|" #bioproject
        r"G(PL|SM|SE|DS)[0-9][0-9]+|"  #geo
        r"EPI\d{5}[0-9]+|EPI\d{6}-[0-9]+|EPI_ISL_\d{5}[0-9]+|" #GISAID
        r"[0-9][a-zA-Z0-9]{3}|"  #pdb
        r"phs\d{6}(?:\.v\d{1,2}\.p\d{1,2})?|" #dbgap
        # r"ORPHA:[0-9]+|ORPHA [0-9]+|"#orphadata
        # r"[0-9][0-9][0-9][0-9]\-[0-9][0-9][0-9][0-9][0-9][0-9]\-[0-9][0-9]|"#eudract
        # r"NCT[0-0][0-9][0-9][0-9][0-9][0-9][0-9][0-9]|"#nct
        # r"TF[0-9][0-9][0-9][0-9][0-9][0-9]|"#treefam
        # r"[0-9][0-9][0-9][0-9][0-9][0-9]|"#omim
        # r"HPSI[0-9][0-9][0-9][0-9](i|pf)\-[a-z]+_[0-9]+|"#hipsci
        # r"[A-Z][A-Z]+i[0-9][0-9][0-9]\-[A-Z]|"#ebisc
        # r"URS[0-9A-Z]+_[0-9]+|" #rnacentral
        # r"[Rr][Hh][Ee][Aa]\:[1-9][0-9][0-9][0-9][0-9]|[Rr][Hh][Ee][Aa]\:[1-9][0-9]+|[Rr][Hh][Ee][Aa]\:[1-9][0-9]+|"#rhea
        # r"MTBLS[0-9]+|"#metabolights
        # r"MINT\-[0-9]+|IM\-[0-9]+|"#mint
        # r"EBI\-[0-9]+|"#intact
        # r"EGAD[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]|"#ega
        # r"EFO_[0-9]+|EFO:[0-9]+|"#efomapping_dict
        # r"CHEBI:[0-9]+|"#chebi
        # r"S\-BIAD[0-9]+|"#bia
        # r"AF\-[OPQ][0-9][A-Z0-9][A-Z0-9][A-Z0-9][0-9]+\-F[0-9]|AF\-[A-NR-Z][0-9]([A-Z][A-Z0-9][A-Z0-9][0-9])+\-F[0-9]|"#alphafold
        r"R\-HSA\-[0-9]+|"#reactome
        r"RF[0-9][0-9][0-9][0-9][0-9]|" #rfam
        r"([0-9]+\.\-\.\-\.\-|[0-9]+\.[0-9]+\.\-\.\-|[0-9]+\.[0-9]+\.[0-9]+\.\-|[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)|BTO: ?([0-9][0-9][0-9][0-9][0-9][0-9][0-9])|"#brenda
        r"HG0[0-4][0-9][0-9][0-9]|(NA|GM)[0-2][0-9][0-9][0-9][0-9]|"#1000genomes
        r"TI[0-9]+|(?:(E|D|S)RA|(E|D|S)RZ|(E|D|S)RR|(E|D|S)RX|ERS|(E|D|S)RP)\d{5}(\d+)?|[A-Z]\d{5}|[A-Z]{2}\d{6}|[A-RT-Z][A-Z]{3}[S]?\d{8}([0-9])?|[A-Z]{3}\d{5}" #gen
        r")"     
    )  
```
