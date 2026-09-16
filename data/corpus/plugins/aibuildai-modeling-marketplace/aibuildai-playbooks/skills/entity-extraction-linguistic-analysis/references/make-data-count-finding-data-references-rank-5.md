# 5th Place - by standing on the shoulders of Giants

Competition: make-data-count-finding-data-references
Rank: #5
Source: https://www.kaggle.com/c/make-data-count-finding-data-references/writeups/5th-place-by-standing-on-the-shoulders-of-giants

Thank you to the hosts and participants for such a fun and interesting competition. 

Special shoutout to @rdmpage who's initial understanding, blog posts and engagement was extremely valuable.

# Competing in the Fog

I worked very hard on some solutions that would find all possible accessions but that was not the point of the competition. It was helpful in getting used to data and my own learning, however the competition truly began when the Giant was released.

# Finding the giants

@keakohv left the following comment [here](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/600330#3273757)
> Neither, actually. The process for DOI-only LB 0.27 goes like this:
> - Take Data Citations Corpus v3 version (2025-02-01) publication-DOI citation pairs provided by DataCite.
> - Keep only citation pairs where publication is in this competition's set of articles.
> - Check if the DOI is actually present in the PDF or XML of the article, if not, then remove (similar to the "Missing" type citations in labels).
> - Create features, including metadata from the Citations Corpus, for example the publisher, the journal, the title of the dataset etc.
> - Train a classifier on the features to predict Primary/Secondary.
> 
> I made these DOI-only experiments about a month ago but have not explored further. My goal then was to understand what might be the logic behind the host's labelling method. I was quite sure, still am, that they at least partly used DataCite or some similar source for DOIs.

The hosts of the competition left the following comment on the Welcome Post. [link to comment](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/584337#3224495)

> Hi @rdmpage -- answering your questions here and noting that we read through all of the other threads/comments! We conducted human labeling on the online version and PDFs for the citation type. The data mentions were extracted from available data citation sources including MDC data citations corpus and Europe PMC.  Papers have been provided to help participants access to the content faster. They have been pulled from Open Access sources based on what publishers have made available (where available & whatever format). Neither are more relevant than the other (that XML was not used for labeling doesn’t necessarily mean the PDFs are more relevant for participants). We know there are idiosyncrasies in this large dataset. The very fact of them demonstrates why this competition is so essential to science - so we can get the most comprehensive and accurate picture of linkages between data and papers. For the competition itself, all participants are scored based on the same evaluation dataset. As we and @inversion have noted, we will post more information soon about this last point!

Here we see the host clearly explain where the accessions have come from and how they are classified.
1. all accessions are contained within [Europe PMC](https://europepmc.org/pub/databases/pmc/) and [DCCv3](https://zenodo.org/records/14897662)
1. the labellers used only PDF when determining whether Primary or Secondary

# Climbing the giants

So we download all the accessions and form a table of `article_id` and `dataset_id`. Using this on the `train_labels.csv` we can then detect accessions that haven't been included in the competition. The most complicated of these families to filter correctly was `gen.csv` - in the end I just filtered all accessions that started with ones that were hit in the training but not in `train_labels.csv`

From this we get a solution that public LB of `0.82` and private LB of `0.70` that I have posted [here](https://www.kaggle.com/code/mccocoful/mdc-cpu-submission)

# The view from the top

Qwen needs the context the human labellers had in order to classify the accessions. I concentrated on the XML as it is already structured and contains a great deal of information. We first identify the container where the ID appears, then if that container has a tag-id we find all the places that tag-id is held. This helps when the ID is only in a table or in the reference section. Another trick was to only include the actual row containing the ID if a table was found.

At this point with relatively little time left in the competition I decided to concentrate on DOI and SAMN as they were actually classifiable and they really needed to be classified (50/50 primary/secondary) whereas the majority of the accession ID are just secondary. I collected all the metadata for these IDs and supplied these in the context as well.

## Example Prompt
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
### Core Instructions ###
* Inspect WINDOW taking particular interest in ID, given below.
* The ID is specifically a data citation - that relates to data held in an open-access repository.
* Determine whether the WINDOW context holds evidence that the WINDOW authors are responsible for the ID held in the public repository.
* After thinking, give your final answer using the rubric:
    * Owner: the WINDOW authors have some sort of ownership around ID.
    * User: the data has be re-used/referenced/compared in the WINDOW.
    * None: there is no evidence to determine ownership.
* When reviewing the METADATA remember: 
    * The METADATA is collected from several sources and hence has various formats for authour names and dates.
    * The most important thing is finding the overlap of WINDOW author(s) with the ID author(s); usually one author overlap is enough to assume Owner.
* The final answer should be wrapped in \boxed{} containing only User, Owner or None.

# ID
https://doi.org/10.17882/47142

# METADATA
## ID METADATA
[Title]: A global bio-optical database derived from Biogeochemical Argo float measurements within the layer of interest for field and remote ocean color applications
[Authors]: Organelli, Emanuele; Barbieux, Marie; Claustre, Herv; Schmechtig, Catherine; Poteau, Antoine; Bricaud, Annick; Uitz, Julia; Dortenzio, Fabrizio; Dallolmo, Giorgio
## WINDOW METADATA
[Title]: Assessing the Variability in the Relationship Between the Particulate Backscattering Coefficient and the Chlorophyll <i>a</i> Concentration From a Global BiogeochemicalArgo Database
[Authors]: Marie Barbieux; Julia Uitz; Annick Bricaud; Emanuele Organelli; Antoine Poteau; Catherine Schmechtig; Bernard Gentili; Grigor Obolensky; Edouard Leymarie; Christophe Penkerc'h; Fabrizio D'Ortenzio; Herv Claustre
[Date]: 2018-2

# WINDOW
## Paragraph
<p xmlns="http://www.tei-c.org/ns/1.0" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><s>Sherbrooke, Canada) are acknowledged for useful comments and fruitful discussion.</s><s>We also thank the International Argo Program and the CORIOLIS project that contribute to make the data freely and publicly available.</s><s>Data referring to <ref type="bibr">(Organelli et al., 2016a)</ref> (doi:10.17882/47142)</s><s>and <ref target="#b8" type="bibr">(Barbieux et al., 2017)</ref> (doi: 10.17882/49388) are freely available on SEANOE.</s></p>

## References Condensed
<biblstruct xmlns="http://www.tei-c.org/ns/1.0" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xml:id="b100">

<monogr>
<title level="j">SEANOE</title>
<imprint>
<date type="published" when="2016">2016</date>
</imprint>
</monogr>
<note type="raw_reference">Organelli, E., M. Barbieux, H. Claustre, C. Schmechtig, A. Poteau, A. Bricaud, J. Uitz, F. D'Ortenzio, and G. Dall'Olmo (2016a), A global bio-optical database derived from Biogeochemical Argo float measurements within the layer of interest for field and remote ocean colour applications, SEANOE, doi:10.17882/47142.</note>
</biblstruct><|im_end|>
<|im_start|>assistant
```

## Inference

Now with the context I infer with Qwen2.5-32B-Instruct-AWQ and Qwen3-32B-AWQ extracting the answer from `\\boxed{}` and then ensemble via majority voting with my CPU submission.

## Final Trick

this is a bit of a weird one - but in the `train_labels.csv` I noticed I was getting some FP for some accessions when a DOI is included. These DOI's were special in that they appeared in EuropePMC text-mined-terms as well as DCC - so I filtered ACC when these types of DOI are matched.

```python
sub_ensemble = ensemble.select('article_id', 'dataset_id', pl.Series('type', final_types)).unique(['article_id', 'dataset_id'])

corpus = ( 
    get_corpus()
)

sub_w_families = sub_ensemble.with_columns(pkey=pl.col('article_id').str.to_lowercase()).join(corpus, ['pkey', 'dataset_id'])

ids_to_drop = (
    sub_w_families.group_by('article_id')
    .agg('dataset_id', 'family')
    .with_columns(pl.col('family').list.unique().alias('uf'))
    .filter(pl.col('uf').list.len()>1)
    .filter(pl.col('uf').list.contains('mdc_priority'))
    .filter(pl.col('uf').list.set_difference({"mdc", "mdc_priority"}).list.len()>0)
    .drop('uf')
    .explode(['dataset_id', 'family'])
    .filter(~pl.col('family').str.contains(r"(:?mdc|gen|pfam)"))
    .select('article_id', 'dataset_id')
)

sub_ensemble = sub_ensemble.join(ids_to_drop, pkeys, how='anti')
```

this did not improve LB but I trusted CV. CV was relatively stable for me throughout the competition.


# Final Words

As this competition was all about citations this post would not be complete without a reference section !

Page, R. (2024). Problems with the DataCite Data Citation Corpus [https://doi.org/10.59350/t80g1-xys37](https://doi.org/10.59350/t80g1-xys37)

Page, R. (2024). The Data Citation Corpus revisited [https://doi.org/10.59350/wvwva-v7125](https://doi.org/10.59350/wvwva-v7125)

DataCite, & Make Data Count. (2024). Data Citation Corpus Data File (v1.1) [Data set]. DataCite. [https://doi.org/10.5281/zenodo.11216814](https://doi.org/10.5281/zenodo.11216814)
