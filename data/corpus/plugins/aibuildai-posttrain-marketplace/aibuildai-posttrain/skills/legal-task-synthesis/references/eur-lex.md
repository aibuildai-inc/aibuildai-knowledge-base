# EUR-Lex and Cellar: EU law in 24 languages

EUR-Lex is the European Union's portal to its legislation, case law and Official Journal. The same texts, with metadata, are stored in Cellar, the Publications Office's repository. For task synthesis it is the main multilingual source: one act exists in parallel in the Union's official languages, and many acts come with an official human-written summary.

**Holds**: EU treaties, regulations, directives, decisions, Court of Justice case law and the Official Journal. Cellar is "the common data repository of the Publications Office of the European Union", storing "multilingual publications and metadata" as machine-readable data [1]. EUR-Lex-Sum, built from this source, pairs legal acts with human-written summaries in up to 24 languages (`legal-dataset`: `dennlinger__eur-lex-sum.md`).

**Access**: Cellar offers a SPARQL interface over its metadata, web services, and RSS and Atom feeds [1]. The endpoint addresses are in the Publications Office's developer pages, which were not read for this card.

**Terms**: Commission Decision 2011/833/EU on the reuse of Commission documents provides that "All documents shall be available for reuse" for "commercial or non-commercial purposes" and "without charge", with conditions that may include "the obligation for the reuser to acknowledge the source of the documents" and "the obligation not to distort the original meaning or message of the documents" [2]. How EUR-Lex applies this to its own content is stated in its legal notice, which loads its text with scripts and could not be read for this card, so the licence wording there is not quoted. Datasets built from EUR-Lex do not agree on what it is: MultiEURLEX's card says the original EU data is CC BY 4.0, and EUR-Lex-Sum's card says CC BY-SA 4.0 (both in `legal-dataset`). Read the notice yourself before redistributing.

**Use it for**: summarisation with official summaries as targets; multilingual versions of one task from one act; classification and retrieval using the EuroVoc subject labels; tasks on how a directive's articles apply to a fact pattern.

**Feeds these benchmarks**: MultiEURLEX, EUR-Lex-Sum, and the EU-law task in LexGLUE (all carded in `legal-dataset`). A summarisation task synthesized from an act that is in EUR-Lex-Sum's test split recreates a test item.

**Trap**: "do not distort the original meaning" is a reuse condition [2], and model-written paraphrases, translations and simplified versions of an act are exactly the kind of derived text that can distort it. Keep the original article text in the task and mark any model-written restatement as such.

## What to do

1. Identify every act by its CELEX number and store it, with the language and the date of the version used.
2. For multilingual tasks, take the official language versions; do not machine-translate a version that already exists.
3. Exclude the acts in the test splits of MultiEURLEX and EUR-Lex-Sum when either will be reported (`decontamination-and-splits.md`).

## Sources

Every source was read on 2026-10-03. The EUR-Lex legal notice (https://eur-lex.europa.eu/content/legal-notice/legal-notice.html) and data-reuse help page returned only navigation to this skill's reader and were not read.

[1] Publications Office of the European Union, "Cellar". https://op.europa.eu/en/web/cellar

[2] Commission Decision 2011/833/EU of 12 December 2011 on the reuse of Commission documents, Articles 4 and 6. https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32011D0833
