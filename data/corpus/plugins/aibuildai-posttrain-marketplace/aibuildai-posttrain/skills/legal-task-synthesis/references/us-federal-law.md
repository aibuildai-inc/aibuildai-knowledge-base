# U.S. federal statutes, regulations and bills

The text of U.S. federal law is free to copy and comes with machine-readable bulk files and dated versions. It is the source for statute and regulation tasks: applying a rule to facts, finding the governing provision, tracking what changed between two dates.

**Holds**:

- **govinfo** (Government Publishing Office): bulk XML for Congressional bill text, status and summaries (113th Congress onward), the Code of Federal Regulations (1996 to present), the Federal Register (2000 to present) and the Statutes at Large (1789 to present), among others; sitemaps "to help developers crawl the entirety of the public dataset"; and a link service that resolves citations to the CFR, bills, the Federal Register and the U.S. Code [1].
- **eCFR**: the continuously updated Code of Federal Regulations, with history. "The Timeline view provides an overview of all changes to the content you are viewing since 2017", any past date can be viewed, and two dates can be compared, with removed text "on a red background with strikethrough" and added text "on a green background" [2].
- **Federal Register**: daily rules, proposed rules and notices, through an API that needs no key: "FederalRegister.gov APIs do not require API keys" [3].

**Access**: free. The govinfo API needs an api.data.gov key; its bulk repository and sitemaps do not [1].

**Terms**: "Copyright protection under this title is not available for any work of the United States Government" (17 U.S.C. 105(a)) [4]. For state and federal law generally, the Supreme Court held in Georgia v. Public.Resource.Org, Inc. (decided 2020-04-27) that judges and legislators cannot be authors, for copyright purposes, of the works they produce in their official capacity, and that the annotations in Georgia's official code, made for the legislature by its commission, were not copyrightable either [5]. Commentary a private publisher writes on its own account is a different matter and is not settled by these sources.

**Use it for**: grounded question answering where the provision is supplied; "which provision governs" retrieval tasks; change tasks built from two dated versions of one section (the eCFR comparison gives the diff, so the answer is known by construction); summarising a bill, where the official summary is a human-written target.

**Feeds these benchmarks**: BillSum pairs U.S. Congressional and California bills with their official summaries; the Nemotron legal pretraining set has an `eCFR` config; Pile of Law has a `state_codes` config; HousingQA's statutes are 2021 state housing law (all carded in `legal-dataset`). LegalBench's `sara_*` tasks use U.S. tax statutes (`legal-evaluation`).

**Trap**: the convenient copy is not the official one, and every copy has a date. FederalRegister.gov "is not an official legal edition of the Federal Register, and does not replace the official print version or the official electronic version on GPO's govinfo.gov" [3]. A statute task without the date of the text teaches law of unknown vintage, and a model may answer from the version it memorised instead of the version supplied (`labels-from-the-record.md`, the BLT amended-text task).

## What to do

1. Pin a capture date for every provision and store it on the task. For regulations, use eCFR's dated versions so the date is exact.
2. Prefer tasks that supply the provision text. A task that asks for the rule from memory is graded against a date the model cannot see.
3. Build change tasks from version pairs: the before and after text give the gold answer without a model writing it.
4. Record the citation in a form eyecite can parse, so a later check can find statutory citations in model output (`courtlistener-recap.md`).

## Sources

Every source was read on 2026-10-03. The U.S. Code download page at uscode.house.gov returned a maintenance notice on that date and was not read.

[1] govinfo, "Developer Hub". https://www.govinfo.gov/developers

[2] eCFR, "eCFR Changes Through Time". https://www.ecfr.gov/reader-aids/using-ecfr/ecfr-changes-through-time

[3] Federal Register, API documentation v1. https://www.federalregister.gov/developers/documentation/api/v1

[4] 17 U.S.C. 105, read in the Legal Information Institute's copy. https://www.law.cornell.edu/uscode/text/17/105

[5] Georgia v. Public.Resource.Org, Inc., 590 U.S. ___ (2020), slip opinion. https://www.supremecourt.gov/opinions/19pdf/18-1150_7m58.pdf
