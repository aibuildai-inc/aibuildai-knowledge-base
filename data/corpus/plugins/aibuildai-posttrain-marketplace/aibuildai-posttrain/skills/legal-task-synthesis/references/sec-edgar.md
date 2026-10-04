# SEC EDGAR: real commercial agreements as task material

EDGAR is the U.S. Securities and Exchange Commission's filing system. Public companies attach their material contracts and merger agreements to filings as exhibits, which makes EDGAR the largest open body of real, negotiated commercial agreements: credit agreements, employment agreements, leases, licences, supply agreements, merger agreements. For task synthesis it is the main source of transactional documents.

**Holds**: exhibits to 10-K, 10-Q and 8-K filings. Regulation S-K Item 601 defines exhibit 10, material contracts, as "Every contract not made in the ordinary course of business that is material to the registrant", and exhibit 2 as "Any material plan of acquisition, disposition, reorganization, readjustment, succession, liquidation, or arrangement" [4]. Full-text search covers "all EDGAR filings submitted electronically since 2001", including "all attachments (such as exhibits)" [3].

**Access**: free, no account. The SEC asks for two things: "Current max request rate: 10 requests/second", and "Please declare your user agent in request headers", in the form `Sample Company Name AdminContact@<sample company domain>.com` [1]. It "reserves the right to limit request rates to preserve fair access for all users" [1]. Index files sit under `/Archives/edgar/full-index/` and `/Archives/edgar/daily-index/` [1]. The `data.sec.gov` JSON APIs (filing history per company, XBRL facts) "do not require any authentication or API keys", and two archives, `submissions.zip` and `companyfacts.zip`, are "recompiled nightly" [2].

**Terms**: the SEC pages read for this card state access rules, not a licence for the exhibits. The agreements are written by the filing companies and their counsel, not by the government. `chenghao/sec-material-contracts` republishes 1,141,632 Exhibit 10 documents under CC BY-SA 4.0 (see its card in `legal-dataset`); that is the republisher's licence statement, not the SEC's.

**Use it for**: clause finding and term extraction, issue spotting against a checklist, comparing a draft with an executed precedent, summarising a long agreement, and as the realistic core of a document set (`document-sets.md`). Filter by exhibit description to pick agreement types.

**Already packaged**: `chenghao/sec-material-contracts` (1994 to the first quarter of 2025, raw HTML and SGML), `pile-of-law/pile-of-law` configs `edgar` and `atticus_contracts`, and `theatticusproject/cuad` (510 labelled contracts). All three have cards in `legal-dataset`. Start from these before crawling.

**Feeds these benchmarks**: CUAD's 510 contracts were "filed on SEC EDGAR", and LegalBench's `cuad_*` and `contract_qa` tasks, and part of LegalBench-RAG, are built from them (`legal-dataset`: `theatticusproject__cuad-qa.md`; [5]). MAUD is built from public-company merger agreements, the kind of document filed as exhibit 2, and feeds LegalBench's `maud_*` tasks. A task set synthesized from EDGAR exhibits can therefore contain a benchmark's own documents under new questions. See `decontamination-and-splits.md`.

**Trap**: exhibits are filed with parts missing, by rule. A registrant may redact "specific provisions or terms" of a material contract if it "customarily and actually treats that information as private or confidential" and the information is not material (Item 601(b)(10)(iv); (b)(2)(ii) says the same for acquisition agreements); schedules may be omitted if they "do not contain information material to an investment or voting decision" (601(a)(5)); and personal details such as "bank account numbers, social security numbers, home addresses" may be redacted (601(a)(6)) [4]. A synthesized question whose answer sits in a redacted term or an omitted schedule has no answer in the document. Check each item's answer span exists before keeping it (`checking-synthesized-tasks.md`).

## What to do

1. Use a packaged copy for volume and the live system for anything after the copy's cut-off date. Filings made after a benchmark's release cannot be among its documents.
2. Record the accession URL, the form type, the exhibit number and the filing date on every document. The filing date is the date of the law the parties drafted against.
3. Strip HTML and SGML wrappers before generating tasks from the text (the `legal-dataset` card for `chenghao/sec-material-contracts` gives the measured formats).
4. Keep each company's filings on one side of any split: amendments and restatements repeat most of the original agreement.

## Sources

Every source was read on 2026-10-03.

[1] SEC, "Accessing EDGAR Data". https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data

[2] SEC, "EDGAR Application Programming Interfaces". https://www.sec.gov/search-filings/edgar-application-programming-interfaces

[3] SEC, EDGAR full-text search FAQ. https://www.sec.gov/edgar/search/efts-faq.html

[4] 17 CFR 229.601 (Regulation S-K, Item 601), read in the Legal Information Institute's copy. https://www.law.cornell.edu/cfr/text/17/229.601

[5] "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain": built from PrivacyQA, CUAD, MAUD and ContractNLI. https://arxiv.org/html/2408.10343
