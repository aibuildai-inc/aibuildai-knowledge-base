# Building the document set for a long legal task

Read when the target is work over a set of files: reviewing a data room, preparing a chronology from a docket, drafting from a term sheet and precedents. The documents are the hard part of synthesizing such a task, because the answer depends on facts spread across them.

There are three ways to get a document set, and they trade realism against a known answer.

## Three ways

| Way | What you get | What it costs | Published example |
|---|---|---|---|
| Real documents | real drafting, real length, real mess | the answer is unknown until someone reads them; files are missing or redacted; real people are named | CUAD and MAUD annotate real EDGAR agreements; CLERC uses real opinions (`labels-from-the-record.md`) |
| Fully written for the task | every fact is chosen, so the answer is known; no real person appears | documents can read as too clean and too consistent; writing them well needs lawyers | Harvey LAB; APEX-Agents |
| Real documents with planted changes | real texture, plus a list of exactly what was changed | the planted items can stand out from their surroundings | LePhantomCite (legal briefs); FLAWS (scientific papers) |

**Fully written.** Harvey says LAB began "with real client matters handled by practicing lawyers", which were broken "into the discrete tasks an associate would typically be delegated to complete", and that a matter's files "mix key and peripheral files and embed issues across multiple documents" [1]. Its documents are synthetic, generated in batches under lawyers' guidance and review, with synthetic people, companies and facts required (`legal-dataset`: `harveyai__harvey-labs.md`). APEX-Agents had professionals, corporate lawyers among them, work in teams with assigned roles for 5 to 10 days and produce each world's files "from scratch", averaging 166 files a world [2].

**Planted changes.** LePhantomCite takes 1,000 excerpts from federal appellate briefs and 300 model-written holdings and injects errors into half of them, in five categories: a citation that does not exist (altered volume, reporter or page), a real case name on the wrong citation, a wrong pin cite, a misquote, and a case cited for a proposition it does not support [3]. FLAWS does the same to scientific papers, inserting claim-invalidating errors with a model to make 713 paper-error pairs [4]. In both, the record of what was inserted is the answer key.

**Generated text that cannot be memorised.** BLT builds hierarchical sections by "repeated application of the logical form A ⇒ B" with defined terms that "are nonces that are not real words but are pronounceable", so that no model has seen the text in training [5]. The same idea applies to a contract: change the parties, amounts and dates of a real form and the model cannot answer from memory of the filing.

## Design points

- **Peripheral files.** A set that contains only the relevant documents trains nothing about finding them. Include files that are plausible and beside the point [1].
- **Issues that span documents.** The facts that decide a criterion should sit in different files: a cap in the main agreement, an exclusion in an amendment, a date in an email [1].
- **File formats the work uses.** Public matter-file benchmarks ship Word documents, spreadsheets, emails, slides and text, and expect Word or spreadsheet deliverables (`legal-dataset`: Harvey LAB and RedlineBench cards). A task that hands over clean plain text skips the reading step.
- **Length.** Applied Compute reports that in one benchmark the 90th-percentile task has nearly 100,000 tokens of source documents and the largest over 200,000 [6]. Decide the length range from the target and the model's context, and vary it.
- **A facts ledger.** For written or altered sets, keep one table of every fact that matters (parties, dates, amounts, defined terms, which document states each) and every planted issue. Generate the documents from the ledger, and write the rubric from the ledger (`writing-rubrics.md`).
- **Consistency checks.** Anything not deliberately planted must agree across documents. Extract names, dates and amounts from the finished files and compare them with the ledger.
- **Hide the key.** The ledger and rubric must not sit where the agent can read them. LAB's system prompt forbids the agent to read the task file because the rubric sits beside the documents (`legal-dataset` card); a training environment should keep them out of the sandbox altogether.

## What to do

1. Choose the way by what you can verify. If no lawyer will read the real documents, a planted or written set with a ledger gives a rubric you can trust.
2. Start a planted set from real documents in the public sources (`sec-edgar.md`, `courtlistener-recap.md`), outside any benchmark's corpus.
3. Use invented names for people and companies in anything written for the task (`personal-data-and-terms.md`).
4. Plant issues of the kinds the practice area actually turns on, and record each one with the document and location.
5. Run a model on the finished task before keeping it: if it passes every criterion without opening the files, the instruction leaks the answer.

## Sources

Every source was read on 2026-10-03.

[1] Harvey, "Open-Sourcing Harvey's Long Horizon Legal Agent Benchmark". https://www.harvey.ai/blog/introducing-harveys-legal-agent-benchmark

[2] "APEX-Agents". https://arxiv.org/html/2601.14242

[3] Liu, Stammbach and Henderson, "Who Checks the Citations? Benchmarking Legal Hallucination Detection". https://arxiv.org/html/2606.21155

[4] Xi, Rao, Payan and Shah, "FLAWS: A Benchmark for Error Identification and Localization in Scientific Papers". https://arxiv.org/abs/2511.21843

[5] Blair-Stanek, Holzenberger and Van Durme, "BLT: Can Large Language Models Handle Basic Legal Text?". https://arxiv.org/html/2311.09693

[6] Applied Compute, "Training a State-of-the-Art Legal Agent with Harvey". https://www.appliedcompute.com/case-studies/harvey
