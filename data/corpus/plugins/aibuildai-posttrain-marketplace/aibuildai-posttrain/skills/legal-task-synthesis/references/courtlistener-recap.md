# CourtListener and RECAP: U.S. opinions, dockets and a citation checker

CourtListener is the Free Law Project's open database of U.S. court decisions, with bulk files, an API, and the RECAP Archive of federal filings bought on PACER. For task synthesis it is the main U.S. litigation source, and its citation lookup is the cheapest check that a cited case exists.

**Holds**: bulk CSV files for courts, dockets, opinion clusters and opinions, the citation map, parentheticals, oral arguments and financial disclosures [1]. Free Law Project counts "9+ million decisions", oral argument audio and 16,191 judges behind the API [2]. The RECAP Archive holds "tens of millions of PACER documents, including every free opinion in PACER": when someone with the RECAP browser extension buys a docket or PDF on PACER, it "will be added to the RECAP Archive" [3].

**Access**: two routes, and the difference matters.

- **Bulk data** is the route for volume. Files are generated "quarterly on the last day of March, June, September, and December", each a complete snapshot and not a set of changes, as UTF-8 CSV with a header row, written by PostgreSQL's COPY command and served from an S3 bucket [1].
- **The REST API (v4)** is throttled. "By default, authenticated users may make up to 5 requests per minute, 50 requests per hour, and 125 requests per day", on rolling windows, and "Creating more than one account per project, person, or organization is forbidden" [4]. Until 2026-05-07 the default was 5,000 requests an hour; accounts that had made 1,000 or more requests kept the old allowance, members get full access, and an EDU membership is free for academic researchers and students [2].

**Terms**: "Our bulk data files are free of known copyright restrictions", published with the Public Domain Mark [1]. CourtListener's terms page disallows automated readers and was not read for this card. U.S. federal opinions are government works (`us-federal-law.md`); briefs and other filings in RECAP are written by the parties' lawyers, and their status is not addressed by the pages read here.

**Use it for**: tasks whose answer is in the record itself (`labels-from-the-record.md`): the holding a later court attributes to a case, the authority a paragraph cites, a disposition, a procedural history. Dockets with their filings are the realistic core of a litigation document set.

**The citation checker**: the Citation Lookup and Verification API takes a citation or a block of text and looks each case citation up in CourtListener's database, more than 18.1 million citations on the check date (the page shows a live count); the page says "This can be useful as a guardrail to help prevent hallucinated citations" [5]. Per citation it returns 200 (found), 404 (valid form, not in the database), 300 (several matches) or 400 (unknown reporter). Limits: 250 citations per request, 64,000 characters of text per request, and "60 valid citations per minute". "This API does not look up statutes, law journals, id, or supra citations" [5]. It parses with eyecite, an open-source extractor (BSD licence) that recognises full and short case citations, statutory and law-journal citations, `supra` and `id.` [6]. Run eyecite locally against the bulk citation map when the volume exceeds the API's limits.

**Feeds these benchmarks**: U.S. opinions are the source of CaseHOLD, CLERC, LePaRD, the Lawma tasks and CaselawQA (`decontamination-and-splits.md`). `common-pile/caselaw_access_project` in `legal-dataset` already packages about 6.9 million opinions from the Caselaw Access Project and CourtListener.

**Trap**: PACER is not free, so RECAP is a sample of what someone paid for, not the whole docket. PACER charges "$0.10 per page", capped at "$3 per document", with fees waived at "$30 or less" in a quarter, while "Court opinions are always free" [7]. Federal Rule of Civil Procedure 5.2(c) also limits remote access in Social Security and immigration cases to the docket and the court's dispositions, "but not any other part of the case file or the administrative record" [8]. A document set built from RECAP is missing filings for reasons unrelated to the case.

## What to do

1. Take opinions from bulk files or from the packaged copy in `legal-dataset`; use the API for lookups, not for crawling.
2. Record the court, the decision date and the docket number on every opinion, and the bulk snapshot date on the set.
3. Run every model-written answer's case citations through eyecite and the lookup before keeping the item. A 404 is not proof of invention (the database is not complete), so send 404s to review instead of deleting them silently.
4. Read `personal-data-and-terms.md` before using filings that name private people.

## Sources

Every source was read on 2026-10-03.

[1] Free Law Project wiki, "Bulk Legal Data". https://wiki.free.law/c/courtlistener/help/api/bulk-data/bulk-legal-data

[2] Free Law Project, "Full CourtListener Data Access via API Now Included with Membership" (2026-05-07). https://free.law/2026/05/07/api-included-in-memberships/

[3] Free Law Project, "RECAP". https://free.law/recap/

[4] Free Law Project wiki, CourtListener REST API v4 overview, rate limits section. https://wiki.free.law/c/courtlistener/help/api/rest/v4/overview

[5] CourtListener, "Citation Lookup and Verification API". https://www.courtlistener.com/help/api/rest/citation-lookup/

[6] eyecite repository README. https://github.com/freelawproject/eyecite

[7] PACER, "PACER Pricing: How fees work". https://pacer.uscourts.gov/pacer-pricing-how-fees-work

[8] Federal Rule of Civil Procedure 5.2, read in the Legal Information Institute's copy. https://www.law.cornell.edu/rules/frcp/rule_5.2
