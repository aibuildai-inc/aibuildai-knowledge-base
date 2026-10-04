# United Kingdom: legislation.gov.uk and Find Case Law

The UK publishes its legislation and its court judgments through The National Archives, under two different licences. Legislation is open for any use. Judgments are open to read and republish, but bulk programmatic use needs a separate licence. That difference decides what a synthesis pipeline may do.

**Holds**: legislation.gov.uk carries UK legislation with stable URIs for each item and each section. Find Case Law carries court and tribunal judgments.

**Access**: legislation is served in machine-readable form from the same URI as the web page: append `/data.xml` for XML, `/data.rdf` for RDF or `/data.feed` for an Atom feed [1][2].

**Terms, legislation**: content is available under the Open Government Licence v3.0, except where otherwise stated [1].

**Terms, judgments**: Find Case Law uses the Open Justice Licence, version 2.0 [3][4]. It lets you "copy, publish, distribute and transmit the Information" and "exploit the Information commercially", with the attribution "Contains information licensed under the Open Justice - Licence v2.0" [4]. But it excludes "computational analysis of the Information (including indexing by search engines)" [4]. The guidance page says so plainly: "The Open Justice Licence does not permit computational analysis", which covers "programmatic searching in bulk across Find Case Law records to identify, extract or enrich contents", and "you will need to apply for a licence to perform computational analysis" [3]. The licence is also "not a data sharing agreement for personal data", and users must "comply with any judicial decision that restricts the use of personal data within the Information" [4].

**Use it for**: statute tasks from legislation.gov.uk under the open licence. Judgment-based tasks only after the computational-analysis licence is in hand.

**Trap**: a judgment page that anyone can read and republish is not thereby a page a pipeline may mine. Generating tasks from Find Case Law in bulk is the activity the Open Justice Licence carves out [3][4]. Apply for the licence first, and record the grant with the dataset.

## What to do

1. For UK statute tasks, fetch `/data.xml` per section and store the URI and the retrieval date.
2. For UK case law, either obtain the computational-analysis licence or use a few judgments selected and read by hand.
3. Keep the two sources' licences separate in the dataset's documentation; a mixed UK set inherits the stricter terms for its judgment-derived items.

## Sources

Every source was read on 2026-10-03.

[1] legislation.gov.uk, "Developer zone". https://www.legislation.gov.uk/developer

[2] legislation.gov.uk, "Formats". https://www.legislation.gov.uk/developer/formats

[3] The National Archives, "Re-use Find Case Law records". https://caselaw.nationalarchives.gov.uk/re-use-find-case-law-records

[4] The National Archives, "Open Justice Licence", version 2.0. https://caselaw.nationalarchives.gov.uk/open-justice-licence
