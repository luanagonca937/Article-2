# Search protocol

String (all four databases):

("poultry" OR broiler OR layer) AND ("precision livestock farming" OR "digital transformation") AND (feasibility OR ROI OR cost)

| Database | Field searched | Raw results | Eligible (articles) | Exported |
|---|---|---:|---:|---:|
| ScienceDirect | basic search "Find articles with these terms" (includes full text) | 12,183 | 8,191 (research articles) | 6,000 |
| Scopus | Article title, Abstract, Keywords | 265 | 126 (document type: Article; books, conference papers and reviews excluded) | 126 |
| Web of Science Core Collection | Topic | 141 | 106 (articles) | 106 |
| IEEE Xplore | basic search, all content types | 191 | 24 (journals) | 24 |

- Exported on 2 November 2026, accessed through the CAPES Periodicals Portal. No publication-year limit.
- ScienceDirect exports stopped at 6,000 of 8,191 eligible articles.
- Repeating the search on 5 November 2026 returned the same counts in ScienceDirect and IEEE Xplore and one more record in Scopus (266) and Web of Science (142).
- Export the results as BibTeX with abstracts and put the files in `data/raw_bib/`. ScienceDirect exports 100 records per file.
- `poultry_records_204.csv` lists the 204 records analyzed (title, year, journal, DOI), so the subset can be checked without the exports.
