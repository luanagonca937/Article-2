# Reproduction package: literature-driven topic modeling and multicriteria screening of digital poultry technologies

Code and processed data for the article *A Literature-Driven Topic Modeling and Multicriteria Screening Framework for Economic-Feasibility Evidence of Digital Technologies in Poultry Farming*.

The package runs from any folder. Inputs are read from `data/` and outputs are written to `GENERATED_RESULTS/`, both next to the script.

## What is and is not included

| Included | Not included, and why |
|---|---|
| All analysis code (`code/`), MIT license | Exported BibTeX records: subject to the license terms of ScienceDirect, Scopus, Web of Science and IEEE Xplore |
| Search protocol and the list of the 204 analyzed records (`data/`) | Full texts of the articles: copyrighted |
| Archived result tables reported in the article (`archived_results/`) | Excerpts of article text in the candidate-value table |

## Run

Requires Python 3.10 or newer.

1. Repeat the search in `data/SEARCH_PROTOCOL.md` and export the records as BibTeX with abstracts.
2. Put the `.bib` files in `data/raw_bib/`.
3. Run:

| System | Command |
|---|---|
| Windows | `run_windows.bat` |
| Linux / macOS | `bash run_linux_mac.sh` |
| Any (libraries already installed) | `python reproduce_article2.py` |

The script filters the poultry records, runs LDA and AviTech-Bayes, runs the multicriteria analysis, draws the figures and compares seven main values with those in the article (`data/EXPECTED_RESULTS.json`). It prints `PASS` or `FAIL` for each value and saves `GENERATED_RESULTS/VERIFICATION_REPORT.json`.

Seeds are fixed (LDA 1234, AviTech-Bayes 42, Monte Carlo 20260703, bootstrap 20260713). Because the databases add records over time, a new export can differ slightly from the one used in the article (2 November 2026), and some values may then fail.

Tested with Python 3.11.9 and the versions in `requirements-tested.txt`.

## Code

| File | Role |
|---|---|
| `code/filter_poultry_records.py` | Keeps records with poultry terms in title, abstract or keywords |
| `code/lda_avitech_bayes_analysis.py` | Preprocessing, LDA, AviTech-Bayes and method comparison |
| `code/lda_bibliometric_module.py`, `code/avitech_bayes_module.py` | Modules used by the previous script |
| `code/multicriteria_without_experts.py` | Topic-derived weights, decision matrix, TOPSIS, VIKOR and robustness checks |
| `code/make_figures.py` | Figures 1 to 8 of the first submitted version; later figures were drawn from the same tables |
| `code/make_graphical_abstract.py` | Graphical abstract (numbers taken from the article); run `python code/make_graphical_abstract.py <output_folder>` |
| `code/extract_fulltext_values.py` | Candidate economic values from full texts obtained by the user |
| `code/economic_value_extraction_rules.py` | Extraction rules used by the previous script |

## Origin of the code

The analysis scripts are the ones that produced the article's results (run on 2 November 2026). For this public version they were translated into English: file names, output-folder and output-file names, a few variable names, comments, docstrings and console messages. The computations were not changed. Three kinds of Portuguese text were kept on purpose because they are data: the Portuguese stop-word lists, the accent-normalization patterns and two Portuguese search terms of the guideline dictionary. The hard-coded folder path was also replaced in two scripts, so the package runs on any computer. After the translation, the package was run again from the exported records and the main values matched the article (see the verification note in the repository upload instructions).

| File (original name) | Origin | Changes |
|---|---|---|
| `lda_avitech_bayes_analysis.py` (`2_lda_bnmf_bibliometrica_sem_hoshin.py`) | Authors' analysis script (26 June 2026) | Renamed; output folders and messages in English |
| `avitech_bayes_module.py` (`2_ bay_tec.py`) | Authors' AviTech-Bayes script (19 June 2026) | Renamed; comments, messages and output names in English |
| `multicriteria_without_experts.py` (`SEMespecialista_multicriterio.py`) | Authors' multicriteria script (13 July 2026) | Renamed; output folders in English |
| `lda_bibliometric_module.py` (`2_ artig_lda_bibliometrica_corrigido.py`) | Authors' LDA module (22 June 2026) | Renamed; folder path; variable names, topic labels, messages and output names in English |
| `economic_value_extraction_rules.py` (`V9_12_auto_economic_value_extraction.py`) | Authors' extraction rules (21 July 2026) | Renamed; folder path |
| `filter_poultry_records.py`, `make_figures.py`, `make_graphical_abstract.py`, `extract_fulltext_values.py`, `reproduce_article2.py` | Written for this study with AI assistance and reviewed by the authors | Not applicable |

Candidate values with confidence of at least 0.80 were checked manually against the source articles before being reported.

## License

Code: MIT (see `LICENSE`). Cite the article and this repository (`CITATION.cff`).
