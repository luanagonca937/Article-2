"""Reproduce the results of Article 2 from the exported BibTeX records.

Steps
1. Filter the exported records to those with poultry terms in title, abstract or keywords.
2. Run LDA and AviTech-Bayes on the filtered records (method comparison and topics).
3. Run the literature-driven multicriteria analysis (weights, TOPSIS, VIKOR, robustness).
4. Draw the manuscript figures.
5. Compare the main values with those reported in the article.

The exported records are not redistributed. Place the BibTeX files exported with the search
protocol in data/SEARCH_PROTOCOL.md into data/raw_bib/ before running.
Outputs go to GENERATED_RESULTS/ next to this script.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
RAW = HERE / "data" / "raw_bib"
OUT = HERE / "GENERATED_RESULTS"
BIB = OUT / "poultry_bib"
RES = OUT / "results"
TABLES = RES / "6_multicriteria" / "without_expert_topic_weights" / "TABLES"


def run(args, cwd=None):
    print("\n>", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable] + [str(a) for a in args], cwd=cwd, check=True,
                   env={**os.environ, "MPLBACKEND": "Agg"})


def main():
    if not any(RAW.glob("*.bib")):
        sys.exit(f"No .bib files in {RAW}. See data/SEARCH_PROTOCOL.md.")
    OUT.mkdir(exist_ok=True)
    run([CODE / "filter_poultry_records.py", RAW, BIB])
    run([CODE / "lda_avitech_bayes_analysis.py", "--input-dir", BIB, "--results-dir", RES])
    shutil.copy(CODE / "multicriteria_without_experts.py", RES / "multicriteria_without_experts.py")
    run([RES / "multicriteria_without_experts.py"], cwd=RES)
    run([CODE / "make_figures.py", RES, OUT / "figures"])
    verify()


def verify():
    expected = json.loads((HERE / "data" / "EXPECTED_RESULTS.json").read_text(encoding="utf8"))
    comp = pd.read_excel(RES / "5_LDA_BNMF_common_results" / "05_method_comparison_metrics.xlsx").set_index("method")
    rank = pd.read_excel(TABLES / "10_no_expert_topsis_vikor_technology_ranking.xlsx").sort_values("final_no_expert_rank")
    agree = pd.read_excel(TABLES / "19_topsis_vikor_rank_agreement_metrics.xlsx").set_index("metric")["value"]
    mc = pd.read_excel(TABLES / "21_monte_carlo_rank_acceptability_summary.xlsx").set_index("technology_label")
    docs = len(pd.read_csv(RES / "1_bibliometric_analysis" / "00_base_complete_deduplicated_for_topic_models.csv", low_memory=False))
    got = {
        "documents_after_deduplication": docs,
        "composite_AviTech-Bayes": round(float(comp.loc["AviTech-Bayes", "composite_score"]), 2),
        "composite_LDA": round(float(comp.loc["LDA", "composite_score"]), 2),
        "top_three_final_rank": list(rank["technology_label"].head(3)),
        "spearman_topsis_vikor": round(float(agree["Spearman rank correlation"]), 3),
        "monte_carlo_first_ML_AI": round(float(mc.loc["Machine learning / AI", "probability_rank_1"]), 3),
        "monte_carlo_first_sensors": round(float(mc.loc["Sensors and monitoring", "probability_rank_1"]), 3),
    }
    report, tally = [], {"PASS": 0, "FAIL": 0}
    for key, value in expected.items():
        status = "PASS" if got.get(key) == value else "FAIL"
        tally[status] += 1
        report.append({"value": key, "expected": value, "obtained": got.get(key), "status": status})
        print(f"{status}  {key}: expected {value}, obtained {got.get(key)}")
    (OUT / "VERIFICATION_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(tally)
    sys.exit(1 if tally["FAIL"] else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        verify()
    else:
        main()
