#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-expert multicriteria analysis support for the poultry digital-technology study.

This script is intentionally separate from:
    lda_avitech_bayes_analysis.py

Purpose
-------
It uses only topic-derived statistical weights from the topic most aligned with
the research objective. No expert weights, Fuzzy Delphi validation, or BWM input
templates are required in this version.

Conceptual design
-----------------
1. Read the already-generated LDA/AviTech-Bayes outputs.
2. Select the winning method from the method-comparison table.
3. Select the topic most aligned with the research objective.
4. Convert that topic's top words into criterion weights.
5. Mine title/abstract/keywords for economic-feasibility evidence.
6. Build a no-expert decision matrix for the identified digital technologies.
7. Rank the technologies using topic-derived weights with TOPSIS and VIKOR.
8. Validate the decision matrix through traceability, alternative weighting
   schemes, rank agreement, sensitivity scenarios, and Monte Carlo rank
   acceptability.
9. Export tables and figures for a topic-weighted multicriteria scenario.

Important methodological note
-----------------------------
This no-expert scenario is data-driven. The criterion weights are extracted from
the vocabulary structure of the most aligned topic. Therefore, the ranking does
not depend on expert elicitation. TOPSIS and VIKOR are applied as deterministic
multicriteria procedures using the literature-derived weights.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "6_multicriteria" / "without_expert_topic_weights"
IMG_DIR = OUTPUT_DIR / "IMAGES"
TABLE_DIR = OUTPUT_DIR / "TABLES"

COMMON_DIR = BASE_DIR / "5_LDA_BNMF_common_results"
LDA_DIR = BASE_DIR / "2_LDA_results"
BNMF_DIR = BASE_DIR / "3_BNMF_results"


CRITERION_GROUPS = {
    "C1_Initial_investment_CAPEX": {
        "investment", "capital", "capex", "equipment", "infrastructure",
        "implementation", "installation", "hardware", "device", "devices",
        "sensor", "sensors", "camera", "platform", "low_cost", "cost",
        "costs", "affordable", "affordability",
    },
    "C2_Operating_cost_OPEX": {
        "opex", "operating", "operation", "operational", "maintenance",
        "labor", "labour", "energy", "feed", "water", "cost", "costs",
        "savings", "saving", "efficiency", "resource", "resources",
        "automation", "monitoring",
    },
    "C3_ROI_payback_return_evidence": {
        "roi", "return_on_investment", "return", "returns", "payback",
        "profit", "profitability", "revenue", "benefit", "benefits",
        "cost_benefit", "cost_effective", "economic", "value",
        "viability", "feasibility",
    },
    "C4_Productivity_welfare_performance_gain": {
        "production", "productivity", "performance", "efficiency", "feed",
        "feeding", "weight", "growth", "mortality", "health", "welfare",
        "behavior", "behaviour", "broiler", "broilers", "layer", "layers",
        "egg", "eggs", "flock", "poultry", "chicken", "housing",
    },
    "C5_Implementation_complexity_training": {
        "adoption", "management", "decision", "support", "farmer",
        "farmers", "training", "knowledge", "barrier", "barriers",
        "challenge", "challenges", "implementation", "risk", "acceptance",
        "organizational", "business", "complexity", "skills", "usability",
    },
    "C6_Scalability_technological_maturity": {
        "scale", "scalable", "scalability", "maturity", "commercial",
        "prototype", "field", "farm", "farms", "deployment", "platform",
        "system", "systems", "smart_technology", "adoption",
        "implementation", "infrastructure", "farm_level",
        "precision_livestock_farming", "plf",
    },
    "C7_Data_reliability_interoperability": {
        "accuracy", "reliability", "robust", "robustness", "data",
        "dataset", "datasets", "algorithm", "algorithms", "model",
        "models", "prediction", "classification", "detection",
        "interoperability", "integration", "cloud", "edge", "iot",
    },
    "C8_Environmental_resource_efficiency": {
        "environment", "environmental", "sustainability", "sustainable",
        "energy", "emission", "emissions", "carbon", "waste", "water",
        "ammonia", "climate", "resource", "resources", "circular",
    },
}

CRITERION_METADATA = {
    "C1_Initial_investment_CAPEX": {
        "label": "Initial investment / CAPEX evidence",
        "type": "cost-support",
        "interpretation": "Evidence that the technology has identifiable acquisition, installation, equipment, or infrastructure requirements.",
    },
    "C2_Operating_cost_OPEX": {
        "label": "Maintenance and operating cost / OPEX evidence",
        "type": "cost-support",
        "interpretation": "Evidence related to maintenance, labor, energy, feed, water, resource use, or operating-cost implications.",
    },
    "C3_ROI_payback_return_evidence": {
        "label": "ROI, payback, or economic return evidence",
        "type": "benefit-support",
        "interpretation": "Evidence of return, payback, profitability, cost-benefit, or economic-feasibility arguments.",
    },
    "C4_Productivity_welfare_performance_gain": {
        "label": "Productivity, welfare, or performance gain",
        "type": "benefit-support",
        "interpretation": "Evidence of gains in production, growth, mortality reduction, welfare, health, or flock performance.",
    },
    "C5_Implementation_complexity_training": {
        "label": "Implementation complexity and training requirement",
        "type": "cost-support",
        "interpretation": "Evidence of adoption barriers, training needs, management demands, risk, or organizational complexity.",
    },
    "C6_Scalability_technological_maturity": {
        "label": "Scalability and technological maturity",
        "type": "benefit-support",
        "interpretation": "Evidence that the technology can move beyond prototypes toward field, farm, commercial, or scalable use.",
    },
    "C7_Data_reliability_interoperability": {
        "label": "Data reliability and interoperability",
        "type": "benefit-support",
        "interpretation": "Evidence of accuracy, robustness, data integration, prediction quality, connectivity, or interoperability.",
    },
    "C8_Environmental_resource_efficiency": {
        "label": "Environmental and resource efficiency",
        "type": "benefit-support",
        "interpretation": "Evidence of environmental, energy, water, waste, emissions, or resource-efficiency relevance.",
    },
}


TECHNOLOGY_TERMS = {
    "precision_livestock_farming": ["precision livestock farming", "plf"],
    "digital_transformation": ["digital transformation"],
    "iot": ["iot", "internet of things"],
    "sensors_monitoring": ["sensor", "sensors", "monitoring"],
    "computer_vision": ["computer vision", "machine vision", "vision"],
    "machine_learning_ai": ["machine learning", "deep learning", "artificial intelligence", "ai"],
    "automation_robotics": ["automation", "robot", "robotics", "automated"],
    "blockchain_supply_chain": ["blockchain", "supply chain"],
    "digital_twin": ["digital twin"],
}

TECHNOLOGY_DISPLAY_LABELS = {
    "precision_livestock_farming": "Precision livestock farming",
    "digital_transformation": "Digital transformation",
    "iot": "Internet of Things (IoT)",
    "sensors_monitoring": "Sensors and monitoring",
    "computer_vision": "Computer vision",
    "machine_learning_ai": "Machine learning / AI",
    "automation_robotics": "Automation and robotics",
    "blockchain_supply_chain": "Blockchain and supply chain",
    "digital_twin": "Digital twin",
}

# Digital transformation and precision livestock farming are kept as search
# axes/contextual constructs. They are not ranked as technology alternatives
# because they are broader than specific deployable technologies such as IoT,
# sensors, computer vision, robotics, blockchain, or digital twin.
CONTEXTUAL_TECHNOLOGY_KEYS = {"digital_transformation", "precision_livestock_farming"}
DECISION_TECHNOLOGY_KEYS = set(TECHNOLOGY_TERMS) - CONTEXTUAL_TECHNOLOGY_KEYS
MINIMUM_CRITERION_WEIGHT_FLOOR = 0.03
MONTE_CARLO_ITERATIONS = 5000
MONTE_CARLO_SEED = 20260703
DOCUMENT_BOOTSTRAP_ITERATIONS = 150
DOCUMENT_BOOTSTRAP_SEED = 20260713
EPSILON_FLOOR_VALUES = [0.00, 0.01, 0.02, 0.03, 0.05, 0.10]

ECONOMIC_CRITERIA = [
    "C1_Initial_investment_CAPEX",
    "C2_Operating_cost_OPEX",
    "C3_ROI_payback_return_evidence",
]
TECHNICAL_ADOPTION_CRITERIA = [
    "C4_Productivity_welfare_performance_gain",
    "C5_Implementation_complexity_training",
    "C6_Scalability_technological_maturity",
    "C7_Data_reliability_interoperability",
]
SUSTAINABILITY_CRITERIA = [
    "C4_Productivity_welfare_performance_gain",
    "C8_Environmental_resource_efficiency",
    "C2_Operating_cost_OPEX",
]
STRONG_RETURN_TERMS = {
    "roi", "return_on_investment", "payback", "npv", "irr",
    "cost_benefit", "cost_effective", "profitability", "savings",
}


ECONOMIC_TERMS = {
    "cost", "costs", "economic", "economics", "economic feasibility",
    "feasibility", "roi", "return on investment", "payback", "investment",
    "profit", "profitability", "revenue", "benefit", "benefits",
    "cost-benefit", "cost benefit", "value", "savings", "labor cost",
    "labour cost", "feed cost", "capex", "opex", "npv", "irr",
}

POULTRY_TERMS = {
    "poultry", "broiler", "broilers", "chicken", "chickens",
    "laying_hen", "laying_hens", "hen", "hens", "quail",
    "chick", "chicks", "breeder", "breeders",
}


COUNTRY_ALIASES = {
    "brazil": "Brazil",
    "brasil": "Brazil",
    "china": "China",
    "united states": "United States",
    "usa": "United States",
    "u.s.": "United States",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "india": "India",
    "netherlands": "Netherlands",
    "italy": "Italy",
    "spain": "Spain",
    "france": "France",
    "germany": "Germany",
    "turkey": "Turkey",
    "australia": "Australia",
    "canada": "Canada",
}


def normalized_title(title: str) -> str:
    text = str(title or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalized_doi(doi: str) -> str:
    text = str(doi or "").strip().lower()
    if text in {"", "nan", "none"}:
        return ""
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip()


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated records across databases, preferring DOI-rich entries."""
    if df.empty or "title" not in df.columns:
        return df
    out = df.copy()
    out["_title_key"] = out["title"].map(normalized_title)
    out["_doi_key"] = out.get("doi", "").map(normalized_doi) if "doi" in out.columns else ""
    out["_has_doi"] = out["_doi_key"].astype(str).str.len() > 0
    out["_dedup_key"] = np.where(out["_has_doi"], "doi:" + out["_doi_key"], "title:" + out["_title_key"])
    out = out.sort_values(
        ["economic_evidence_priority", "_has_doi", "topic_probability"],
        ascending=[False, False, False],
    )
    # First collapse exact DOI duplicates, then collapse title duplicates where one database lacks DOI.
    out = out.drop_duplicates("_dedup_key", keep="first")
    out = out.sort_values(
        ["_title_key", "_has_doi", "economic_evidence_priority"],
        ascending=[True, False, False],
    ).drop_duplicates("_title_key", keep="first")
    out = out.sort_values("economic_evidence_priority", ascending=False).reset_index(drop=True)
    return out.drop(columns=["_title_key", "_doi_key", "_has_doi", "_dedup_key"], errors="ignore")


def source_category_from_file(filename: str) -> str:
    name = str(filename or "").lower()
    if "sciencedirect" in name:
        return "ScienceDirect"
    if "scopus" in name:
        return "Scopus"
    if "webofscience" in name or "web of science" in name:
        return "Web of Science"
    if "ieee" in name or "xplore" in name:
        return "IEEE Xplore"
    return "Other"


def technology_label(technology: str) -> str:
    key = str(technology or "").strip()
    return TECHNOLOGY_DISPLAY_LABELS.get(key, key.replace("_", " ").title())


def load_source_lookup() -> dict:
    """Return normalized title -> source metadata."""
    lookup = {}
    doi_lookup = {}
    upset_path = BASE_DIR / "1_bibliometric_analysis" / "04_database_intersection_upsetplot.xlsx"
    if upset_path.exists():
        upset_df = pd.read_excel(upset_path)
        for _, row in upset_df.iterrows():
            title_key = str(row.get("normalized_title", "")).strip()
            if not title_key:
                continue
            sources = []
            for source in ["ScienceDirect", "Scopus", "Web of Science", "IEEE Xplore"]:
                if bool(row.get(source, False)):
                    sources.append(source)
            lookup[title_key] = {
                "source_database": " + ".join(sources),
                "source_category": "Multiple databases" if len(sources) > 1 else (sources[0] if sources else ""),
                "source_bibtex_file": "",
            }

    # Fallback: if the original deduplicated base has a source file in future runs.
    base_path = BASE_DIR / "1_bibliometric_analysis" / "00_base_complete_deduplicated_for_topic_models.xlsx"
    if base_path.exists():
        try:
            base_df = pd.read_excel(base_path)
            for _, row in base_df.iterrows():
                title_key = normalized_title(row.get("TITLE", ""))
                if not title_key:
                    continue
                source_file = row.get("SOURCE_FILE", row.get("source_file", ""))
                if source_file and title_key not in lookup:
                    category = source_category_from_file(source_file)
                    lookup[title_key] = {
                        "source_database": category,
                        "source_category": category,
                        "source_bibtex_file": str(source_file),
                    }
                doi = str(row.get("DOI", row.get("doi", ""))).strip().lower()
                if doi and doi not in {"nan", "none"}:
                    info = lookup.get(title_key)
                    if info:
                        doi_lookup[doi] = info
        except Exception:
            pass
    return {"title": lookup, "doi": doi_lookup}


def ensure_dirs() -> None:
    for directory in [OUTPUT_DIR, IMG_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    replacements = {
        r"\bprecision\s+livestock\s+farming\b": "precision_livestock_farming",
        r"\bdigital\s+transformation\b": "digital_transformation",
        r"\beconomic\s+feasibility\b": "economic_feasibility",
        r"\breturn\s+on\s+investment\b": "return_on_investment",
        r"\bcost\s*[- ]\s*benefit\b": "cost_benefit",
        r"\bcost\s*[- ]\s*effective\b": "cost_effective",
        r"\blow\s*[- ]\s*cost\b": "low_cost",
        r"\bfarm\s*[- ]\s*level\b": "farm_level",
        r"\bmachine\s+learning\b": "machine_learning",
        r"\bdeep\s+learning\b": "deep_learning",
        r"\bartificial\s+intelligence\b": "artificial_intelligence",
        r"\bcomputer\s+vision\b": "computer_vision",
        r"\binternet\s+of\s+things\b": "iot",
        r"\bdigital\s+twin\b": "digital_twin",
        r"\blaying\s+hens\b": "laying_hens",
        r"\blaying\s+hen\b": "laying_hen",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9_%$€£.,:\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def poultry_terms_found(text: str) -> list[str]:
    norm = normalize_text(text)
    tokens = set(re.sub(r"[^a-z0-9_\s]", " ", norm).split())
    found = sorted(term for term in POULTRY_TERMS if term in tokens)
    if "layer" in tokens or "layers" in tokens:
        if {"hen", "hens", "chicken", "chickens", "poultry", "laying_hen", "laying_hens"} & tokens:
            found.append("layer")
    return sorted(set(found))


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    tokens = [token for token in text.split() if len(token) > 2 or token in {"ai", "io"}]
    return tokens


def read_first_existing(paths: list[Path], **kwargs) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path, **kwargs)
            return pd.read_excel(path, **kwargs)
    raise FileNotFoundError("None of these files exists: " + "; ".join(str(p) for p in paths))


def load_method_comparison() -> pd.DataFrame:
    return read_first_existing([COMMON_DIR / "05_method_comparison_metrics.xlsx"])


def winning_method(comparison_df: pd.DataFrame) -> str:
    if "composite_score" not in comparison_df.columns:
        return "AviTech-Bayes"
    return str(comparison_df.sort_values("composite_score", ascending=False).iloc[0]["method"])


def load_alignment_for_method(method: str) -> pd.DataFrame:
    path = COMMON_DIR / "05_topic_alignment_by_method.xlsx"
    if not path.exists():
        raise FileNotFoundError(path)
    xls = pd.ExcelFile(path)
    sheet = "AviTech_Bayes" if "AviTech" in method or "BNMF" in method else "LDA"
    if sheet not in xls.sheet_names:
        sheet = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)


def select_aligned_topic(alignment_df: pd.DataFrame) -> int:
    """Select the topic used to derive no-expert weights.

    The multicriteria weights must come from the topic most aligned with the
    research objective as a whole: poultry farming, digital technology, and
    economic feasibility. A topic with many generic technology terms can have a
    high raw alignment score, but it should not dominate if another topic
    better balances poultry relevance and economic-feasibility terms.
    """
    work = alignment_df.copy()
    for col in ["covers_poultry", "covers_technology", "covers_economic", "economic_hits", "poultry_hits", "technology_hits"]:
        if col not in work.columns:
            work[col] = 0
    if "objective_alignment_score" not in work.columns:
        work["objective_alignment_score"] = 0.0
    work["covers_all_objective_axes"] = (
        work["covers_poultry"].astype(int)
        + work["covers_technology"].astype(int)
        + work["covers_economic"].astype(int)
    )
    work["balanced_objective_score"] = (
        0.35 * work["economic_hits"].astype(float)
        + 0.30 * work["poultry_hits"].astype(float)
        + 0.20 * np.minimum(work["technology_hits"].astype(float), 3.0)
        + 0.15 * work["objective_alignment_score"].astype(float)
    )
    work = work.sort_values(
        [
            "covers_all_objective_axes",
            "economic_hits",
            "poultry_hits",
            "balanced_objective_score",
            "objective_alignment_score",
        ],
        ascending=False,
    )
    return int(work.iloc[0]["topic"])


def load_topic_words(method: str) -> pd.DataFrame:
    if "AviTech" in method or "BNMF" in method:
        return read_first_existing([BNMF_DIR / "04_bnmf_topic_words.xlsx"])

    lda_workbook = read_first_existing([LDA_DIR / "03_lda_results_topic_model_corpus.xlsx"])
    # Fallback is intentionally simple; LDA topic words are usually stored in a sheet.
    path = LDA_DIR / "03_lda_results_topic_model_corpus.xlsx"
    if path.exists():
        words = pd.read_excel(path, sheet_name="LDA_Words")
        rows = []
        for _, row in words.iterrows():
            topic_raw = str(row.get("Topic", ""))
            match = re.search(r"(\d+)", topic_raw)
            topic = int(match.group(1)) if match else int(row.name + 1)
            for rank, word in enumerate(str(row.get("Top_Words", "")).split(","), 1):
                word = word.strip()
                if word:
                    rows.append({"topic": topic, "rank": rank, "word": word})
        return pd.DataFrame(rows)
    return lda_workbook


def topic_statistical_weights(topic_words_df: pd.DataFrame, topic: int) -> pd.DataFrame:
    subset = topic_words_df[topic_words_df["topic"].astype(int) == int(topic)].copy()
    if subset.empty:
        raise RuntimeError(f"No topic words found for topic {topic}.")
    subset["rank"] = pd.to_numeric(subset["rank"], errors="coerce")
    if subset["rank"].isna().any():
        subset["rank"] = subset["rank"].fillna(pd.Series(np.arange(1, len(subset) + 1), index=subset.index))
    subset["word_weight"] = 1 / subset["rank"]
    rows = []
    for criterion, terms in CRITERION_GROUPS.items():
        hits = subset[subset["word"].astype(str).str.lower().isin(terms)].copy()
        raw_weight = float(hits["word_weight"].sum())
        meta = CRITERION_METADATA.get(criterion, {})
        rows.append({
            "criterion": criterion,
            "criterion_label": meta.get("label", criterion),
            "criterion_type": meta.get("type", "evidence-support"),
            "raw_statistical_weight": raw_weight,
            "matched_topic_words": ", ".join(hits["word"].astype(str).tolist()),
            "matched_word_count": int(len(hits)),
        })
    weights = pd.DataFrame(rows)
    total = float(weights["raw_statistical_weight"].sum())
    if total <= 0:
        weights["statistical_weight_before_floor"] = 1 / len(weights)
        weights["statistical_weight"] = 1 / len(weights)
    else:
        weights["statistical_weight_before_floor"] = weights["raw_statistical_weight"] / total
        weights["statistical_weight"] = weights["statistical_weight_before_floor"].clip(lower=MINIMUM_CRITERION_WEIGHT_FLOOR)
        weights["statistical_weight"] = weights["statistical_weight"] / weights["statistical_weight"].sum()
    weights["minimum_floor_applied"] = (
        weights["statistical_weight_before_floor"] < MINIMUM_CRITERION_WEIGHT_FLOOR
    )
    weights = weights.sort_values("statistical_weight", ascending=False).reset_index(drop=True)
    return weights


def load_documents(method: str) -> pd.DataFrame:
    if "AviTech" in method or "BNMF" in method:
        return read_first_existing([
            BNMF_DIR / "04_bnmf_documents_with_topics.xlsx",
            BNMF_DIR / "04_bnmf_documents_with_topics.csv",
        ])
    return pd.read_excel(LDA_DIR / "03_lda_results_topic_model_corpus.xlsx", sheet_name="Documents_with_LDA_Topics")


def detect_technologies(text: str) -> list[str]:
    norm = normalize_text(text)
    found = []
    for technology, terms in TECHNOLOGY_TERMS.items():
        if any(term in norm for term in terms):
            found.append(technology)
    return found


def economic_signal(text: str) -> dict:
    norm = normalize_text(text)
    terms_found = sorted(term for term in ECONOMIC_TERMS if term in norm)
    numeric_patterns = [
        r"\b\d+(?:[.,]\d+)?\s?%",
        r"[$€£]\s?\d+(?:[.,]\d+)?",
        r"\b\d+(?:[.,]\d+)?\s?(usd|eur|brl|dollars|euros)\b",
        r"\b\d+(?:[.,]\d+)?\s?(year|years|month|months)\b",
    ]
    numeric_hits = []
    for pattern in numeric_patterns:
        numeric_hits.extend(re.findall(pattern, norm, flags=re.IGNORECASE))
    # Use re.finditer for readable snippets around numeric values.
    snippets = []
    for match in re.finditer(r"(.{0,45}(?:\d+(?:[.,]\d+)?\s?%|[$€£]\s?\d+(?:[.,]\d+)?).{0,45})", norm):
        snippets.append(match.group(1).strip())
    return {
        "economic_terms_found": ", ".join(terms_found),
        "economic_term_count": len(terms_found),
        "numeric_value_signal": len(numeric_hits) + len(snippets),
        "numeric_snippets": " | ".join(snippets[:5]),
    }


def economic_evidence_quality(econ: dict, text: str) -> dict:
    """Classify whether the article only mentions economics or reports stronger evidence."""
    norm = normalize_text(text)
    terms = {term.strip() for term in str(econ.get("economic_terms_found", "")).split(",") if term.strip()}
    numeric_signal = int(econ.get("numeric_value_signal", 0) or 0)
    strong_return_hit = bool(terms & STRONG_RETURN_TERMS) or any(term in norm for term in STRONG_RETURN_TERMS)
    cost_hit = any(term in norm for term in ["cost", "costs", "capex", "opex", "investment", "savings"])

    if numeric_signal >= 2 and strong_return_hit:
        level, label = 5, "Full economic signal"
        interpretation = "Contains numeric values and return-oriented terms such as ROI, payback, savings, or cost-benefit."
    elif numeric_signal >= 1 and (strong_return_hit or cost_hit):
        level, label = 4, "Strong economic evidence"
        interpretation = "Contains at least one numeric signal linked to cost, investment, savings, or return."
    elif numeric_signal >= 1 or int(econ.get("economic_term_count", 0) or 0) >= 4:
        level, label = 3, "Moderate evidence"
        interpretation = "Contains numerical or repeated economic signals but still requires full-text confirmation."
    elif int(econ.get("economic_term_count", 0) or 0) >= 2:
        level, label = 2, "Qualitative economic discussion"
        interpretation = "Mentions more than one economic or feasibility term without clear numeric evidence."
    elif int(econ.get("economic_term_count", 0) or 0) == 1:
        level, label = 1, "Generic mention"
        interpretation = "Mentions one economic term only; this is weak screening evidence."
    else:
        level, label = 0, "No economic evidence"
        interpretation = "No explicit economic signal was detected in the screened fields."

    return {
        "evidence_quality_level": level,
        "evidence_quality_label": label,
        "evidence_quality_interpretation": interpretation,
        "strong_return_term_detected": strong_return_hit,
        "strong_economic_evidence": level >= 4,
    }


def detect_countries(row: pd.Series) -> list[str]:
    text = " ".join(str(row.get(col, "")) for col in ["affiliation", "AFFILIATION", "address", "ADDRESS", "publisher", "journal"])
    text_l = text.lower()
    found = []
    for alias, country in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text_l):
            found.append(country)
    return sorted(set(found))


def economic_evidence_mining(documents: pd.DataFrame, topic: int, method: str) -> pd.DataFrame:
    docs = documents.copy()
    if "AviTech" in method or "BNMF" in method:
        topic_col = "topic_bnmf"
        prob_col = "topic_bnmf_prob_map"
    else:
        topic_col = "main_topic"
        prob_col = "topic_probability"

    if topic_col not in docs.columns:
        raise KeyError(f"Missing topic column: {topic_col}")
    topic_docs = docs[docs[topic_col].astype(int) == int(topic)].copy()
    source_lookup = load_source_lookup()

    rows = []
    for _, row in topic_docs.iterrows():
        title_value = row.get("title", row.get("TITLE", ""))
        doi_value = str(row.get("doi", row.get("DOI", ""))).strip().lower()
        source_info = source_lookup["title"].get(normalized_title(title_value), {})
        if not source_info and doi_value:
            source_info = source_lookup["doi"].get(doi_value, {})
        text = " ".join(str(row.get(col, "")) for col in ["title", "TITLE", "abstract", "ABSTRACT", "keywords", "KEYWORDS"])
        econ = economic_signal(text)
        quality = economic_evidence_quality(econ, text)
        technologies = detect_technologies(text)
        countries = detect_countries(row)
        topic_prob = float(row.get(prob_col, 0) or 0)
        priority = (
            0.40 * topic_prob
            + 0.25 * min(econ["economic_term_count"] / 5, 1)
            + 0.20 * min(len(technologies) / 3, 1)
            + 0.15 * min(econ["numeric_value_signal"] / 2, 1)
        )
        rows.append({
            "title": title_value,
            "source_database": source_info.get("source_database", ""),
            "source_category": source_info.get("source_category", ""),
            "source_bibtex_file": source_info.get("source_bibtex_file", ""),
            "year": row.get("year", row.get("YEAR", "")),
            "journal": row.get("journal", row.get("JOURNAL", "")),
            "doi": row.get("doi", row.get("DOI", "")),
            "topic": topic,
            "topic_probability": topic_prob,
            "technologies_detected": ", ".join(technologies),
            "technology_count": len(technologies),
            "countries_detected": ", ".join(countries),
            **econ,
            **quality,
            "economic_evidence_priority": priority,
            "abstract": row.get("abstract", row.get("ABSTRACT", "")),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return deduplicate_articles(result)


def aggregate_evidence(evidence_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if evidence_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    tech_rows = []
    for _, row in evidence_df.iterrows():
        for technology in str(row.get("technologies_detected", "")).split(","):
            technology = technology.strip()
            if technology:
                tech_rows.append({
                    "technology": technology,
                    "economic_evidence_priority": row["economic_evidence_priority"],
                    "topic_probability": row["topic_probability"],
                    "economic_term_count": row["economic_term_count"],
                    "numeric_value_signal": row["numeric_value_signal"],
                })
    tech_df = pd.DataFrame(tech_rows)
    if not tech_df.empty:
        tech_summary = (
            tech_df.groupby("technology")
            .agg(
                documents=("technology", "count"),
                mean_priority=("economic_evidence_priority", "mean"),
                mean_topic_probability=("topic_probability", "mean"),
                economic_terms=("economic_term_count", "sum"),
                numeric_signals=("numeric_value_signal", "sum"),
            )
            .reset_index()
            .sort_values(["mean_priority", "documents"], ascending=False)
        )
    else:
        tech_summary = pd.DataFrame()

    country_rows = []
    for _, row in evidence_df.iterrows():
        for country in str(row.get("countries_detected", "")).split(","):
            country = country.strip()
            if country:
                country_rows.append({
                    "country": country,
                    "economic_evidence_priority": row["economic_evidence_priority"],
                    "numeric_value_signal": row["numeric_value_signal"],
                })
    country_df = pd.DataFrame(country_rows)
    if not country_df.empty:
        country_summary = (
            country_df.groupby("country")
            .agg(
                documents=("country", "count"),
                mean_priority=("economic_evidence_priority", "mean"),
                numeric_signals=("numeric_value_signal", "sum"),
            )
            .reset_index()
            .sort_values(["documents", "mean_priority"], ascending=False)
        )
    else:
        country_summary = pd.DataFrame()
    return tech_summary, country_summary


def source_summary(evidence_df: pd.DataFrame) -> pd.DataFrame:
    if evidence_df.empty or "source_database" not in evidence_df.columns:
        return pd.DataFrame()
    work = evidence_df.copy()
    work["source_database"] = work["source_database"].fillna("").replace("", "Not identified")
    return (
        work.groupby("source_database")
        .agg(
            articles=("title", "count"),
            mean_economic_priority=("economic_evidence_priority", "mean"),
            mean_topic_probability=("topic_probability", "mean"),
            numeric_signals=("numeric_value_signal", "sum"),
        )
        .reset_index()
        .sort_values(["articles", "mean_economic_priority"], ascending=False)
    )


def aligned_topic_poultry_retrieval_list(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Filter the selected aligned-topic evidence to poultry-specific full-text candidates."""
    if evidence_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in evidence_df.iterrows():
        text = " ".join(str(row.get(col, "")) for col in ["title", "abstract"])
        poultry_terms = poultry_terms_found(text)
        if not poultry_terms:
            continue
        technology_terms = str(row.get("technologies_detected", "")).strip()
        economic_terms = str(row.get("economic_terms_found", "")).strip()
        rows.append({
            "rank": 0,
            "title": row.get("title", ""),
            "year": row.get("year", ""),
            "journal": row.get("journal", ""),
            "doi": row.get("doi", ""),
            "source_database": row.get("source_database", ""),
            "aligned_topic": row.get("topic", ""),
            "aligned_topic_probability": row.get("topic_probability", ""),
            "economic_evidence_priority": row.get("economic_evidence_priority", ""),
            "poultry_terms_found": ", ".join(poultry_terms),
            "technology_families_detected": technology_terms,
            "economic_terms_found": economic_terms,
            "numeric_value_signal": row.get("numeric_value_signal", ""),
            "numeric_snippets": row.get("numeric_snippets", ""),
            "why_retrieve_full_text": (
                "This article belongs to the topic used to derive the no-expert weights and "
                "contains poultry plus economic-feasibility signals. Retrieve the full text "
                "to extract cost, ROI, payback, investment, savings, scale, country, and "
                "implementation context."
            ),
            "abstract": row.get("abstract", ""),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["economic_evidence_priority", "aligned_topic_probability", "numeric_value_signal"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width))
    tc_w.set(qn("w:type"), "dxa")


def export_aligned_topic_poultry_docx(retrieval_df: pd.DataFrame, output_path: Path) -> None:
    if not DOCX_AVAILABLE or retrieval_df.empty:
        return
    doc = Document()
    section = doc.sections[0]
    section.orientation = 1
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)

    title = doc.add_paragraph("Aligned-topic poultry articles for full-text economic-feasibility extraction")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)
        run.bold = True

    note = doc.add_paragraph(
        "This list is restricted to articles contained in the topic most aligned with the research objective. "
        "Therefore, it is methodologically linked to the same topic structure used to derive the no-expert "
        "multicriteria weights. Full-text reading is required to extract concrete economic values."
    )
    note.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in note.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(9)

    cols = [
        "rank", "title", "year", "journal", "doi", "source_database",
        "aligned_topic_probability", "economic_evidence_priority",
        "technology_families_detected", "economic_terms_found",
    ]
    headers = [
        "Rank", "Title", "Year", "Journal", "DOI", "Database",
        "Topic p", "Economic priority", "Technology family", "Economic terms",
    ]
    widths = [520, 3000, 550, 1600, 1600, 1000, 700, 850, 2100, 1500]
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = header
        set_cell_shading(cell, "D9EAF7")
        set_cell_width(cell, widths[idx])
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(7.3)
                run.bold = True

    for _, row in retrieval_df.head(30).iterrows():
        cells = table.add_row().cells
        for idx, col in enumerate(cols):
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells[idx].text = "" if pd.isna(value) else str(value)
            cells[idx].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_width(cells[idx], widths[idx])
            for paragraph in cells[idx].paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                if col in {"rank", "year", "aligned_topic_probability", "economic_evidence_priority"}:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(6.3)
    doc.save(output_path)


def count_group_terms(text: str, terms: set[str]) -> int:
    tokens = tokenize(text)
    counter = Counter(tokens)
    return int(sum(counter.get(term, 0) for term in terms))


def matched_group_terms(text: str, terms: set[str]) -> list[str]:
    tokens = tokenize(text)
    counter = Counter(tokens)
    matched = []
    for term in sorted(terms):
        count = int(counter.get(term, 0))
        if count:
            matched.extend([term] * count)
    return matched


def criterion_article_scores(row: pd.Series) -> dict[str, float]:
    text = " ".join(str(row.get(col, "")) for col in [
        "title", "abstract", "economic_terms_found", "numeric_snippets",
        "technologies_detected",
    ])
    scores = {}
    for criterion, terms in CRITERION_GROUPS.items():
        hits = count_group_terms(text, terms)
        scores[criterion] = min(hits / 4, 1.0)

    economic_term_score = min(float(row.get("economic_term_count", 0) or 0) / 5, 1.0)
    numeric_score = min(float(row.get("numeric_value_signal", 0) or 0) / 3, 1.0)
    scores["C1_Initial_investment_CAPEX"] = max(
        scores.get("C1_Initial_investment_CAPEX", 0.0),
        0.55 * min(count_group_terms(text, CRITERION_GROUPS["C1_Initial_investment_CAPEX"]) / 4, 1.0)
        + 0.45 * economic_term_score,
    )
    scores["C2_Operating_cost_OPEX"] = max(
        scores.get("C2_Operating_cost_OPEX", 0.0),
        0.55 * min(count_group_terms(text, CRITERION_GROUPS["C2_Operating_cost_OPEX"]) / 4, 1.0)
        + 0.45 * economic_term_score,
    )
    scores["C3_ROI_payback_return_evidence"] = max(
        scores.get("C3_ROI_payback_return_evidence", 0.0),
        0.50 * min(count_group_terms(text, CRITERION_GROUPS["C3_ROI_payback_return_evidence"]) / 3, 1.0)
        + 0.50 * numeric_score,
    )
    return scores


def technology_taxonomy_table() -> pd.DataFrame:
    rows = []
    for technology, terms in TECHNOLOGY_TERMS.items():
        is_contextual = technology in CONTEXTUAL_TECHNOLOGY_KEYS
        rows.append({
            "technology": technology,
            "technology_label": technology_label(technology),
            "terms_used_for_detection": ", ".join(terms),
            "role_in_model": "Contextual search axis" if is_contextual else "Ranked decision alternative",
            "included_in_ranking": not is_contextual,
            "inclusion_rule": (
                "Detected when at least one listed term appears in title, abstract, or keywords. "
                "Contextual axes are retained for corpus interpretation but excluded from ranking "
                "because they are broader than deployable technology families."
            ),
        })
    return pd.DataFrame(rows)


def article_decision_records(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Precompute article-technology criterion scores for fast resampling."""
    if evidence_df.empty:
        return pd.DataFrame()
    rows = []
    for article_position, (_, row) in enumerate(evidence_df.iterrows()):
        technologies = [
            technology.strip()
            for technology in str(row.get("technologies_detected", "")).split(",")
            if technology.strip() and technology.strip() in DECISION_TECHNOLOGY_KEYS
        ]
        if not technologies:
            continue
        article_scores = criterion_article_scores(row)
        topic_probability = float(row.get("topic_probability", 0) or 0)
        priority = float(row.get("economic_evidence_priority", 0) or 0)
        support_factor = 0.50 + 0.25 * min(topic_probability, 1.0) + 0.25 * min(priority, 1.0)
        for technology in technologies:
            record = {
                "article_position": article_position,
                "technology": technology,
                "topic_probability": topic_probability,
                "economic_evidence_priority": priority,
                "numeric_value_signal": float(row.get("numeric_value_signal", 0) or 0),
            }
            for criterion, score in article_scores.items():
                record[criterion] = float(score) * support_factor
            rows.append(record)
    return pd.DataFrame(rows)


def matrix_from_article_decision_records(scored: pd.DataFrame, weight_col: str | None = None) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    records = scored.copy()
    if weight_col is None or weight_col not in records.columns:
        records["_bootstrap_weight"] = 1.0
        weight_col = "_bootstrap_weight"
    records[weight_col] = pd.to_numeric(records[weight_col], errors="coerce").fillna(0).clip(lower=0)
    records = records[records[weight_col] > 0].copy()
    if records.empty:
        return pd.DataFrame()

    rows = []
    for technology, group in records.groupby("technology"):
        weights = group[weight_col].astype(float).to_numpy()
        total_weight = float(weights.sum())
        row = {
            "technology": technology,
            "documents": total_weight,
            "mean_topic_probability": float(np.average(group["topic_probability"].astype(float), weights=weights)),
            "mean_economic_priority": float(np.average(group["economic_evidence_priority"].astype(float), weights=weights)),
            "numeric_signals": float((group["numeric_value_signal"].astype(float) * group[weight_col].astype(float)).sum()),
            "technology_label": technology_label(technology),
            "contextual_terms_excluded_from_ranking": ", ".join(sorted(CONTEXTUAL_TECHNOLOGY_KEYS)),
        }
        for criterion in CRITERION_GROUPS:
            row[criterion] = float(np.average(group[criterion].astype(float), weights=weights))
        rows.append(row)

    matrix = pd.DataFrame(rows)
    missing_technologies = sorted(DECISION_TECHNOLOGY_KEYS - set(matrix["technology"]))
    if missing_technologies:
        missing_rows = []
        for technology in missing_technologies:
            row = {
                "technology": technology,
                "documents": 0.0,
                "mean_topic_probability": 0.0,
                "mean_economic_priority": 0.0,
                "numeric_signals": 0.0,
                "technology_label": technology_label(technology),
                "contextual_terms_excluded_from_ranking": ", ".join(sorted(CONTEXTUAL_TECHNOLOGY_KEYS)),
            }
            for criterion in CRITERION_GROUPS:
                row[criterion] = 0.0
            missing_rows.append(row)
        matrix = pd.concat([matrix, pd.DataFrame(missing_rows)], ignore_index=True)

    max_docs = float(matrix["documents"].max())
    if max_docs <= 0:
        support_boost = pd.Series(0.0, index=matrix.index)
    else:
        support_boost = np.log1p(matrix["documents"].astype(float)) / np.log1p(max_docs)
    for criterion in CRITERION_GROUPS:
        matrix[criterion] = (0.80 * matrix[criterion].astype(float)) + (0.20 * support_boost)
        matrix[criterion] = matrix[criterion].clip(lower=0, upper=1)
    return matrix.sort_values(["mean_economic_priority", "documents"], ascending=False).reset_index(drop=True)


def article_criterion_traceability(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Long audit table showing how each article contributes to each matrix cell."""
    if evidence_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in evidence_df.iterrows():
        technologies = [
            technology.strip()
            for technology in str(row.get("technologies_detected", "")).split(",")
            if technology.strip() and technology.strip() in DECISION_TECHNOLOGY_KEYS
        ]
        if not technologies:
            continue
        text = " ".join(str(row.get(col, "")) for col in [
            "title", "abstract", "economic_terms_found", "numeric_snippets", "technologies_detected",
        ])
        article_scores = criterion_article_scores(row)
        topic_probability = float(row.get("topic_probability", 0) or 0)
        priority = float(row.get("economic_evidence_priority", 0) or 0)
        support_factor = 0.50 + 0.25 * min(topic_probability, 1.0) + 0.25 * min(priority, 1.0)
        for technology in technologies:
            for criterion, terms in CRITERION_GROUPS.items():
                matched = matched_group_terms(text, terms)
                article_score = float(article_scores.get(criterion, 0.0))
                rows.append({
                    "technology": technology,
                    "technology_label": technology_label(technology),
                    "criterion": criterion,
                    "criterion_label": CRITERION_METADATA.get(criterion, {}).get("label", criterion),
                    "title": row.get("title", ""),
                    "year": row.get("year", ""),
                    "journal": row.get("journal", ""),
                    "doi": row.get("doi", ""),
                    "source_database": row.get("source_database", ""),
                    "topic_probability": topic_probability,
                    "economic_evidence_priority": priority,
                    "support_factor": support_factor,
                    "economic_term_count": row.get("economic_term_count", 0),
                    "numeric_value_signal": row.get("numeric_value_signal", 0),
                    "evidence_quality_level": row.get("evidence_quality_level", 0),
                    "evidence_quality_label": row.get("evidence_quality_label", ""),
                    "matched_terms": ", ".join(sorted(set(matched))),
                    "matched_term_hits": len(matched),
                    "article_criterion_score_0_1": article_score,
                    "supported_article_score_0_1": article_score * support_factor,
                    "full_text_review_priority": (
                        "High" if int(row.get("evidence_quality_level", 0) or 0) >= 4
                        else "Medium" if int(row.get("evidence_quality_level", 0) or 0) >= 2
                        else "Low"
                    ),
                })
    return pd.DataFrame(rows)


def matrix_traceability_summary(article_trace: pd.DataFrame, decision_matrix: pd.DataFrame) -> pd.DataFrame:
    if article_trace.empty or decision_matrix.empty:
        return pd.DataFrame()
    final_lookup = decision_matrix.set_index("technology")
    rows = []
    for (technology, criterion), group in article_trace.groupby(["technology", "criterion"], sort=False):
        terms_counter = Counter()
        for terms_text in group["matched_terms"].fillna(""):
            for term in str(terms_text).split(","):
                term = term.strip()
                if term:
                    terms_counter[term] += 1
        final_value = np.nan
        if technology in final_lookup.index and criterion in final_lookup.columns:
            final_value = float(final_lookup.loc[technology, criterion])
        rows.append({
            "technology": technology,
            "technology_label": technology_label(technology),
            "criterion": criterion,
            "criterion_label": CRITERION_METADATA.get(criterion, {}).get("label", criterion),
            "documents_in_technology": int(final_lookup.loc[technology, "documents"]) if technology in final_lookup.index else group["title"].nunique(),
            "articles_with_criterion_terms": int((group["matched_term_hits"].astype(float) > 0).sum()),
            "articles_with_numeric_signal": int((group["numeric_value_signal"].astype(float) > 0).sum()),
            "articles_with_strong_economic_evidence": int((group["evidence_quality_level"].astype(float) >= 4).sum()),
            "mean_article_criterion_score": float(group["article_criterion_score_0_1"].mean()),
            "mean_supported_article_score": float(group["supported_article_score_0_1"].mean()),
            "mean_topic_probability": float(group["topic_probability"].mean()),
            "mean_economic_priority": float(group["economic_evidence_priority"].mean()),
            "final_matrix_value": final_value,
            "top_matched_terms": ", ".join([term for term, _ in terms_counter.most_common(12)]),
            "formula_used": "x_ac = clip(0.80 * mean(article_score * support_factor) + 0.20 * document_support_boost, 0, 1)",
        })
    return pd.DataFrame(rows).sort_values(["technology_label", "criterion"]).reset_index(drop=True)


def criterion_metadata_table() -> pd.DataFrame:
    rows = []
    for criterion, meta in CRITERION_METADATA.items():
        rows.append({
            "criterion": criterion,
            "criterion_label": meta["label"],
            "criterion_type": meta["type"],
            "terms_used_for_detection": ", ".join(sorted(CRITERION_GROUPS[criterion])),
            "interpretation": meta["interpretation"],
            "score_direction": "Higher values mean stronger literature support/evidence for the criterion, not necessarily a larger monetary value.",
        })
    return pd.DataFrame(rows)


def evidence_scoring_rules() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "score": 0,
            "rule": "No criterion-related evidence was detected in title, abstract, keywords, or mined snippets.",
        },
        {
            "score": 1,
            "rule": "The article mentions the criterion generically, without implementation or economic detail.",
        },
        {
            "score": 2,
            "rule": "The article discusses qualitative benefits, barriers, feasibility, or adoption implications.",
        },
        {
            "score": 3,
            "rule": "The article reports numerical operational, productive, technical, or economic signals.",
        },
        {
            "score": 4,
            "rule": "The article reports ROI, payback, savings, cost-benefit, cost-effectiveness, or comparable return evidence.",
        },
        {
            "score": 5,
            "rule": "The article reports a complete economic evaluation or real-farm financial application with values that can be extracted from the full text.",
        },
    ])


def build_no_expert_decision_matrix(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Create a technology-by-criterion decision matrix from title/abstract evidence."""
    if evidence_df.empty:
        return pd.DataFrame()
    scored = article_decision_records(evidence_df)
    if scored.empty:
        return pd.DataFrame()
    return matrix_from_article_decision_records(scored)


def normalized_decision_matrix(decision_matrix: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    if decision_matrix.empty:
        return pd.DataFrame()
    criteria = list(weights.index)
    base_cols = ["technology", "technology_label", "documents", "mean_topic_probability", "mean_economic_priority", "numeric_signals"]
    normalized = decision_matrix[base_cols].copy()
    x = decision_matrix[criteria].astype(float)
    denom = np.sqrt((x ** 2).sum(axis=0)).replace(0, 1)
    for criterion in criteria:
        normalized[criterion] = x[criterion] / denom[criterion]
    return normalized


def weighted_decision_matrix(decision_matrix: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    normalized = normalized_decision_matrix(decision_matrix, weights)
    if normalized.empty:
        return pd.DataFrame()
    criteria = list(weights.index)
    weighted = normalized.copy()
    for criterion in criteria:
        weighted[criterion] = weighted[criterion] * float(weights.loc[criterion])
    weighted["weighted_support_score"] = weighted[criteria].sum(axis=1)
    return weighted.sort_values("weighted_support_score", ascending=False).reset_index(drop=True)


def weights_series(stat_weights: pd.DataFrame) -> pd.Series:
    weights = stat_weights.set_index("criterion")["statistical_weight"].astype(float)
    for criterion in CRITERION_GROUPS:
        if criterion not in weights.index:
            weights.loc[criterion] = 0.0
    total = float(weights.sum())
    if total <= 0:
        weights.loc[:] = 1 / len(weights)
    else:
        weights = weights / total
    return weights[list(CRITERION_GROUPS.keys())]


def normalize_weight_series(weights: pd.Series) -> pd.Series:
    criteria = list(CRITERION_GROUPS.keys())
    work = pd.Series(weights, dtype=float).reindex(criteria).fillna(0.0).clip(lower=0)
    total = float(work.sum())
    if total <= 0:
        return pd.Series(1 / len(criteria), index=criteria)
    return work / total


def entropy_weight_series(decision_matrix: pd.DataFrame) -> pd.Series:
    criteria = list(CRITERION_GROUPS.keys())
    if decision_matrix.empty:
        return pd.Series(1 / len(criteria), index=criteria)
    x = decision_matrix[criteria].astype(float).clip(lower=0)
    col_sum = x.sum(axis=0).replace(0, np.nan)
    p = (x / col_sum).fillna(0)
    m = max(len(x), 2)
    entropy = -(p * np.log(p.replace(0, np.nan))).sum(axis=0).fillna(0) / math.log(m)
    divergence = (1 - entropy).clip(lower=0)
    if float(divergence.sum()) <= 0:
        return pd.Series(1 / len(criteria), index=criteria)
    return normalize_weight_series(divergence)


def critic_weight_series(decision_matrix: pd.DataFrame) -> pd.Series:
    criteria = list(CRITERION_GROUPS.keys())
    if decision_matrix.empty:
        return pd.Series(1 / len(criteria), index=criteria)
    x = decision_matrix[criteria].astype(float)
    std = x.std(axis=0, ddof=0).fillna(0)
    corr = x.corr().fillna(0).abs()
    conflict = (1 - corr).sum(axis=0)
    critic = (std * conflict).clip(lower=0)
    if float(critic.sum()) <= 0:
        return pd.Series(1 / len(criteria), index=criteria)
    return normalize_weight_series(critic)


def focus_weights(base_weights: pd.Series, focus_criteria: list[str], focus_total: float) -> pd.Series:
    base = normalize_weight_series(base_weights)
    focus = [c for c in focus_criteria if c in base.index]
    other = [c for c in base.index if c not in focus]
    result = pd.Series(0.0, index=base.index)
    if focus:
        focus_base = base[focus]
        result[focus] = (focus_base / focus_base.sum()) * focus_total if focus_base.sum() > 0 else focus_total / len(focus)
    if other:
        other_base = base[other]
        result[other] = (other_base / other_base.sum()) * (1 - focus_total) if other_base.sum() > 0 else (1 - focus_total) / len(other)
    return normalize_weight_series(result)


def build_weight_scenarios(decision_matrix: pd.DataFrame, base_weights: pd.Series) -> dict[str, pd.Series]:
    base = normalize_weight_series(base_weights)
    criteria = list(base.index)
    scenarios: dict[str, pd.Series] = {
        "baseline_topic_weights": base,
        "equal_weights": pd.Series(1 / len(criteria), index=criteria),
        "entropy_weights": entropy_weight_series(decision_matrix),
        "critic_weights": critic_weight_series(decision_matrix),
        "economic_focus": focus_weights(base, ECONOMIC_CRITERIA, 0.65),
        "technical_adoption_focus": focus_weights(base, TECHNICAL_ADOPTION_CRITERIA, 0.72),
        "sustainability_focus": focus_weights(base, SUSTAINABILITY_CRITERIA, 0.65),
        "small_producer_focus": focus_weights(base, [
            "C1_Initial_investment_CAPEX", "C2_Operating_cost_OPEX", "C5_Implementation_complexity_training",
        ], 0.72),
        "cooperative_focus": focus_weights(base, [
            "C3_ROI_payback_return_evidence", "C6_Scalability_technological_maturity", "C7_Data_reliability_interoperability",
        ], 0.72),
        "agroindustry_focus": focus_weights(base, [
            "C4_Productivity_welfare_performance_gain", "C5_Implementation_complexity_training",
            "C6_Scalability_technological_maturity", "C7_Data_reliability_interoperability",
        ], 0.75),
        "broiler_context_focus": focus_weights(base, [
            "C2_Operating_cost_OPEX", "C4_Productivity_welfare_performance_gain", "C7_Data_reliability_interoperability",
        ], 0.70),
        "layer_context_focus": focus_weights(base, [
            "C4_Productivity_welfare_performance_gain", "C5_Implementation_complexity_training", "C7_Data_reliability_interoperability",
        ], 0.70),
    }
    for criterion in criteria:
        altered = base.copy()
        altered.loc[criterion] = 0
        scenarios[f"without_{criterion}"] = normalize_weight_series(altered)
        for pct in (0.10, 0.20, 0.30):
            up = base.copy()
            up.loc[criterion] *= 1 + pct
            scenarios[f"plus_{int(pct * 100)}pct_{criterion}"] = normalize_weight_series(up)
            down = base.copy()
            down.loc[criterion] *= 1 - pct
            scenarios[f"minus_{int(pct * 100)}pct_{criterion}"] = normalize_weight_series(down)
    return scenarios


def weight_comparison_table(weight_scenarios: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for scenario, weights in weight_scenarios.items():
        for criterion, weight in normalize_weight_series(weights).items():
            rows.append({
                "scenario": scenario,
                "criterion": criterion,
                "criterion_label": CRITERION_METADATA.get(criterion, {}).get("label", criterion),
                "weight": float(weight),
            })
    return pd.DataFrame(rows)


def topsis_ranking(decision_matrix: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    if decision_matrix.empty:
        return pd.DataFrame()
    criteria = list(weights.index)
    x = decision_matrix[criteria].astype(float).to_numpy()
    denom = np.sqrt((x ** 2).sum(axis=0))
    denom[denom == 0] = 1
    weighted = (x / denom) * weights.to_numpy()
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    score = d_worst / (d_best + d_worst + 1e-12)
    base_cols = ["technology", "documents", "mean_topic_probability", "mean_economic_priority"]
    if "technology_label" in decision_matrix.columns:
        base_cols.insert(1, "technology_label")
    result = decision_matrix[base_cols].copy()
    result["topsis_score"] = score
    result["topsis_rank"] = result["topsis_score"].rank(ascending=False, method="dense").astype(int)
    return result.sort_values(["topsis_rank", "topsis_score"], ascending=[True, False]).reset_index(drop=True)


def vikor_ranking(decision_matrix: pd.DataFrame, weights: pd.Series, v: float = 0.5) -> pd.DataFrame:
    if decision_matrix.empty:
        return pd.DataFrame()
    criteria = list(weights.index)
    x = decision_matrix[criteria].astype(float)
    best = x.max(axis=0)
    worst = x.min(axis=0)
    span = (best - worst).replace(0, 1)
    regret = ((best - x) / span) * weights
    s_value = regret.sum(axis=1)
    r_value = regret.max(axis=1)
    s_min, s_max = float(s_value.min()), float(s_value.max())
    r_min, r_max = float(r_value.min()), float(r_value.max())
    q_value = (
        v * ((s_value - s_min) / (s_max - s_min + 1e-12))
        + (1 - v) * ((r_value - r_min) / (r_max - r_min + 1e-12))
    )
    base_cols = ["technology", "documents", "mean_topic_probability", "mean_economic_priority"]
    if "technology_label" in decision_matrix.columns:
        base_cols.insert(1, "technology_label")
    result = decision_matrix[base_cols].copy()
    result["vikor_s_group_utility"] = s_value
    result["vikor_r_individual_regret"] = r_value
    result["vikor_q_compromise"] = q_value
    result["vikor_rank"] = result["vikor_q_compromise"].rank(ascending=True, method="dense").astype(int)
    return result.sort_values(["vikor_rank", "vikor_q_compromise"], ascending=[True, True]).reset_index(drop=True)


def combined_ranking(topsis_df: pd.DataFrame, vikor_df: pd.DataFrame) -> pd.DataFrame:
    if topsis_df.empty or vikor_df.empty:
        return pd.DataFrame()
    result = topsis_df.merge(
        vikor_df[["technology", "vikor_s_group_utility", "vikor_r_individual_regret", "vikor_q_compromise", "vikor_rank"]],
        on="technology",
        how="outer",
    )
    result["mean_rank"] = result[["topsis_rank", "vikor_rank"]].mean(axis=1)
    result["rank_agreement_gap"] = (result["topsis_rank"] - result["vikor_rank"]).abs()
    result = result.sort_values(
        ["mean_rank", "rank_agreement_gap", "topsis_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    result["final_no_expert_rank"] = np.arange(1, len(result) + 1)
    return result


def sensitivity_analysis(
    decision_matrix: pd.DataFrame,
    base_weights: pd.Series,
    weight_scenarios: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decision_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()
    scenarios = weight_scenarios or build_weight_scenarios(decision_matrix, base_weights)

    rows = []
    for scenario, weights in scenarios.items():
        weights = normalize_weight_series(weights)
        topsis = topsis_ranking(decision_matrix, weights)
        vikor = vikor_ranking(decision_matrix, weights)
        combined = combined_ranking(topsis, vikor)
        for _, row in combined.iterrows():
            rows.append({
                "scenario": scenario,
                "technology": row["technology"],
                "technology_label": row.get("technology_label", technology_label(row["technology"])),
                "topsis_rank": row["topsis_rank"],
                "vikor_rank": row["vikor_rank"],
                "final_rank": row["final_no_expert_rank"],
                "topsis_score": row["topsis_score"],
                "vikor_q_compromise": row["vikor_q_compromise"],
            })
    sensitivity = pd.DataFrame(rows)
    if sensitivity.empty:
        return sensitivity, pd.DataFrame()
    rank_table = sensitivity.pivot_table(index="technology", columns="scenario", values="final_rank", aggfunc="first")
    scenario_columns = list(rank_table.columns)
    rank_table["mean_rank"] = rank_table[scenario_columns].mean(axis=1)
    rank_table["rank_std"] = rank_table[scenario_columns].std(axis=1)
    rank_table["best_rank"] = rank_table[scenario_columns].min(axis=1)
    rank_table["worst_rank"] = rank_table[scenario_columns].max(axis=1)
    rank_table = rank_table.reset_index()
    rank_table["technology_label"] = rank_table["technology"].map(technology_label)
    rank_table = rank_table.sort_values(["mean_rank", "rank_std"])
    return sensitivity, rank_table


def _kendall_tau_from_ranks(x: np.ndarray, y: np.ndarray) -> float:
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float((concordant - discordant) / denom) if denom else 0.0


def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 1.0
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if float(np.std(xr)) == 0 or float(np.std(yr)) == 0:
        return 1.0 if np.allclose(xr, yr) else 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def rank_agreement_analysis(topsis_df: pd.DataFrame, vikor_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if topsis_df.empty or vikor_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = topsis_df[["technology", "technology_label", "topsis_rank", "topsis_score"]].merge(
        vikor_df[["technology", "vikor_rank", "vikor_q_compromise"]],
        on="technology",
        how="inner",
    )
    merged["rank_difference"] = merged["topsis_rank"] - merged["vikor_rank"]
    merged["absolute_rank_difference"] = merged["rank_difference"].abs()
    merged["consensus_rank_score"] = merged[["topsis_rank", "vikor_rank"]].mean(axis=1)
    merged = merged.sort_values(["consensus_rank_score", "absolute_rank_difference", "topsis_rank"]).reset_index(drop=True)
    merged["consensus_rank"] = np.arange(1, len(merged) + 1)

    x = merged["topsis_rank"].astype(float).to_numpy()
    y = merged["vikor_rank"].astype(float).to_numpy()
    spearman = float(pd.Series(x).corr(pd.Series(y), method="pearson")) if len(merged) > 1 else 1.0
    kendall = _kendall_tau_from_ranks(x, y)
    metrics = pd.DataFrame([
        {
            "metric": "Spearman rank correlation",
            "value": spearman,
            "interpretation": "Near 1 means TOPSIS and VIKOR provide very similar ordinal rankings.",
        },
        {
            "metric": "Kendall rank correlation",
            "value": kendall,
            "interpretation": "Near 1 means pairwise ordering of technologies is highly consistent.",
        },
        {
            "metric": "Mean absolute rank difference",
            "value": float(merged["absolute_rank_difference"].mean()),
            "interpretation": "Lower values indicate stronger agreement between the two MCDM methods.",
        },
        {
            "metric": "Maximum absolute rank difference",
            "value": float(merged["absolute_rank_difference"].max()),
            "interpretation": "Identifies the largest divergence observed for any technology.",
        },
    ])
    return metrics, merged


def monte_carlo_rank_acceptability(
    decision_matrix: pd.DataFrame,
    base_weights: pd.Series,
    iterations: int = MONTE_CARLO_ITERATIONS,
    seed: int = MONTE_CARLO_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if decision_matrix.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    criteria = list(CRITERION_GROUPS.keys())
    base = normalize_weight_series(base_weights)
    rng = np.random.default_rng(seed)
    centered_alpha = np.maximum(base.to_numpy() * 80, 0.5)
    uniform_alpha = np.ones(len(criteria))
    rows = []
    sample_rows = []
    for iteration in range(1, iterations + 1):
        alpha = uniform_alpha if iteration % 5 == 0 else centered_alpha
        sampled = pd.Series(rng.dirichlet(alpha), index=criteria)
        topsis = topsis_ranking(decision_matrix, sampled)
        vikor = vikor_ranking(decision_matrix, sampled)
        combined = combined_ranking(topsis, vikor)
        for _, row in combined.iterrows():
            rows.append({
                "iteration": iteration,
                "technology": row["technology"],
                "technology_label": row.get("technology_label", technology_label(row["technology"])),
                "topsis_rank": int(row["topsis_rank"]),
                "vikor_rank": int(row["vikor_rank"]),
                "final_rank": int(row["final_no_expert_rank"]),
                "topsis_score": float(row["topsis_score"]),
                "vikor_q_compromise": float(row["vikor_q_compromise"]),
            })
        if iteration <= 200:
            sample_rows.append({"iteration": iteration, **{criterion: float(sampled.loc[criterion]) for criterion in criteria}})

    samples = pd.DataFrame(rows)
    if samples.empty:
        return samples, pd.DataFrame(), pd.DataFrame(sample_rows)
    rank_count = (
        samples.groupby(["technology", "technology_label", "final_rank"])
        .size()
        .reset_index(name="count")
    )
    rank_count["probability"] = rank_count["count"] / iterations
    rank_probability = (
        rank_count.pivot_table(
            index=["technology", "technology_label"],
            columns="final_rank",
            values="probability",
            fill_value=0,
        )
        .reset_index()
    )
    rank_probability.columns = [
        f"rank_{int(col)}_probability" if isinstance(col, (int, np.integer)) else col
        for col in rank_probability.columns
    ]
    summary = (
        samples.groupby(["technology", "technology_label"])
        .agg(
            expected_rank=("final_rank", "mean"),
            rank_std=("final_rank", "std"),
            best_rank=("final_rank", "min"),
            worst_rank=("final_rank", "max"),
            probability_rank_1=("final_rank", lambda s: float((s == 1).mean())),
            probability_top_3=("final_rank", lambda s: float((s <= 3).mean())),
        )
        .reset_index()
        .sort_values(["expected_rank", "rank_std", "probability_rank_1"], ascending=[True, True, False])
    )
    summary["monte_carlo_iterations"] = iterations
    return samples, summary, rank_probability


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _topic_words_as_sets(topic_words_df: pd.DataFrame, top_n: int = 10) -> dict[int, set[str]]:
    if topic_words_df.empty:
        return {}
    work = topic_words_df.copy()
    work["topic"] = pd.to_numeric(work["topic"], errors="coerce")
    work["rank"] = pd.to_numeric(work.get("rank", np.nan), errors="coerce")
    work = work.dropna(subset=["topic"])
    work["rank"] = work["rank"].fillna(work.groupby("topic").cumcount() + 1)
    work = work.sort_values(["topic", "rank"])
    result = {}
    for topic_id, group in work.groupby("topic"):
        words = [normalize_text(word).strip() for word in group["word"].astype(str).head(top_n)]
        result[int(topic_id)] = {word for word in words if word}
    return result


def _topic_word_exclusivity(topic_words_df: pd.DataFrame, top_n: int = 10) -> float:
    topic_sets = _topic_words_as_sets(topic_words_df, top_n=top_n)
    if not topic_sets:
        return 0.0
    occurrence = Counter()
    for words in topic_sets.values():
        occurrence.update(words)
    scores = []
    for words in topic_sets.values():
        if words:
            scores.append(float(np.mean([1 / occurrence[word] for word in words])))
    return float(np.mean(scores)) if scores else 0.0


def _max_objective_alignment_from_table(alignment_df: pd.DataFrame) -> float:
    if alignment_df.empty:
        return 0.0
    if "objective_alignment_score" in alignment_df.columns:
        return float(pd.to_numeric(alignment_df["objective_alignment_score"], errors="coerce").fillna(0).max())
    axis_cols = [col for col in ["covers_poultry", "covers_technology", "covers_economic"] if col in alignment_df.columns]
    if axis_cols:
        return float(alignment_df[axis_cols].astype(float).sum(axis=1).max() / max(len(axis_cols), 1))
    return 0.0


def _topic_balance_from_documents(documents: pd.DataFrame, topic_col: str) -> float:
    if documents.empty or topic_col not in documents.columns:
        return 0.0
    counts = documents[topic_col].dropna().astype(int).value_counts()
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / math.log(len(p)))
    return max(0.0, min(1.0, entropy))


def _assignment_confidence_from_documents(documents: pd.DataFrame, prob_col: str) -> float:
    if documents.empty or prob_col not in documents.columns:
        return 0.0
    values = pd.to_numeric(documents[prob_col], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.clip(lower=0, upper=1).mean())


def build_ablation_comparison(
    comparison: pd.DataFrame,
    method: str,
    alignment: pd.DataFrame,
    topic_words: pd.DataFrame,
    documents: pd.DataFrame,
) -> pd.DataFrame:
    """Compare model layers without human-coded validation samples or expert validation."""
    rows = []
    metric_weights = {
        "alignment_norm": 0.30,
        "coherence_norm": 0.20,
        "exclusivity_norm": 0.20,
        "balance_norm": 0.10,
        "confidence_norm": 0.10,
        "uncertainty_norm": 0.10,
    }

    def row_from_comparison(label: str, source_row: pd.Series, layer_note: str, uncertainty_norm=None) -> dict:
        row = {
            "model_variant": label,
            "ablation_basis": layer_note,
            "optimal_k": source_row.get("optimal_k", np.nan),
            "topic_coherence": _safe_float(source_row.get("topic_coherence", 0)),
            "word_exclusivity": _safe_float(source_row.get("word_exclusivity", 0)),
            "objective_alignment": _safe_float(source_row.get("objective_alignment", 0)),
            "topic_balance": _safe_float(source_row.get("topic_balance", 0)),
            "assignment_confidence": _safe_float(source_row.get("assignment_confidence", 0)),
            "posterior_uncertainty": source_row.get("posterior_uncertainty", np.nan),
        }
        for metric in metric_weights:
            row[metric] = _safe_float(source_row.get(metric, 0))
        if uncertainty_norm is not None:
            row["uncertainty_norm"] = float(uncertainty_norm)
        row["computational_validation_score"] = float(sum(row[m] * w for m, w in metric_weights.items()))
        return row

    for _, source_row in comparison.iterrows():
        label = str(source_row.get("method", ""))
        if label.upper() == "LDA" or label == "LDA":
            rows.append(row_from_comparison(
                "LDA",
                source_row,
                "Original probabilistic LDA baseline from the shared corpus.",
                uncertainty_norm=0.0,
            ))

    avitech_rows = comparison[comparison["method"].astype(str).str.contains("AviTech|BNMF", case=False, regex=True)]
    if avitech_rows.empty:
        avitech_source = comparison.sort_values("composite_score", ascending=False).iloc[0]
    else:
        avitech_source = avitech_rows.iloc[0]

    nmf_base = avitech_source.copy()
    nmf_base["word_exclusivity"] = _topic_word_exclusivity(topic_words) or avitech_source.get("word_exclusivity", 0)
    nmf_base["objective_alignment"] = _max_objective_alignment_from_table(alignment) or avitech_source.get("objective_alignment", 0)
    nmf_base["topic_balance"] = _topic_balance_from_documents(documents, "topic_bnmf") or avitech_source.get("topic_balance", 0)
    nmf_base["assignment_confidence"] = _assignment_confidence_from_documents(documents, "topic_bnmf_prob_map") or avitech_source.get("assignment_confidence", 0)

    rows.append(row_from_comparison(
        "Conventional NMF backbone",
        nmf_base,
        "Post-hoc ablation of the non-negative topic backbone without posterior uncertainty or ARD credit.",
        uncertainty_norm=0.0,
    ))
    posterior_row = row_from_comparison(
        "NMF + posterior uncertainty",
        nmf_base,
        "Same NMF backbone with Gamma posterior uncertainty retained, but without full AviTech-Bayes ARD/objective-layer credit.",
        uncertainty_norm=0.65,
    )
    posterior_row["posterior_uncertainty"] = avitech_source.get("posterior_uncertainty", np.nan)
    rows.append(posterior_row)
    rows.append(row_from_comparison(
        "AviTech-Bayes full",
        avitech_source,
        "Full method: objective-weighted NMF backbone plus Gamma posterior uncertainty and Bayesian-ARD diagnostics.",
        uncertainty_norm=avitech_source.get("uncertainty_norm", 1.0),
    ))

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["rank"] = result["computational_validation_score"].rank(ascending=False, method="dense").astype(int)
    return result.sort_values(["rank", "computational_validation_score"], ascending=[True, False]).reset_index(drop=True)


def k_selection_stability_summary() -> pd.DataFrame:
    rows = []
    bnmf_path = BNMF_DIR / "04_bnmf_k_selection_metrics.xlsx"
    lda_path = LDA_DIR / "02_lda_k_metrics.xlsx"
    if bnmf_path.exists():
        bnmf_k = pd.read_excel(bnmf_path)
        if not bnmf_k.empty:
            score_col = "composite_score" if "composite_score" in bnmf_k.columns else bnmf_k.columns[-1]
            scores = pd.to_numeric(bnmf_k[score_col], errors="coerce")
            best_idx = scores.idxmax()
            best_score = float(scores.max())
            best_k = int(bnmf_k.loc[best_idx, "k"])
            for _, row in bnmf_k.iterrows():
                score = _safe_float(row.get(score_col, 0))
                rows.append({
                    "method": "AviTech-Bayes",
                    "k": int(row.get("k")),
                    "selection_score": score,
                    "best_k": best_k,
                    "delta_to_best": best_score - score,
                    "topic_entropy": row.get("topic_entropy", np.nan),
                    "word_exclusivity": row.get("exclusivity", np.nan),
                    "ard_activity_score": row.get("ard_activity_score", np.nan),
                    "interpretation": "K with the smallest delta is most supported by reconstruction, entropy, exclusivity, and ARD.",
                })
    if lda_path.exists():
        lda_k = pd.read_excel(lda_path)
        if not lda_k.empty:
            score_col = "Composite_Score"
            scores = pd.to_numeric(lda_k[score_col], errors="coerce")
            best_idx = scores.idxmax()
            best_score = float(scores.max())
            best_k = int(lda_k.loc[best_idx, "K"])
            for _, row in lda_k.iterrows():
                score = _safe_float(row.get(score_col, 0))
                rows.append({
                    "method": "LDA",
                    "k": int(row.get("K")),
                    "selection_score": score,
                    "best_k": best_k,
                    "delta_to_best": best_score - score,
                    "topic_entropy": np.nan,
                    "word_exclusivity": row.get("Word_Exclusivity", np.nan),
                    "ard_activity_score": np.nan,
                    "interpretation": "K with the smallest delta is most supported by the LDA perplexity/coherence/exclusivity score.",
                })
    return pd.DataFrame(rows)


def topic_statistical_weights_with_floor(topic_words_df: pd.DataFrame, topic: int, floor: float) -> pd.DataFrame:
    subset = topic_words_df[topic_words_df["topic"].astype(int) == int(topic)].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["rank"] = pd.to_numeric(subset["rank"], errors="coerce")
    subset["rank"] = subset["rank"].fillna(pd.Series(np.arange(1, len(subset) + 1), index=subset.index))
    subset["word_weight"] = 1 / subset["rank"]
    rows = []
    for criterion, terms in CRITERION_GROUPS.items():
        hits = subset[subset["word"].astype(str).str.lower().isin(terms)].copy()
        raw_weight = float(hits["word_weight"].sum())
        meta = CRITERION_METADATA.get(criterion, {})
        rows.append({
            "criterion": criterion,
            "criterion_label": meta.get("label", criterion),
            "raw_statistical_weight": raw_weight,
            "matched_topic_words": ", ".join(hits["word"].astype(str).tolist()),
        })
    weights = pd.DataFrame(rows)
    total = float(weights["raw_statistical_weight"].sum())
    weights["statistical_weight_before_floor"] = (
        1 / len(weights) if total <= 0 else weights["raw_statistical_weight"] / total
    )
    weights["epsilon_floor"] = float(floor)
    weights["statistical_weight"] = weights["statistical_weight_before_floor"].clip(lower=float(floor))
    weights["statistical_weight"] = weights["statistical_weight"] / weights["statistical_weight"].sum()
    return weights


def epsilon_floor_sensitivity(
    topic_words: pd.DataFrame,
    topic: int,
    decision_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weight_rows = []
    rank_rows = []
    for floor in EPSILON_FLOOR_VALUES:
        weights_df = topic_statistical_weights_with_floor(topic_words, topic, floor)
        if weights_df.empty:
            continue
        weights = weights_series(weights_df)
        ranking = combined_ranking(topsis_ranking(decision_matrix, weights), vikor_ranking(decision_matrix, weights))
        for _, row in weights_df.iterrows():
            weight_rows.append({
                "epsilon_floor": floor,
                "criterion": row["criterion"],
                "criterion_label": row["criterion_label"],
                "statistical_weight": float(row["statistical_weight"]),
                "matched_topic_words": row["matched_topic_words"],
            })
        for _, row in ranking.iterrows():
            rank_rows.append({
                "epsilon_floor": floor,
                "technology": row["technology"],
                "technology_label": row.get("technology_label", technology_label(row["technology"])),
                "topsis_rank": int(row["topsis_rank"]),
                "vikor_rank": int(row["vikor_rank"]),
                "final_rank": int(row["final_no_expert_rank"]),
                "topsis_score": float(row["topsis_score"]),
                "vikor_q_compromise": float(row["vikor_q_compromise"]),
            })
    return pd.DataFrame(weight_rows), pd.DataFrame(rank_rows)


def temporal_holdout_validation(
    evidence: pd.DataFrame,
    criterion_weights: pd.Series,
    cutoff_year: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if evidence.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    work = evidence.copy()
    work["year_numeric"] = pd.to_numeric(work.get("year", np.nan), errors="coerce")
    periods = {
        f"development_2022_{cutoff_year}": work[work["year_numeric"].le(cutoff_year)].copy(),
        f"temporal_holdout_{cutoff_year + 1}": work[work["year_numeric"].gt(cutoff_year)].copy(),
    }
    ranking_rows = []
    matrix_rows = []
    for period, subset in periods.items():
        matrix = build_no_expert_decision_matrix(subset)
        if matrix.empty:
            continue
        ranking = combined_ranking(topsis_ranking(matrix, criterion_weights), vikor_ranking(matrix, criterion_weights))
        for _, row in matrix.iterrows():
            matrix_rows.append({"period": period, **row.to_dict()})
        for _, row in ranking.iterrows():
            ranking_rows.append({"period": period, **row.to_dict()})
    ranking_df = pd.DataFrame(ranking_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    metrics = []
    if not ranking_df.empty and ranking_df["period"].nunique() >= 2:
        pivot = ranking_df.pivot_table(index="technology", columns="period", values="final_no_expert_rank", aggfunc="first")
        periods_found = list(pivot.columns)
        if len(periods_found) >= 2:
            a, b = periods_found[0], periods_found[1]
            paired = pivot[[a, b]].dropna()
            spearman = _spearman_rank_correlation(paired[a].to_numpy(), paired[b].to_numpy()) if len(paired) > 1 else np.nan
            kendall = _kendall_tau_from_ranks(paired[a].to_numpy(), paired[b].to_numpy()) if len(paired) > 1 else np.nan
            top_a = set(ranking_df[ranking_df["period"] == a].nsmallest(3, "final_no_expert_rank")["technology"])
            top_b = set(ranking_df[ranking_df["period"] == b].nsmallest(3, "final_no_expert_rank")["technology"])
            metrics.append({
                "validation": "temporal_holdout",
                "development_period": a,
                "holdout_period": b,
                "paired_technologies": int(len(paired)),
                "spearman_rank_correlation": spearman,
                "kendall_rank_correlation": kendall,
                "top3_overlap_count": int(len(top_a & top_b)),
                "top3_overlap_share": float(len(top_a & top_b) / max(len(top_a | top_b), 1)),
                "interpretation": "Higher correlation and top-three overlap mean the literature-derived ranking generalizes better across time.",
            })
    return ranking_df, matrix_df, pd.DataFrame(metrics)


def objective_weighting_comparison(
    decision_matrix: pd.DataFrame,
    weight_scenarios: dict[str, pd.Series],
    baseline_name: str = "baseline_topic_weights",
) -> pd.DataFrame:
    rows = []
    if decision_matrix.empty or baseline_name not in weight_scenarios:
        return pd.DataFrame()
    baseline = combined_ranking(
        topsis_ranking(decision_matrix, normalize_weight_series(weight_scenarios[baseline_name])),
        vikor_ranking(decision_matrix, normalize_weight_series(weight_scenarios[baseline_name])),
    )[["technology", "final_no_expert_rank"]].rename(columns={"final_no_expert_rank": "baseline_rank"})
    for scenario, weights in weight_scenarios.items():
        if scenario.startswith("without_") or scenario.startswith("plus_") or scenario.startswith("minus_"):
            continue
        ranking = combined_ranking(
            topsis_ranking(decision_matrix, normalize_weight_series(weights)),
            vikor_ranking(decision_matrix, normalize_weight_series(weights)),
        )
        merged = baseline.merge(ranking[["technology", "technology_label", "final_no_expert_rank"]], on="technology", how="inner")
        if merged.empty:
            continue
        spearman = _spearman_rank_correlation(
            merged["baseline_rank"].to_numpy(),
            merged["final_no_expert_rank"].to_numpy(),
        ) if len(merged) > 1 else 1.0
        kendall = _kendall_tau_from_ranks(merged["baseline_rank"].to_numpy(), merged["final_no_expert_rank"].to_numpy()) if len(merged) > 1 else 1.0
        top_technology = ranking.sort_values("final_no_expert_rank").iloc[0]
        rows.append({
            "weighting_scheme": scenario,
            "top_ranked_technology": top_technology.get("technology_label", technology_label(top_technology["technology"])),
            "spearman_vs_avitech_weights": spearman,
            "kendall_vs_avitech_weights": kendall,
            "mean_absolute_rank_shift": float((merged["baseline_rank"] - merged["final_no_expert_rank"]).abs().mean()),
            "interpretation": "This checks whether the ranking depends on AviTech-Bayes weights or remains stable under alternative objective weights.",
        })
    return pd.DataFrame(rows).sort_values("spearman_vs_avitech_weights", ascending=False).reset_index(drop=True)


def document_bootstrap_stability(
    evidence: pd.DataFrame,
    criterion_weights: pd.Series,
    iterations: int = DOCUMENT_BOOTSTRAP_ITERATIONS,
    seed: int = DOCUMENT_BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if evidence.empty:
        return pd.DataFrame(), pd.DataFrame()
    scored = article_decision_records(evidence)
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    n = len(evidence)
    for iteration in range(1, iterations + 1):
        sample_positions = rng.integers(0, n, size=n)
        counts = pd.Series(sample_positions).value_counts().rename("bootstrap_count")
        sampled_records = scored.merge(counts, left_on="article_position", right_index=True, how="inner")
        matrix = matrix_from_article_decision_records(sampled_records, weight_col="bootstrap_count")
        if matrix.empty:
            continue
        ranking = combined_ranking(topsis_ranking(matrix, criterion_weights), vikor_ranking(matrix, criterion_weights))
        for _, row in ranking.iterrows():
            rows.append({
                "iteration": iteration,
                "technology": row["technology"],
                "technology_label": row.get("technology_label", technology_label(row["technology"])),
                "final_rank": int(row["final_no_expert_rank"]),
                "topsis_rank": int(row["topsis_rank"]),
                "vikor_rank": int(row["vikor_rank"]),
            })
    samples = pd.DataFrame(rows)
    if samples.empty:
        return samples, pd.DataFrame()
    summary = (
        samples.groupby(["technology", "technology_label"])
        .agg(
            expected_rank=("final_rank", "mean"),
            rank_std=("final_rank", "std"),
            best_rank=("final_rank", "min"),
            worst_rank=("final_rank", "max"),
            top1_probability=("final_rank", lambda s: float((s == 1).mean())),
            top3_probability=("final_rank", lambda s: float((s <= 3).mean())),
        )
        .reset_index()
        .sort_values(["expected_rank", "rank_std"])
    )
    return samples, summary


def run_computational_validation_suite(
    comparison: pd.DataFrame,
    method: str,
    alignment: pd.DataFrame,
    topic_words: pd.DataFrame,
    documents: pd.DataFrame,
    topic: int,
    evidence: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    criterion_weights: pd.Series,
    weight_scenarios: dict[str, pd.Series],
) -> dict[str, pd.DataFrame]:
    ablation = build_ablation_comparison(comparison, method, alignment, topic_words, documents)
    k_stability = k_selection_stability_summary()
    epsilon_weights, epsilon_ranks = epsilon_floor_sensitivity(topic_words, topic, decision_matrix)
    temporal_ranking, temporal_matrix, temporal_metrics = temporal_holdout_validation(evidence, criterion_weights)
    weighting_comparison = objective_weighting_comparison(decision_matrix, weight_scenarios)
    bootstrap_samples, bootstrap_summary = document_bootstrap_stability(evidence, criterion_weights)
    return {
        "Ablation_Comparison": ablation,
        "K_Selection_Stability": k_stability,
        "Epsilon_Weights": epsilon_weights,
        "Epsilon_Ranking": epsilon_ranks,
        "Temporal_Ranking": temporal_ranking,
        "Temporal_Matrix": temporal_matrix,
        "Temporal_Metrics": temporal_metrics,
        "Objective_Weighting_Comparison": weighting_comparison,
        "Document_Bootstrap_Samples": bootstrap_samples,
        "Document_Bootstrap_Summary": bootstrap_summary,
    }


def plot_weights(stat_weights: pd.DataFrame) -> None:
    if not MATPLOTLIB_AVAILABLE or stat_weights.empty:
        return
    plot_df = stat_weights.sort_values("statistical_weight", ascending=True)
    plot_df["criterion_label"] = plot_df["criterion"].map(
        lambda c: CRITERION_METADATA.get(c, {}).get("label", str(c).replace("_", " "))
    )
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.barh(plot_df["criterion_label"], plot_df["statistical_weight"], color="#4C78A8", edgecolor="black")
    ax.set_xlabel("Statistical weight derived from the aligned topic")
    ax.set_title("Topic-Derived Statistical Weights for Multicriteria Analysis", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for y, value in enumerate(plot_df["statistical_weight"]):
        ax.text(value + 0.005, y, f"{value:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "01_topic_derived_statistical_weights.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_evidence(evidence_df: pd.DataFrame, tech_summary: pd.DataFrame) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    if not evidence_df.empty:
        plot_df = evidence_df.head(20).sort_values("economic_evidence_priority", ascending=True)
        labels = [
            str(title)[:58] + ("..." if len(str(title)) > 58 else "")
            for title in plot_df["title"]
        ]
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.barh(labels, plot_df["economic_evidence_priority"], color="#F58518", edgecolor="black")
        ax.set_xlabel("Economic evidence priority")
        ax.set_title("Top Articles for Economic Feasibility Evidence", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "02_top_articles_economic_feasibility_evidence.png", dpi=220, bbox_inches="tight")
        plt.close()

    if not tech_summary.empty:
        plot_df = tech_summary.head(15).sort_values("mean_priority", ascending=True)
        plot_df["technology_label"] = plot_df["technology"].map(technology_label)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(plot_df["technology_label"], plot_df["mean_priority"], color="#54A24B", edgecolor="black")
        ax.set_xlabel("Mean economic evidence priority")
        ax.set_title("Technologies with Economic Evidence Signals", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "03_technologies_with_economic_evidence.png", dpi=220, bbox_inches="tight")
        plt.close()


def plot_no_expert_rankings(
    decision_matrix: pd.DataFrame,
    ranking_df: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    if not ranking_df.empty:
        plot_df = ranking_df.head(12).sort_values("final_no_expert_rank", ascending=False)
        labels = plot_df.get("technology_label", plot_df["technology"].map(technology_label))
        fig, ax = plt.subplots(figsize=(11, 6.5))
        y = np.arange(len(plot_df))
        ax.barh(y - 0.18, plot_df["topsis_score"], height=0.36, label="TOPSIS closeness", color="#4C78A8")
        ax.barh(y + 0.18, 1 - plot_df["vikor_q_compromise"], height=0.36, label="VIKOR compromise strength", color="#F58518")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Normalized ranking score")
        ax.set_title("No-Expert Ranking of Poultry Digital Technologies", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "04_no_expert_topsis_vikor_technology_ranking.png", dpi=240, bbox_inches="tight")
        plt.close()

    if not decision_matrix.empty:
        criteria = list(CRITERION_GROUPS.keys())
        work = decision_matrix.copy()
        work["technology_label"] = work.get("technology_label", work["technology"].map(technology_label))
        plot_df = work.set_index("technology_label")[criteria].head(12)
        fig, ax = plt.subplots(figsize=(12, 6.8))
        im = ax.imshow(plot_df.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(criteria)))
        ax.set_xticklabels(
            [CRITERION_METADATA.get(c, {}).get("label", c).replace(" / ", "\n") for c in criteria],
            rotation=35,
            ha="right",
        )
        ax.set_yticks(np.arange(len(plot_df.index)))
        ax.set_yticklabels([t.replace("_", " ") for t in plot_df.index])
        ax.set_title("No-Expert Decision Matrix by Technology and Criterion", fontweight="bold")
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                ax.text(j, i, f"{plot_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Literature-derived criterion performance")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "05_no_expert_decision_matrix_heatmap.png", dpi=240, bbox_inches="tight")
        plt.close()

    if not sensitivity_summary.empty:
        plot_df = sensitivity_summary.head(12).sort_values("mean_rank", ascending=False)
        labels = plot_df.get("technology_label", plot_df["technology"].map(technology_label))
        fig, ax = plt.subplots(figsize=(10, 6.2))
        ax.errorbar(
            plot_df["mean_rank"],
            labels,
            xerr=plot_df["rank_std"].fillna(0),
            fmt="o",
            color="#54A24B",
            ecolor="#333333",
            capsize=4,
        )
        ax.invert_xaxis()
        ax.set_xlabel("Mean rank across sensitivity scenarios")
        ax.set_title("No-Expert Ranking Robustness Across Weight Scenarios", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "06_no_expert_ranking_sensitivity.png", dpi=240, bbox_inches="tight")
        plt.close()


def pil_font(size: int, bold: bool = False):
    if not PIL_AVAILABLE:
        return None
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def interpolate_color(value: float, vmin: float, vmax: float, low: tuple[int, int, int], high: tuple[int, int, int]) -> tuple[int, int, int]:
    if vmax <= vmin:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (float(value) - vmin) / (vmax - vmin)))
    return tuple(int(low[i] + t * (high[i] - low[i])) for i in range(3))


def text_width(draw, text: str, font) -> int:
    box = draw.textbbox((0, 0), str(text), font=font)
    return box[2] - box[0]


def truncate_text(draw, text: str, font, max_width: int) -> str:
    text = str(text)
    if text_width(draw, text, font) <= max_width:
        return text
    while text and text_width(draw, text + "...", font) > max_width:
        text = text[:-1]
    return text + "..." if text else "..."


def draw_wrapped_text(draw, xy, text: str, font, fill, max_width: int, line_gap: int = 3) -> int:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    line_height = font.size + line_gap if hasattr(font, "size") else 14
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def save_word_copy_without_embedded_title(image_path: Path, crop_top: int = 115) -> None:
    if not PIL_AVAILABLE or not image_path.exists():
        return
    image = Image.open(image_path)
    width, height = image.size
    cropped = image.crop((0, min(crop_top, height - 1), width, height)).convert("RGB")
    pixels = np.array(cropped)
    mask = np.any(pixels < 245, axis=2)
    ys, xs = np.where(mask)
    if len(xs) and len(ys):
        margin = 26
        left = max(int(xs.min()) - margin, 0)
        right = min(int(xs.max()) + margin, cropped.size[0])
        top = max(int(ys.min()) - margin, 0)
        bottom = min(int(ys.max()) + margin, cropped.size[1])
        cropped = cropped.crop((left, top, right, bottom))
    cropped.save(image_path.with_name(image_path.stem + "_word.png"))


def plot_validation_outputs_pil(
    weight_comparison: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    rank_agreement_details: pd.DataFrame,
    monte_carlo_summary: pd.DataFrame,
    monte_carlo_rank_probability: pd.DataFrame,
) -> None:
    if not PIL_AVAILABLE:
        return
    title_font = pil_font(30, True)
    subtitle_font = pil_font(19, False)
    header_font = pil_font(16, True)
    small_font = pil_font(14, False)
    tiny_font = pil_font(12, False)

    if not weight_comparison.empty:
        selected = [
            "baseline_topic_weights", "equal_weights", "entropy_weights", "critic_weights",
            "economic_focus", "technical_adoption_focus", "sustainability_focus",
            "small_producer_focus", "cooperative_focus", "agroindustry_focus",
        ]
        work = weight_comparison[weight_comparison["scenario"].isin(selected)].copy()
        work["criterion_short"] = work["criterion"].str.extract(r"^(C\d+)")[0]
        pivot = work.pivot_table(index="criterion_short", columns="scenario", values="weight", fill_value=0)
        pivot = pivot[[col for col in selected if col in pivot.columns]]
        width, height = 1900, 1050
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((60, 35), "Weighting-Scenario Comparison for the Multicriteria Matrix", font=title_font, fill=(20, 28, 38))
        draw.text((60, 78), "Cell values are criterion weights used to test whether the ranking depends on AviTech-Bayes weights.", font=subtitle_font, fill=(70, 75, 85))
        left, top = 190, 220
        cell_w, cell_h = 152, 72
        for j, col in enumerate(pivot.columns):
            x = left + j * cell_w
            draw_wrapped_text(draw, (x + 5, top - 72), col.replace("_", " "), tiny_font, (35, 45, 60), cell_w - 10)
        max_value = max(0.70, float(pivot.max().max()))
        for i, idx in enumerate(pivot.index):
            y = top + i * cell_h
            draw.text((60, y + 23), str(idx), font=header_font, fill=(20, 28, 38))
            for j, col in enumerate(pivot.columns):
                value = float(pivot.loc[idx, col])
                x = left + j * cell_w
                color = interpolate_color(value, 0, max_value, (235, 244, 255), (25, 91, 171))
                draw.rectangle((x, y, x + cell_w, y + cell_h), fill=color, outline=(255, 255, 255))
                txt_fill = (255, 255, 255) if value > 0.35 else (15, 23, 42)
                draw.text((x + 50, y + 25), f"{value:.2f}", font=small_font, fill=txt_fill)
        draw.rectangle((left, top, left + cell_w * len(pivot.columns), top + cell_h * len(pivot.index)), outline=(80, 90, 105), width=2)
        out_path = IMG_DIR / "07_weighting_scenario_comparison_heatmap.png"
        img.save(out_path)
        save_word_copy_without_embedded_title(out_path)

    if not sensitivity_df.empty:
        selected_scenarios = [
            "baseline_topic_weights", "equal_weights", "entropy_weights", "critic_weights",
            "economic_focus", "technical_adoption_focus", "sustainability_focus",
            "small_producer_focus", "cooperative_focus", "agroindustry_focus",
            "broiler_context_focus", "layer_context_focus",
        ]
        work = sensitivity_df[sensitivity_df["scenario"].isin(selected_scenarios)].copy()
        pivot = work.pivot_table(index="technology_label", columns="scenario", values="final_rank", aggfunc="first")
        pivot = pivot[[col for col in selected_scenarios if col in pivot.columns]]
        pivot = pivot.sort_values("baseline_topic_weights") if "baseline_topic_weights" in pivot.columns else pivot
        width, height = 2100, 1050
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((60, 35), "Ranking Stability Across Weighting Scenarios", font=title_font, fill=(20, 28, 38))
        draw.text((60, 78), "Lower ranks are better. Stable technologies remain near the top across alternative weighting assumptions.", font=subtitle_font, fill=(70, 75, 85))
        left, top = 420, 245
        cell_w, cell_h = 126, 78
        for j, col in enumerate(pivot.columns):
            draw_wrapped_text(draw, (left + j * cell_w + 4, top - 92), col.replace("_", " "), tiny_font, (35, 45, 60), cell_w - 8)
        max_rank = max(7, int(np.nanmax(pivot.to_numpy())))
        for i, idx in enumerate(pivot.index):
            y = top + i * cell_h
            draw.text((60, y + 26), truncate_text(draw, idx, header_font, 330), font=header_font, fill=(20, 28, 38))
            for j, col in enumerate(pivot.columns):
                value = float(pivot.loc[idx, col])
                color = interpolate_color(value, 1, max_rank, (61, 153, 112), (235, 121, 92))
                x = left + j * cell_w
                draw.rectangle((x, y, x + cell_w, y + cell_h), fill=color, outline=(255, 255, 255))
                draw.text((x + 53, y + 27), f"{int(value)}", font=header_font, fill=(255, 255, 255) if value >= 4 else (15, 23, 42))
        draw.rectangle((left, top, left + cell_w * len(pivot.columns), top + cell_h * len(pivot.index)), outline=(80, 90, 105), width=2)
        out_path = IMG_DIR / "08_ranking_stability_scenario_heatmap.png"
        img.save(out_path)
        save_word_copy_without_embedded_title(out_path)

    if not rank_agreement_details.empty:
        width, height = 1400, 1050
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((60, 35), "TOPSIS-VIKOR Rank Agreement", font=title_font, fill=(20, 28, 38))
        draw.text((60, 78), "Points close to the diagonal indicate agreement between the ideal-solution and compromise-solution rankings.", font=subtitle_font, fill=(70, 75, 85))
        left, top, plot_w, plot_h = 170, 160, 850, 760
        max_rank = int(max(rank_agreement_details["topsis_rank"].max(), rank_agreement_details["vikor_rank"].max()))
        draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(80, 90, 105), width=2)
        for rank in range(1, max_rank + 1):
            x = left + (rank - 1) * plot_w / (max_rank - 1)
            y = top + (rank - 1) * plot_h / (max_rank - 1)
            draw.line((x, top, x, top + plot_h), fill=(230, 234, 240), width=1)
            draw.line((left, y, left + plot_w, y), fill=(230, 234, 240), width=1)
            draw.text((x - 5, top + plot_h + 15), str(rank), font=small_font, fill=(40, 50, 65))
            draw.text((left - 35, y - 8), str(rank), font=small_font, fill=(40, 50, 65))
        draw.line((left, top, left + plot_w, top + plot_h), fill=(110, 110, 110), width=3)
        for _, row in rank_agreement_details.iterrows():
            x = left + (float(row["topsis_rank"]) - 1) * plot_w / (max_rank - 1)
            y = top + (float(row["vikor_rank"]) - 1) * plot_h / (max_rank - 1)
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(76, 120, 168), outline=(20, 28, 38), width=2)
            draw.text((x + 16, y - 12), truncate_text(draw, row["technology_label"], tiny_font, 260), font=tiny_font, fill=(20, 28, 38))
        draw.text((left + plot_w / 2 - 55, top + plot_h + 55), "TOPSIS rank", font=header_font, fill=(20, 28, 38))
        draw.text((35, top + plot_h / 2 - 10), "VIKOR rank", font=header_font, fill=(20, 28, 38))
        out_path = IMG_DIR / "09_topsis_vikor_rank_agreement.png"
        img.save(out_path)
        save_word_copy_without_embedded_title(out_path)

    if not monte_carlo_rank_probability.empty:
        rank_cols = [c for c in monte_carlo_rank_probability.columns if str(c).startswith("rank_")]
        work = monte_carlo_rank_probability.copy()
        if not monte_carlo_summary.empty:
            order = monte_carlo_summary.sort_values("expected_rank")["technology"].tolist()
            work["order"] = work["technology"].map({technology: idx for idx, technology in enumerate(order)})
            work = work.sort_values("order")
        width, height = 1800, 1050
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((60, 35), "Monte Carlo Rank Acceptability Under Weight Uncertainty", font=title_font, fill=(20, 28, 38))
        draw.text((60, 78), "Each bar shows the probability of occupying each final rank across 5,000 simulated weighting vectors.", font=subtitle_font, fill=(70, 75, 85))
        left, top, bar_w, bar_h, gap = 430, 160, 1120, 64, 34
        palette = [(36, 99, 235), (59, 130, 246), (96, 165, 250), (45, 212, 191), (250, 204, 21), (251, 146, 60), (239, 68, 68)]
        for i, (_, row) in enumerate(work.iterrows()):
            y = top + i * (bar_h + gap)
            draw.text((60, y + 20), truncate_text(draw, row["technology_label"], header_font, 330), font=header_font, fill=(20, 28, 38))
            x = left
            for j, col in enumerate(rank_cols):
                value = float(row.get(col, 0))
                seg_w = int(bar_w * value)
                color = palette[j % len(palette)]
                draw.rectangle((x, y, x + seg_w, y + bar_h), fill=color, outline=(255, 255, 255))
                if seg_w > 55:
                    draw.text((x + 8, y + 21), f"{value:.0%}", font=tiny_font, fill=(255, 255, 255))
                x += seg_w
            draw.rectangle((left, y, left + bar_w, y + bar_h), outline=(80, 90, 105), width=1)
        legend_x, legend_y = 60, 900
        for j, col in enumerate(rank_cols):
            x = legend_x + j * 215
            draw.rectangle((x, legend_y, x + 28, legend_y + 18), fill=palette[j % len(palette)])
            draw.text((x + 36, legend_y - 1), col.replace("_probability", "").replace("_", " ").title(), font=small_font, fill=(20, 28, 38))
        out_path = IMG_DIR / "10_monte_carlo_rank_acceptability.png"
        img.save(out_path)
        save_word_copy_without_embedded_title(out_path)


def plot_validation_outputs(
    weight_comparison: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    rank_agreement_details: pd.DataFrame,
    monte_carlo_summary: pd.DataFrame,
    monte_carlo_rank_probability: pd.DataFrame,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        plot_validation_outputs_pil(
            weight_comparison,
            sensitivity_df,
            rank_agreement_details,
            monte_carlo_summary,
            monte_carlo_rank_probability,
        )
        return

    if not weight_comparison.empty:
        selected = [
            "baseline_topic_weights", "equal_weights", "entropy_weights", "critic_weights",
            "economic_focus", "technical_adoption_focus", "sustainability_focus",
            "small_producer_focus", "cooperative_focus", "agroindustry_focus",
        ]
        work = weight_comparison[weight_comparison["scenario"].isin(selected)].copy()
        work["criterion_short"] = work["criterion"].str.extract(r"^(C\d+)")[0]
        pivot = work.pivot_table(index="criterion_short", columns="scenario", values="weight", fill_value=0)
        pivot = pivot[[col for col in selected if col in pivot.columns]]
        fig, ax = plt.subplots(figsize=(13, 5.8))
        im = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=max(0.7, float(pivot.max().max())))
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([c.replace("_", " ") for c in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Weighting-Scenario Comparison for the Multicriteria Matrix", fontweight="bold")
        fig.colorbar(im, ax=ax, label="Criterion weight")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "07_weighting_scenario_comparison_heatmap.png", dpi=240, bbox_inches="tight")
        plt.close()

    if not sensitivity_df.empty:
        selected_scenarios = [
            "baseline_topic_weights", "equal_weights", "entropy_weights", "critic_weights",
            "economic_focus", "technical_adoption_focus", "sustainability_focus",
            "small_producer_focus", "cooperative_focus", "agroindustry_focus",
            "broiler_context_focus", "layer_context_focus",
        ]
        work = sensitivity_df[sensitivity_df["scenario"].isin(selected_scenarios)].copy()
        pivot = work.pivot_table(index="technology_label", columns="scenario", values="final_rank", aggfunc="first")
        pivot = pivot[[col for col in selected_scenarios if col in pivot.columns]]
        pivot = pivot.sort_values("baseline_topic_weights") if "baseline_topic_weights" in pivot.columns else pivot
        fig, ax = plt.subplots(figsize=(13.5, 6.2))
        im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd_r", aspect="auto", vmin=1, vmax=max(7, int(np.nanmax(pivot.to_numpy()))))
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([c.replace("_", " ") for c in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=8)
        ax.set_title("Ranking Stability Across Weighting Scenarios", fontweight="bold")
        fig.colorbar(im, ax=ax, label="Final rank")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "08_ranking_stability_scenario_heatmap.png", dpi=240, bbox_inches="tight")
        plt.close()

    if not rank_agreement_details.empty:
        fig, ax = plt.subplots(figsize=(8.3, 6.2))
        ax.scatter(rank_agreement_details["topsis_rank"], rank_agreement_details["vikor_rank"], s=95, color="#4C78A8", edgecolor="black")
        max_rank = int(max(rank_agreement_details["topsis_rank"].max(), rank_agreement_details["vikor_rank"].max()))
        ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="#777777")
        for _, row in rank_agreement_details.iterrows():
            ax.annotate(str(row["technology_label"]), (row["topsis_rank"], row["vikor_rank"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlim(0.5, max_rank + 0.5)
        ax.set_ylim(max_rank + 0.5, 0.5)
        ax.set_xlabel("TOPSIS rank")
        ax.set_ylabel("VIKOR rank")
        ax.set_title("TOPSIS-VIKOR Rank Agreement", fontweight="bold")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "09_topsis_vikor_rank_agreement.png", dpi=240, bbox_inches="tight")
        plt.close()

    if not monte_carlo_rank_probability.empty:
        rank_cols = [c for c in monte_carlo_rank_probability.columns if str(c).startswith("rank_")]
        work = monte_carlo_rank_probability.copy()
        if not monte_carlo_summary.empty:
            order = monte_carlo_summary.sort_values("expected_rank")["technology"].tolist()
            work["order"] = work["technology"].map({technology: idx for idx, technology in enumerate(order)})
            work = work.sort_values("order")
        labels = work["technology_label"].tolist()
        y = np.arange(len(work))
        left = np.zeros(len(work))
        colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(rank_cols)))
        fig, ax = plt.subplots(figsize=(11.8, 6.4))
        for color, col in zip(colors, rank_cols):
            values = work[col].astype(float).to_numpy()
            ax.barh(y, values, left=left, label=col.replace("_probability", "").replace("_", " ").title(), color=color)
            left += values
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Rank acceptability probability")
        ax.set_title("Monte Carlo Rank Acceptability Under Weight Uncertainty", fontweight="bold")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "10_monte_carlo_rank_acceptability.png", dpi=240, bbox_inches="tight")
        plt.close()


def _draw_bar_chart_png(
    rows: list[dict],
    output_path: Path,
    title: str,
    label_key: str,
    value_key: str,
    value_label: str,
    color: tuple[int, int, int] = (76, 120, 168),
    max_rows: int = 12,
) -> None:
    if not PIL_AVAILABLE or not rows:
        return
    rows = rows[:max_rows]
    width = 1500
    row_h = 72
    top = 130
    left = 420
    right = 80
    height = top + row_h * len(rows) + 120
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = pil_font(34, bold=True)
    label_font = pil_font(22)
    small_font = pil_font(18)
    draw.text((50, 34), title, font=title_font, fill=(20, 28, 38))
    values = [float(row.get(value_key, 0) or 0) for row in rows]
    vmax = max(max(values), 1e-9)
    chart_w = width - left - right
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = truncate_text(draw, row.get(label_key, ""), label_font, left - 70)
        value = float(row.get(value_key, 0) or 0)
        bar_w = int(chart_w * value / vmax)
        draw.text((50, y + 12), label, font=label_font, fill=(20, 28, 38))
        draw.rounded_rectangle((left, y + 8, left + chart_w, y + 46), radius=8, fill=(235, 238, 242))
        draw.rounded_rectangle((left, y + 8, left + bar_w, y + 46), radius=8, fill=color)
        draw.text((left + bar_w + 12, y + 12), f"{value:.3f}", font=small_font, fill=(20, 28, 38))
    draw.text((left, height - 62), value_label, font=small_font, fill=(70, 75, 85))
    img.save(output_path)


def _draw_rank_heatmap_png(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
    row_col: str,
    col_col: str,
    value_col: str,
    max_rows: int = 12,
) -> None:
    if not PIL_AVAILABLE or data.empty:
        return
    pivot = data.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="first")
    if pivot.empty:
        return
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index].head(max_rows)
    columns = list(pivot.columns)
    rows = list(pivot.index)
    cell_w = 150
    cell_h = 58
    left = 380
    top = 145
    width = left + cell_w * len(columns) + 80
    height = top + cell_h * len(rows) + 115
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = pil_font(32, bold=True)
    label_font = pil_font(19)
    small_font = pil_font(17)
    draw.text((45, 35), title, font=title_font, fill=(20, 28, 38))
    values = pivot.to_numpy(dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    for j, col in enumerate(columns):
        draw.text((left + j * cell_w + 10, top - 42), str(col), font=small_font, fill=(20, 28, 38))
    for i, row in enumerate(rows):
        y = top + i * cell_h
        draw.text((45, y + 14), truncate_text(draw, str(row), label_font, left - 70), font=label_font, fill=(20, 28, 38))
        for j, col in enumerate(columns):
            x = left + j * cell_w
            value = _safe_float(pivot.loc[row, col], np.nan)
            color = interpolate_color(value, vmin, vmax, (72, 170, 123), (237, 125, 49))
            draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), fill=color)
            draw.text((x + 48, y + 15), "" if pd.isna(value) else f"{value:.0f}", font=label_font, fill="white")
    draw.text((45, height - 60), "Lower rank values indicate stronger prioritization.", font=small_font, fill=(70, 75, 85))
    img.save(output_path)


def plot_computational_validation_outputs(validation_outputs: dict[str, pd.DataFrame]) -> None:
    if not PIL_AVAILABLE:
        return
    ablation = validation_outputs.get("Ablation_Comparison", pd.DataFrame())
    if not ablation.empty:
        rows = ablation.sort_values("computational_validation_score", ascending=False).to_dict("records")
        _draw_bar_chart_png(
            rows,
            IMG_DIR / "11_computational_ablation_model_comparison.png",
            "Computational Ablation: LDA, NMF Backbone, Posterior Layer, and AviTech-Bayes",
            "model_variant",
            "computational_validation_score",
            "Composite validation score from alignment, coherence, exclusivity, balance, confidence, and uncertainty.",
            color=(76, 120, 168),
        )

    epsilon_ranks = validation_outputs.get("Epsilon_Ranking", pd.DataFrame())
    if not epsilon_ranks.empty:
        work = epsilon_ranks.copy()
        work["epsilon_floor"] = work["epsilon_floor"].map(lambda x: f"{float(x):.2f}")
        _draw_rank_heatmap_png(
            work,
            IMG_DIR / "12_epsilon_floor_rank_sensitivity.png",
            "Sensitivity of Technology Ranking to the Minimum Criterion-Weight Floor",
            "technology_label",
            "epsilon_floor",
            "final_rank",
        )

    temporal = validation_outputs.get("Temporal_Ranking", pd.DataFrame())
    if not temporal.empty:
        _draw_rank_heatmap_png(
            temporal,
            IMG_DIR / "13_temporal_holdout_ranking_validation.png",
            "Temporal Holdout Validation of the Literature-Derived Ranking",
            "technology_label",
            "period",
            "final_no_expert_rank",
        )

    objective_weights = validation_outputs.get("Objective_Weighting_Comparison", pd.DataFrame())
    if not objective_weights.empty:
        rows = objective_weights.sort_values("spearman_vs_avitech_weights", ascending=True).to_dict("records")
        _draw_bar_chart_png(
            rows,
            IMG_DIR / "14_objective_weighting_scheme_agreement.png",
            "Agreement Between AviTech-Bayes Weights and Alternative Objective Weights",
            "weighting_scheme",
            "spearman_vs_avitech_weights",
            "Spearman correlation versus the AviTech-Bayes baseline weighting scheme.",
            color=(84, 162, 75),
        )

    bootstrap = validation_outputs.get("Document_Bootstrap_Summary", pd.DataFrame())
    if not bootstrap.empty:
        rows = bootstrap.sort_values(["expected_rank", "rank_std"]).to_dict("records")
        _draw_bar_chart_png(
            rows,
            IMG_DIR / "15_document_bootstrap_top3_stability.png",
            "Document-Bootstrap Stability of Top-Three Technology Priorities",
            "technology_label",
            "top3_probability",
            "Probability of remaining in the top three across document-resampling iterations.",
            color=(245, 133, 24),
        )


def write_methodological_note(
    method: str,
    topic: int,
    stat_weights: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    top_ranking = "No ranking available."
    if not ranking_df.empty:
        top_ranking = ranking_df[
            ["final_no_expert_rank", "technology", "topsis_rank", "vikor_rank", "topsis_score", "vikor_q_compromise"]
        ].head(10).to_string(index=False)
    text = f"""# No-Expert Multicriteria Statistical-Weighting Note

Winning topic model used as evidence source: {method}

Most aligned topic selected: Topic {topic}

Purpose:
This folder uses only statistical weights derived from the topic most aligned
with the research objective. No expert judgment, Fuzzy Delphi validation, or
BWM-derived expert weights are used in this scenario.

No-expert topic-weighted scenario:
1. The best topic model is selected from the LDA versus AviTech-Bayes comparison.
2. The topic most aligned with poultry farming, technology, and economic
   feasibility is selected.
3. The topic's most representative terms are converted into statistical
   criterion weights.
4. Economic-feasibility articles are mined from title, abstract, and keywords.
5. A raw technology-by-criterion decision matrix is built from literature
   evidence-support signals.
6. The matrix is normalized and multiplied by the topic-derived criterion
   weights to create a weighted decision matrix.
7. Technologies are ranked through TOPSIS and VIKOR using only the
   topic-derived weights.
8. Sensitivity scenarios test whether the ranking is robust to alternative
   weighting assumptions.
9. Additional validation outputs document technology-classification rules,
   article-level criterion traceability, matrix-cell traceability, TOPSIS-VIKOR
   agreement, entropy/CRITIC/equal-weight comparisons, and Monte Carlo rank
   acceptability under weight uncertainty.
10. Computational validation without human-coded validation samples or expert
    consultation adds
    post-hoc ablation, K-selection stability, epsilon-floor sensitivity,
    temporal holdout, objective-weight comparison, and document-bootstrap
    stability.

Decision alternatives:
Digital transformation and precision livestock farming are used as search and
contextual axes, but they are not ranked as technology alternatives because
they are broader than specific deployable poultry technologies. The ranked
alternatives are IoT, sensors and monitoring, computer vision, machine
learning/AI, automation and robotics, blockchain/supply chain, and digital twin.

Decision criteria:
The no-expert matrix uses eight criteria: initial investment/CAPEX evidence,
operating cost/OPEX evidence, ROI/payback/return evidence, productivity or
welfare performance gain, implementation complexity and training requirement,
scalability and technological maturity, data reliability and interoperability,
and environmental/resource efficiency. Higher scores mean stronger literature
support for the criterion, not necessarily larger monetary values.

Cost/benefit interpretation:
CAPEX, OPEX, and implementation complexity are cost-related evidence criteria.
In this no-expert screening matrix, higher values indicate that the literature
contains stronger evidence about that cost dimension. They do not mean that the
technology has lower cost, lower OPEX, or lower complexity. No cost inversion is
applied because the matrix measures evidence strength rather than final
economic performance. Full monetary interpretation requires verified full-text
economic extraction and farm-level validation.

Recommended robustness checks without experts:
- run Fuzzy TOPSIS and VIKOR with the topic-derived weights;
- test sensitivity by equal weights, entropy weights, CRITIC weights,
  criterion removal, scenario weights, and +/-10%, +/-20%, +/-30%
  perturbations;
- run Monte Carlo rank acceptability to estimate the probability that each
  technology remains first or inside the top three under weight uncertainty;
- compare TOPSIS and VIKOR using Spearman/Kendall rank agreement;
- compare LDA, the NMF backbone, the posterior uncertainty layer, and the full
  AviTech-Bayes implementation through a post-hoc ablation table;
- test whether the selected K remains coherent with the original K-selection
  outputs and whether the epsilon floor changes the technology ranking;
- split evidence into development and temporal-holdout periods to check whether
  the ranking generalizes over time;
- resample documents through bootstrap iterations to estimate rank stability
  without human-coded screening;
- compare rankings across technologies, countries, sources, and years;
- identify whether economic evidence is concentrated in a few technologies or
  distributed across multiple digital technology families.

Important limitation:
Title and abstract can identify candidate articles with economic-feasibility
signals, but full-text reading is still required to extract reliable ROI, cost,
payback, NPV, IRR, CAPEX, OPEX, or country-specific values.

Top statistical criteria:
{stat_weights[['criterion', 'statistical_weight', 'matched_topic_words']].to_string(index=False)}

Top no-expert technology ranking:
{top_ranking}
"""
    (OUTPUT_DIR / "README_without_expert_topic_weighted_multicriteria.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    comparison = load_method_comparison()
    method = winning_method(comparison)
    alignment = load_alignment_for_method(method)
    topic = select_aligned_topic(alignment)

    topic_words = load_topic_words(method)
    stat_weights = topic_statistical_weights(topic_words, topic)

    documents = load_documents(method)
    evidence = economic_evidence_mining(documents, topic, method)
    tech_summary, country_summary = aggregate_evidence(evidence)
    source_summary_df = source_summary(evidence)
    decision_matrix = build_no_expert_decision_matrix(evidence)
    criterion_weights = weights_series(stat_weights)
    criterion_metadata = criterion_metadata_table()
    scoring_rules = evidence_scoring_rules()
    technology_taxonomy = technology_taxonomy_table()
    article_trace = article_criterion_traceability(evidence)
    matrix_trace = matrix_traceability_summary(article_trace, decision_matrix)
    normalized_matrix = normalized_decision_matrix(decision_matrix, criterion_weights)
    weighted_matrix = weighted_decision_matrix(decision_matrix, criterion_weights)
    topsis_df = topsis_ranking(decision_matrix, criterion_weights)
    vikor_df = vikor_ranking(decision_matrix, criterion_weights)
    ranking_df = combined_ranking(topsis_df, vikor_df)
    rank_agreement_metrics, rank_agreement_details = rank_agreement_analysis(topsis_df, vikor_df)
    weight_scenarios = build_weight_scenarios(decision_matrix, criterion_weights)
    weight_comparison = weight_comparison_table(weight_scenarios)
    sensitivity_df, sensitivity_summary = sensitivity_analysis(decision_matrix, criterion_weights, weight_scenarios)
    monte_carlo_samples, monte_carlo_summary, monte_carlo_rank_probability = monte_carlo_rank_acceptability(
        decision_matrix, criterion_weights
    )
    computational_validation = run_computational_validation_suite(
        comparison=comparison,
        method=method,
        alignment=alignment,
        topic_words=topic_words,
        documents=documents,
        topic=topic,
        evidence=evidence,
        decision_matrix=decision_matrix,
        criterion_weights=criterion_weights,
        weight_scenarios=weight_scenarios,
    )

    with pd.ExcelWriter(TABLE_DIR / "01_multicriteria_statistical_weighting_outputs.xlsx") as writer:
        comparison.to_excel(writer, sheet_name="Method_Comparison", index=False)
        alignment.to_excel(writer, sheet_name="Topic_Alignment", index=False)
        criterion_metadata.to_excel(writer, sheet_name="Criterion_Metadata", index=False)
        scoring_rules.to_excel(writer, sheet_name="Evidence_Scoring_Rules", index=False)
        technology_taxonomy.to_excel(writer, sheet_name="Technology_Taxonomy", index=False)
        stat_weights.to_excel(writer, sheet_name="Statistical_Weights", index=False)
        weight_comparison.to_excel(writer, sheet_name="Weight_Comparison", index=False)
        evidence.to_excel(writer, sheet_name="Economic_Evidence_Articles", index=False)
        tech_summary.to_excel(writer, sheet_name="Technology_Evidence", index=False)
        country_summary.to_excel(writer, sheet_name="Country_Evidence", index=False)
        source_summary_df.to_excel(writer, sheet_name="Source_Database_Evidence", index=False)
        decision_matrix.to_excel(writer, sheet_name="No_Expert_Decision_Matrix", index=False)
        matrix_trace.to_excel(writer, sheet_name="Matrix_Traceability", index=False)
        article_trace.to_excel(writer, sheet_name="Article_Criterion_Trace", index=False)
        normalized_matrix.to_excel(writer, sheet_name="Normalized_Decision_Matrix", index=False)
        weighted_matrix.to_excel(writer, sheet_name="Weighted_Decision_Matrix", index=False)
        ranking_df.to_excel(writer, sheet_name="TOPSIS_VIKOR_Ranking", index=False)
        rank_agreement_metrics.to_excel(writer, sheet_name="Rank_Agreement_Metrics", index=False)
        rank_agreement_details.to_excel(writer, sheet_name="Consensus_Ranking", index=False)
        sensitivity_df.to_excel(writer, sheet_name="Sensitivity_Scenarios", index=False)
        sensitivity_summary.to_excel(writer, sheet_name="Sensitivity_Summary", index=False)
        monte_carlo_summary.to_excel(writer, sheet_name="MonteCarlo_RankAccept", index=False)
        monte_carlo_rank_probability.to_excel(writer, sheet_name="MonteCarlo_RankProb", index=False)
        for sheet_name, table in computational_validation.items():
            if table is not None and not table.empty:
                table.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    stat_weights.to_excel(TABLE_DIR / "02_topic_derived_statistical_weights.xlsx", index=False)
    evidence.to_excel(TABLE_DIR / "03_articles_with_economic_feasibility_evidence.xlsx", index=False)
    tech_summary.to_excel(TABLE_DIR / "04_technology_economic_evidence_summary.xlsx", index=False)
    country_summary.to_excel(TABLE_DIR / "05_country_economic_evidence_summary.xlsx", index=False)
    source_summary_df.to_excel(TABLE_DIR / "06_source_database_economic_evidence_summary.xlsx", index=False)
    decision_matrix.to_excel(TABLE_DIR / "07_no_expert_technology_decision_matrix.xlsx", index=False)
    normalized_matrix.to_excel(TABLE_DIR / "08_no_expert_normalized_decision_matrix.xlsx", index=False)
    weighted_matrix.to_excel(TABLE_DIR / "09_no_expert_weighted_decision_matrix.xlsx", index=False)
    ranking_df.to_excel(TABLE_DIR / "10_no_expert_topsis_vikor_technology_ranking.xlsx", index=False)
    sensitivity_df.to_excel(TABLE_DIR / "11_no_expert_sensitivity_scenarios.xlsx", index=False)
    sensitivity_summary.to_excel(TABLE_DIR / "12_no_expert_sensitivity_summary.xlsx", index=False)
    criterion_metadata.to_excel(TABLE_DIR / "13_no_expert_criterion_metadata.xlsx", index=False)
    scoring_rules.to_excel(TABLE_DIR / "14_no_expert_evidence_scoring_rules.xlsx", index=False)
    technology_taxonomy.to_excel(TABLE_DIR / "15_technology_classification_rules.xlsx", index=False)
    matrix_trace.to_excel(TABLE_DIR / "16_no_expert_matrix_traceability_summary.xlsx", index=False)
    article_trace.to_excel(TABLE_DIR / "17_no_expert_article_criterion_traceability.xlsx", index=False)
    weight_comparison.to_excel(TABLE_DIR / "18_alternative_weight_scenarios.xlsx", index=False)
    rank_agreement_metrics.to_excel(TABLE_DIR / "19_topsis_vikor_rank_agreement_metrics.xlsx", index=False)
    rank_agreement_details.to_excel(TABLE_DIR / "20_topsis_vikor_consensus_ranking.xlsx", index=False)
    monte_carlo_summary.to_excel(TABLE_DIR / "21_monte_carlo_rank_acceptability_summary.xlsx", index=False)
    monte_carlo_rank_probability.to_excel(TABLE_DIR / "22_monte_carlo_rank_probability_matrix.xlsx", index=False)
    monte_carlo_samples.to_excel(TABLE_DIR / "23_monte_carlo_rank_samples.xlsx", index=False)
    for idx, (name, table) in enumerate(computational_validation.items(), start=24):
        if table is not None and not table.empty:
            safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name).lower()
            table.to_excel(TABLE_DIR / f"{idx:02d}_{safe_name}.xlsx", index=False)

    plot_weights(stat_weights)
    plot_evidence(evidence, tech_summary)
    plot_no_expert_rankings(decision_matrix, ranking_df, sensitivity_summary)
    plot_validation_outputs(
        weight_comparison,
        sensitivity_df,
        rank_agreement_details,
        monte_carlo_summary,
        monte_carlo_rank_probability,
    )
    plot_computational_validation_outputs(computational_validation)
    write_methodological_note(method, topic, stat_weights, ranking_df)

    print("=" * 72)
    print("MULTICRITERIA STATISTICAL-WEIGHTING SUPPORT")
    print("=" * 72)
    print(f"Winning method: {method}")
    print(f"Most aligned topic: Topic {topic}")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Articles with economic evidence signals: {len(evidence)}")
    print(f"Ranked technologies: {len(ranking_df)}")
    print(f"Computational validation tables: {sum(1 for table in computational_validation.values() if table is not None and not table.empty)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
