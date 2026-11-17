#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LDA + BATH-SEP-BNMF/AviTech-Bayes bibliometric analysis.

This script reuses the corrected LDA topic-modeling functions inherited from
an earlier project and compares them with AviTech-Bayes, the new Bayesian NMF
implementation. Both methods run over the same deduplicated corpus focused on
poultry, technology and economic feasibility. The older Hoshin Kanri strategic
analysis is not part of this workflow.

METHOD NOTES FOR THE ARTICLE AND FOR FUTURE CODE MAINTENANCE
============================================================

1. LDA - Latent Dirichlet Allocation
------------------------------------
LDA is a fully probabilistic topic model. It assumes that each document is a
mixture of latent topics and that each topic is a probability distribution over
words. In simplified form:

    theta_d ~ Dirichlet(alpha)        # document-topic probabilities
    phi_k   ~ Dirichlet(beta)         # topic-word probabilities
    z_dn    ~ Categorical(theta_d)    # topic assigned to each word
    w_dn    ~ Categorical(phi_zdn)    # observed word

In this script, the main LDA outputs are:
    - doc_topic: probability of each topic in each document.
    - topic_probability: highest topic probability assigned to a document.
    - topic_words: most probable / most weighted words for each topic.

Therefore, LDA is Bayesian/probabilistic by construction because it is based on
Dirichlet prior distributions for document-topic and topic-word distributions.

2. BNMF / AviTech-Bayes - Bayesian Non-negative Matrix Factorization
--------------------------------------------------------------------
The new method is named "AviTech-Bayes" for communication in the article. It is
implemented as a Bayesian-oriented Non-negative Matrix Factorization method for
the article theme: poultry, precision livestock farming / digital
transformation, and economic feasibility.

Classical NMF factorizes the document-term matrix X as:

    X ~= W x H

where:
    - X = document-term matrix.
    - W = document-topic matrix.
    - H = topic-word matrix.

In ordinary NMF, W and H are deterministic non-negative weights. In
AviTech-Bayes, these weights are normalized and interpreted as topic
probabilities/relative intensities, and a Bayesian layer is added on top of
them. The Bayesian layer uses Gamma sampling around W and H to estimate
posterior uncertainty:

    W* ~ Gamma(W * concentration, rate)
    H* ~ Gamma(H * concentration, rate)

Then W* and H* are normalized again to behave like probability distributions.
This produces:
    - posterior mean of topic activity.
    - posterior standard deviation.
    - credibility interval / uncertainty interval.
    - probability that a topic is inactive under an ARD rule.

The ARD component means Automatic Relevance Determination. In practical terms,
topics with very low posterior activity can be considered weak, redundant, or
inactive candidates. This is why the K selection graph for AviTech-Bayes uses a
composite criterion that includes:
    - reconstruction quality,
    - topic balance / entropy,
    - word exclusivity,
    - Bayesian-ARD topic activity.

So AviTech-Bayes is called Bayesian because it does not only return fixed NMF
weights. It estimates posterior uncertainty around the document-topic and
topic-word structures and uses Bayesian-ARD evidence to evaluate topic
relevance.

3. Comparison criterion
-----------------------
The comparison between LDA and AviTech-Bayes uses a composite score, combining:
    - alignment with the research objective,
    - topic coherence,
    - word exclusivity,
    - topic balance,
    - assignment confidence.

The best method is then copied into a named folder inside:
    4_best_method_results

That folder contains the figures, spreadsheets, and a topic-labeling workbook
needed to manually name / interpret the discovered topics.
"""

import argparse
import importlib.util
import itertools
import json
import re
import shutil
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.preprocessing import normalize

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
LDA_SCRIPT = SCRIPT_DIR / "lda_bibliometric_module.py"
BNMF_SCRIPT = SCRIPT_DIR / "avitech_bayes_module.py"

HOSHIN_TERMS = {
    "hoshin",
    "kanri",
    "hoshin kanri",
    "hoshin-kanri",
    "policy deployment",
}
HOSHIN_PATTERN = re.compile(
    r"\bhoshin\b|\bkanri\b|\bhoshin\s*[- ]?\s*kanri\b|\bpolicy\s+deployment\b",
    flags=re.IGNORECASE,
)

NEW_METHOD_NAME = "AviTech-Bayes"
NEW_METHOD_FULL_NAME = "AviTech-Bayes Topic Modeling"

THEME_COLORS = {
    "poultry": "#54A24B",
    "technology": "#4C78A8",
    "economic": "#F58518",
    "other": "#9E9E9E",
}

OBJECTIVE_AXIS_LABELS = {
    "poultry": "Poultry",
    "technology": "Technology",
    "economic": "Economic feasibility",
}

ALLUVIAL_AXIS_LABELS = {
    **OBJECTIVE_AXIS_LABELS,
    "other": "Other",
}

OBJECTIVE_GROUPS = {
    "poultry": {
        "poultry", "poultry_farming", "broiler", "broilers", "broiler_production",
        "layer", "layers", "layer_production", "hen", "hens", "chicken", "chickens",
        "bird", "birds", "flock", "flocks", "egg", "eggs", "behavior", "behaviour",
        "welfare", "mortality", "feed", "feeding", "housing", "environment", "weight",
    },
    "technology": {
        "precision_livestock_farming", "plf", "digital_transformation", "digital",
        "technology", "sensor", "sensors", "monitoring", "automation", "iot",
        "internet_of_things", "machine_learning", "deep_learning", "artificial_intelligence",
        "ai", "neural_networks", "artificial_neural_networks",
        "convolutional_neural_networks", "computer_vision", "vision", "camera",
        "cameras", "thermal", "smart_technology",
        "smart_manufacturing", "smart_farming", "smart_agriculture",
        "blockchain",
    },
    "economic": {
        "feasibility", "economic_feasibility", "economic", "economics", "cost", "costs",
        "roi", "return_on_investment", "profitability", "investment", "viability",
    },
}

SEARCH_OBJECTIVE_KEEP_TERMS = {
    "accuracy",
    "adoption",
    "ai",
    "algorithm",
    "algorithms",
    "animal",
    "animals",
    "artificial",
    "automation",
    "barn",
    "behavior",
    "behaviour",
    "biosecurity",
    "bird",
    "birds",
    "broiler",
    "broilers",
    "camera",
    "cameras",
    "chicken",
    "chickens",
    "computer",
    "cost",
    "costs",
    "deep",
    "detection",
    "digital",
    "digital_transformation",
    "economic",
    "economic_feasibility",
    "economics",
    "egg",
    "eggs",
    "environment",
    "farm",
    "farmer",
    "farmers",
    "farming",
    "farms",
    "feasibility",
    "feed",
    "feeding",
    "flock",
    "flocks",
    "hen",
    "hens",
    "housing",
    "image",
    "images",
    "imaging",
    "incubation",
    "intelligence",
    "investment",
    "iot",
    "layer",
    "layers",
    "learning",
    "livestock",
    "machine",
    "machine_learning",
    "neural_networks",
    "artificial_neural_networks",
    "convolutional_neural_networks",
    "monitoring",
    "mortality",
    "plf",
    "poultry",
    "poultry_farming",
    "precision",
    "precision_livestock_farming",
    "production",
    "profitability",
    "return_on_investment",
    "roi",
    "sensor",
    "sensors",
    "smart_technology",
    "smart_manufacturing",
    "smart_farming",
    "smart_agriculture",
    "technology",
    "thermal",
    "viability",
    "vision",
    "welfare",
    "weight",
}

SEARCH_OBJECTIVE_PHRASES = {
    r"\bprecision\s+livestock\s+farming\b": "precision_livestock_farming",
    r"\bdigital\s+transformation\b": "digital_transformation",
    r"\beconomic\s+feasibility\b": "economic_feasibility",
    r"\breturn\s+on\s+investment\b": "return_on_investment",
    r"\bmachine\s+learning\b": "machine_learning",
    r"\bdeep\s+learning\b": "deep_learning",
    r"\bartificial\s+neural\s+networks?\b": "artificial_neural_networks",
    r"\bconvolutional\s+neural\s+networks?\b": "convolutional_neural_networks",
    r"\bneural\s+networks?\b": "neural_networks",
    r"\bartificial\s+intelligence\b": "artificial_intelligence",
    r"\bcomputer\s+vision\b": "computer_vision",
    r"\binternet\s+of\s+things\b": "internet_of_things",
    r"\bdigital\s+twin\b": "digital_twin",
    r"\bsmart\s+manufacturing\b": "smart_manufacturing",
    r"\bsmart\s+farming\b": "smart_farming",
    r"\bsmart\s+agriculture\b": "smart_agriculture",
    r"\bsmart\s+technology\b": "smart_technology",
    r"\bsupply\s+chain\b": "supply_chain",
    r"\bvalue\s+chain\b": "value_chain",
    r"\bfood\s+chain\b": "food_chain",
    r"\bproduction\s+chain\b": "production_chain",
    r"\bpoultry\s+farming\b": "poultry_farming",
    r"\bbroiler\s+production\b": "broiler_production",
    r"\blayer\s+production\b": "layer_production",
}

GENERIC_EXCLUDE_TERMS = {
    "abstract",
    "article",
    "articles",
    "author",
    "available",
    "background",
    "case",
    "chapter",
    "chain",
    "conclusion",
    "conclusions",
    "conference",
    "data",
    "design",
    "discussion",
    "effect",
    "effects",
    "empirical",
    "evidence",
    "findings",
    "future",
    "implications",
    "introduction",
    "investigate",
    "investigates",
    "journal",
    "literature",
    "method",
    "methods",
    "methodology",
    "model",
    "models",
    "nan",
    "objective",
    "objectives",
    "paper",
    "papers",
    "practical",
    "present",
    "presents",
    "propose",
    "proposed",
    "purpose",
    "qualitative",
    "quantitative",
    "question",
    "results",
    "review",
    "sample",
    "significant",
    "study",
    "studies",
    "survey",
    "systematic",
    "table",
    "theoretical",
}

GENERIC_EXCLUDE_TERMS.update({
    "across",
    "address",
    "adoption",
    "approach",
    "based",
    "business",
    "capabilities",
    "case",
    "challenges",
    "critical",
    "decision",
    "development",
    "driven",
    "dynamic",
    "efficiency",
    "enhance",
    "enhancing",
    "factor",
    "factors",
    "framework",
    "frameworks",
    "governance",
    "high",
    "impact",
    "including",
    "information",
    "insights",
    "integration",
    "key",
    "knowledge",
    "level",
    "making",
    "multi",
    "network",
    "networks",
    "new",
    "operational",
    "perspective",
    "potential",
    "process",
    "provides",
    "resource",
    "role",
    "significantly",
    "social",
    "strategies",
    "support",
    "system",
    "systems",
    "technologies",
    "three",
    "time",
    "traditional",
    "two",
    "value",
})


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lda = load_module("lda_bibliometric_module", LDA_SCRIPT)
bnmf = load_module("bath_sep_bnmf", BNMF_SCRIPT)


def apply_objective_phrase_replacements(text: str) -> str:
    text = str(text or "").lower()
    for pattern, replacement in SEARCH_OBJECTIVE_PHRASES.items():
        text = re.sub(pattern, f" {replacement} ", text, flags=re.IGNORECASE)
    return text


def normalize_keyword_terms(keywords) -> list:
    """Normalize keyword phrases and keep the three research-objective axes clean."""
    normalized = []
    for keyword in keywords or []:
        text = apply_objective_phrase_replacements(keyword)
        text = re.sub(r"[^a-z0-9_\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if text in SEARCH_OBJECTIVE_KEEP_TERMS or "_" in text:
            normalized.append(text)
        else:
            normalized.extend(token for token in text.split() if token)

    term_set = set(normalized)
    if "digital" in term_set and "transformation" in term_set:
        normalized = [
            term for term in normalized
            if term not in {"digital", "transformation"}
        ]
        normalized.append("digital_transformation")
    term_set = set(normalized)
    phrase_pairs = [
        ({"machine", "learning"}, "machine_learning"),
        ({"deep", "learning"}, "deep_learning"),
        ({"neural", "network"}, "neural_networks"),
        ({"neural", "networks"}, "neural_networks"),
        ({"artificial", "neural", "network"}, "artificial_neural_networks"),
        ({"artificial", "neural", "networks"}, "artificial_neural_networks"),
        ({"convolutional", "neural", "network"}, "convolutional_neural_networks"),
        ({"convolutional", "neural", "networks"}, "convolutional_neural_networks"),
        ({"artificial", "intelligence"}, "artificial_intelligence"),
        ({"computer", "vision"}, "computer_vision"),
        ({"digital", "twin"}, "digital_twin"),
        ({"smart", "manufacturing"}, "smart_manufacturing"),
        ({"smart", "farming"}, "smart_farming"),
        ({"smart", "agriculture"}, "smart_agriculture"),
        ({"smart", "technology"}, "smart_technology"),
        ({"economic", "feasibility"}, "economic_feasibility"),
        ({"return", "investment"}, "return_on_investment"),
        ({"supply", "chain"}, "supply_chain"),
        ({"value", "chain"}, "value_chain"),
        ({"food", "chain"}, "food_chain"),
        ({"production", "chain"}, "production_chain"),
    ]
    for parts, phrase in phrase_pairs:
        if parts.issubset(term_set):
            normalized = [term for term in normalized if term not in parts]
            normalized.append(phrase)
            term_set = set(normalized)
    if {"precision", "livestock", "farming"}.issubset(term_set):
        normalized = [
            term for term in normalized
            if term not in {"precision", "livestock", "farming"}
        ]
        normalized.append("precision_livestock_farming")

    cleaned = []
    seen = set()
    for term in normalized:
        if term == "digital":
            term = "digital_transformation"
        if term == "smart":
            term = "smart_technology"
        if term == "neural":
            term = "neural_networks"
        if term == "behaviour":
            term = "behavior"
        if term == "transformation":
            continue
        if term.isdigit():
            continue
        if len(term) <= 2 and term not in {"ai", "iot"}:
            continue
        if term in {"and", "or", "of", "in", "on", "for", "with", "to", "the", "a", "an"}:
            continue
        if term in GENERIC_EXCLUDE_TERMS:
            continue
        if term not in seen:
            cleaned.append(term)
            seen.add(term)
    return cleaned


def objective_simple_tokenize(text: str) -> list:
    text = apply_objective_phrase_replacements(text)
    text = re.sub(r"\bindustry\s*4\.0\b", " industry4 ", text)
    text = re.sub(r"\bindustry\s*5\.0\b", " industry5 ", text)
    text = re.sub(r"\b4\.0\b", " industry4 ", text)
    text = re.sub(r"\b5\.0\b", " industry5 ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stop_set = set(lda.STOP_WORDS)
    keep_set = set(lda.KEEP_WORDS) | SEARCH_OBJECTIVE_KEEP_TERMS
    filtered = []

    for token in text.split():
        if token in keep_set:
            filtered.append(token)
        elif len(token) <= 2:
            continue
        elif token in stop_set:
            continue
        elif token.isdigit():
            continue
        else:
            filtered.append(token)

    return filtered


def normalized_bibliographic_text_for_dedup(text: str) -> str:
    """Normalize title/abstract text for duplicate detection in this script."""
    text = lda.normalize_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def duplicate_title_abstract_key(row: pd.Series) -> str:
    """Deduplicate by normalized title + normalized abstract, with title fallback."""
    title_norm = normalized_bibliographic_text_for_dedup(row.get("TITLE", ""))
    abstract_norm = normalized_bibliographic_text_for_dedup(row.get("ABSTRACT", ""))
    if abstract_norm:
        return f"{title_norm} || {abstract_norm}"
    return title_norm


def analyze_duplicates_title_abstract(df_list: list, file_names: list):
    """Replacement for the inherited title-only deduplication.

    The old LDA script removes duplicates by normalized title. For this unified
    script, duplicate records are removed only when both title and abstract
    match after normalization. If the abstract is missing, the normalized title
    is used as a fallback.
    """
    print("\n" + "=" * 70)
    print("ARTICLE ANALYSIS BY SOURCE AND DUPLICATES")
    print("=" * 70)
    print("Deduplication rule: normalized TITLE + normalized ABSTRACT")

    summary_data = []
    for df, name in zip(df_list, file_names):
        summary_data.append({"Source": name, "Loaded Articles": len(df)})

    df_summary = pd.DataFrame(summary_data)
    total_before = int(df_summary["Loaded Articles"].sum())

    print("\nARTICLES BY SOURCE (BEFORE DEDUPLICATION):")
    print("-" * 60)
    for _, row in df_summary.iterrows():
        print(f"   {row['Source']:<30} {row['Loaded Articles']:>6} articles")
    print("-" * 60)
    print(f"   {'GRAND TOTAL':<30} {total_before:>6} articles")

    merged = pd.concat(df_list, ignore_index=True)
    if "TITLE" not in merged.columns:
        return merged, total_before, len(merged), 0, 0, []
    if "ABSTRACT" not in merged.columns:
        merged["ABSTRACT"] = ""

    merged["_DUP_KEY"] = merged.apply(duplicate_title_abstract_key, axis=1)
    key_counts = merged["_DUP_KEY"].value_counts()
    duplicated_keys = key_counts[key_counts > 1]
    duplicate_groups = int(len(duplicated_keys))
    removed_records = int(duplicated_keys.sum() - duplicate_groups)

    print("\nDUPLICATES FOUND:")
    print("-" * 60)
    print(f"   Duplicate title + abstract keys: {duplicate_groups}")
    print(f"   Duplicate records removed: {removed_records}")

    unique = merged.drop_duplicates(subset=["_DUP_KEY"], keep="first").copy()
    unique_keys_global = set(unique["_DUP_KEY"].dropna())
    dup_stats_by_base = []

    print("\nDUPLICATE ANALYSIS BY SOURCE:")
    print("-" * 60)
    for i, (df, name) in enumerate(zip(df_list, file_names)):
        original_count = len(df)
        df_work = df.copy()
        if "ABSTRACT" not in df_work.columns:
            df_work["ABSTRACT"] = ""
        df_work["_DUP_KEY"] = df_work.apply(duplicate_title_abstract_key, axis=1)
        keys_in_base = set(df_work["_DUP_KEY"].dropna())
        remaining = len(keys_in_base.intersection(unique_keys_global))

        other_keys = set()
        for j, df2 in enumerate(df_list):
            if i == j:
                continue
            df2_work = df2.copy()
            if "ABSTRACT" not in df2_work.columns:
                df2_work["ABSTRACT"] = ""
            df2_work["_DUP_KEY"] = df2_work.apply(duplicate_title_abstract_key, axis=1)
            other_keys.update(set(df2_work["_DUP_KEY"].dropna()))

        shared = len(keys_in_base.intersection(other_keys))
        removed = max(original_count - remaining, 0)
        dup_stats_by_base.append({
            "Source": name,
            "Original_Articles": original_count,
            "Kept_Articles": remaining,
            "Removed_Duplicates": removed,
            "Shared_Titles": shared,
            "Shared_Title_Abstract_Keys": shared,
        })
        print(f"   {name}:")
        print(f"      Original: {original_count} | Kept: {remaining} | Removed: {removed} | Shared title+abstract: {shared}")

    print("-" * 60)
    print(f"   {'GRAND TOTAL':<30} {len(unique):>6} articles")
    print("=" * 70)

    if duplicate_groups > 0:
        duplicate_rows = merged[merged["_DUP_KEY"].isin(duplicated_keys.index)].copy()
        dup_report = (
            duplicate_rows
            .groupby("_DUP_KEY")
            .agg(
                Title=("TITLE", "first"),
                Abstract=("ABSTRACT", "first"),
                Occurrences=("_DUP_KEY", "size"),
            )
            .reset_index(drop=True)
            .sort_values("Occurrences", ascending=False)
        )
        dup_report["Abstract"] = dup_report["Abstract"].fillna("").astype(str).str.slice(0, 500)
        dup_report.to_excel(lda.config.results_dir / "00_duplicates_report.xlsx", index=False)
        print("Duplicate report saved as: 00_duplicates_report.xlsx")

    unique = unique.drop(columns=["_DUP_KEY"], errors="ignore")
    return unique.reset_index(drop=True), total_before, len(unique), duplicate_groups, removed_records, dup_stats_by_base


def configure_modules(input_dir: Path, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    lda.config.base_dir = input_dir
    lda.config.results_dir = output_dirs(results_dir)["bibliometric"]
    lda.config.results_dir.mkdir(parents=True, exist_ok=True)

    # Align the inherited LDA tokenizer with the poultry/technology/economic
    # feasibility objective. Hoshin Kanri is not used as an analysis axis here.
    keep_words = {w.lower() for w in lda.KEEP_WORDS}
    keep_words.update(SEARCH_OBJECTIVE_KEEP_TERMS)

    stop_words = set(lda.STOP_WORDS)
    stop_words.update(GENERIC_EXCLUDE_TERMS)
    stop_words = stop_words - keep_words

    lda.STOP_WORDS = sorted(stop_words)
    lda.KEEP_WORDS = sorted(keep_words)
    lda.simple_tokenize = objective_simple_tokenize
    lda.run_lda_simple = run_lda_simple_quiet
    lda.analyze_duplicates = analyze_duplicates_title_abstract

    bnmf.Config.OUTPUT_DIR = output_dirs(results_dir)["bnmf"]
    bnmf.Config.WORDCLOUD_DIR = bnmf.Config.OUTPUT_DIR / "wordclouds"
    bnmf.Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bnmf.Config.WORDCLOUD_DIR.mkdir(parents=True, exist_ok=True)


def run_lda_simple_quiet(dtm, n_topics: int):
    lda_model = lda.LatentDirichletAllocation(
        n_components=n_topics,
        random_state=lda.config.seed,
        max_iter=min(int(getattr(lda.config, "lda_iterations", 160)), 160),
        verbose=0,
        n_jobs=-1,
        learning_method="batch",
    )
    doc_topic = lda_model.fit_transform(dtm)
    return lda_model, doc_topic


def select_optimal_k_fast(dtm, feature_names, max_topics: int):
    start_k = 3 if max_topics >= 3 else 2
    max_topics = min(max_topics, 6)
    rows = []
    print("\n=== FAST LDA TOPIC EVALUATION ===")
    print(f"Testing K from {start_k} to {max_topics} with capped iterations...")
    for k in range(start_k, max_topics + 1):
        try:
            model = lda.LatentDirichletAllocation(
                n_components=k,
                random_state=lda.config.seed,
                max_iter=80,
                verbose=0,
                n_jobs=-1,
                learning_method="batch",
            )
            model.fit(dtm)
            log_perp = lda.calculate_log_perplexity(model, dtm)
            coherence = lda.calculate_topic_coherence(model, feature_names, dtm, lda.config.n_words_per_topic)
            topic_words = lda.get_top_words(model, feature_names, lda.config.n_words_per_topic)
            exclusivity = topic_exclusivity([words for _, words in sorted(topic_words.items())])
            composite = -log_perp + (coherence * 10) + (exclusivity * 2)
            rows.append(
                {
                    "K": k,
                    "Log_Perplexity": log_perp,
                    "Topic_Coherence": coherence,
                    "Word_Exclusivity": exclusivity,
                    "Composite_Score": composite,
                }
            )
            print(f"  K={k}: LogPerp={log_perp:.4f}, Coherence={coherence:.4f}, Exclusivity={exclusivity:.4f}, Score={composite:.4f}")
        except Exception as exc:
            print(f"  K={k}: Error - {str(exc)[:90]}")

    if not rows:
        fallback_k = 3 if max_topics >= 3 else 2
        return fallback_k, pd.DataFrame(), f"Fast K selection failed; using K={fallback_k}."

    metrics = pd.DataFrame(rows)
    best_k = int(metrics.sort_values("Composite_Score", ascending=False).iloc[0]["K"])
    premise = (
        "Fast LDA K selection: K=1 avoided; K=2 avoided when corpus size allows; "
        "criterion combines log perplexity, coherence and word exclusivity."
    )
    print(f"Best LDA K selected: {best_k}")
    return best_k, metrics, premise


def output_dirs(results_dir: Path) -> dict:
    dirs = {
        "bibliometric": results_dir / "1_bibliometric_analysis",
        "lda": results_dir / "2_LDA_results",
        "bnmf": results_dir / "3_BNMF_results",
        "best": results_dir / "4_best_method_results",
        "common": results_dir / "5_LDA_BNMF_common_results",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def tokenize_for_vocabulary(text) -> list:
    if text is None or pd.isna(text):
        return []
    text = apply_objective_phrase_replacements(text)
    text = re.sub(r"\bindustry\s*4\.0\b", " industry4 ", text)
    text = re.sub(r"\bindustry\s*5\.0\b", " industry5 ", text)
    text = re.sub(r"\b4\.0\b", " industry4 ", text)
    text = re.sub(r"\b5\.0\b", " industry5 ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [token for token in text.split() if len(token) > 2 and not token.isdigit()]


def adapt_vocabulary_to_bibtex(df: pd.DataFrame, results_dir: Path):
    """Adapt keep/excluded words to the actual BibTeX corpus."""
    title_counter = Counter()
    keyword_counter = Counter()
    abstract_counter = Counter()
    abstract_doc_counter = Counter()

    for _, row in df.iterrows():
        title_counter.update(tokenize_for_vocabulary(row.get("TITLE", "")))

        keyword_text = row.get("KEYWORDS", "")
        if keyword_text is not None and not pd.isna(keyword_text):
            for keyword in lda.split_keywords(keyword_text):
                keyword_counter.update(tokenize_for_vocabulary(keyword))

        abstract_tokens = tokenize_for_vocabulary(row.get("ABSTRACT", ""))
        abstract_counter.update(abstract_tokens)
        abstract_doc_counter.update(set(abstract_tokens))

    base_stop = set(lda.STOP_WORDS)
    base_keep = {word.lower() for word in lda.KEEP_WORDS}
    base_keep.update(SEARCH_OBJECTIVE_KEEP_TERMS)

    corpus_size = max(len(df), 1)
    min_keep_score = max(3, int(corpus_size * 0.01))
    keep_scores = {}
    candidate_words = set(title_counter) | set(keyword_counter) | set(abstract_counter)

    for word in candidate_words:
        if word in GENERIC_EXCLUDE_TERMS:
            continue
        if word in base_stop and word not in base_keep:
            continue
        score = (4 * keyword_counter[word]) + (2 * title_counter[word]) + abstract_counter[word]
        if score >= min_keep_score and (keyword_counter[word] > 0 or title_counter[word] > 0):
            keep_scores[word] = score

    adaptive_keep = {
        word
        for word, _ in sorted(keep_scores.items(), key=lambda item: (-item[1], item[0]))[:350]
    }
    adaptive_keep.update(base_keep)

    very_common_cutoff = max(5, int(corpus_size * 0.30))
    adaptive_stop = set(GENERIC_EXCLUDE_TERMS)
    for word, doc_count in abstract_doc_counter.items():
        if word in adaptive_keep:
            continue
        if doc_count >= very_common_cutoff and keyword_counter[word] == 0 and title_counter[word] == 0:
            adaptive_stop.add(word)

    adaptive_stop = adaptive_stop - adaptive_keep

    lda.STOP_WORDS = sorted((base_stop | adaptive_stop | GENERIC_EXCLUDE_TERMS) - adaptive_keep)
    lda.KEEP_WORDS = sorted(adaptive_keep)

    dirs = output_dirs(results_dir)
    vocab_dir = dirs["common"] / "Vocabulary_Adaptation"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    keep_df = pd.DataFrame(
        [
            {
                "word": word,
                "keyword_freq": keyword_counter[word],
                "title_freq": title_counter[word],
                "abstract_freq": abstract_counter[word],
                "score": keep_scores.get(word, 0),
            }
            for word in sorted(adaptive_keep)
        ]
    ).sort_values(["score", "keyword_freq", "title_freq", "abstract_freq"], ascending=False)

    stop_df = pd.DataFrame(
        [
            {
                "word": word,
                "doc_freq": abstract_doc_counter[word],
                "keyword_freq": keyword_counter[word],
                "title_freq": title_counter[word],
                "reason": "generic_or_methodological" if word in GENERIC_EXCLUDE_TERMS else "too_common_in_abstracts",
            }
            for word in sorted(adaptive_stop)
        ]
    ).sort_values(["reason", "doc_freq", "word"], ascending=[True, False, True])

    keep_df.to_excel(vocab_dir / "adaptive_keep_words_from_bibtex.xlsx", index=False)
    stop_df.to_excel(vocab_dir / "adaptive_excluded_words_from_bibtex.xlsx", index=False)
    pd.DataFrame({"stop_word": lda.STOP_WORDS}).to_excel(vocab_dir / "final_stop_words_used.xlsx", index=False)
    pd.DataFrame({"keep_word": lda.KEEP_WORDS}).to_excel(vocab_dir / "final_keep_words_used.xlsx", index=False)
    pd.DataFrame(
        {
            "protected_search_objective_term": sorted(SEARCH_OBJECTIVE_KEEP_TERMS)
        }
    ).to_excel(vocab_dir / "protected_terms_from_search_string.xlsx", index=False)
    pd.DataFrame(
        [
            {"phrase_pattern": pattern, "token_used": token}
            for pattern, token in SEARCH_OBJECTIVE_PHRASES.items()
        ]
    ).to_excel(vocab_dir / "protected_phrases_as_tokens.xlsx", index=False)

    print("\n=== ADAPTIVE VOCABULARY FROM BIBTEX ===")
    print(f"Adaptive keep words: {len(adaptive_keep)}")
    print(f"Adaptive excluded words: {len(adaptive_stop)}")
    print(f"Vocabulary files saved in: {vocab_dir}")
    return keep_df, stop_df


def has_hoshin_kanri_signal(row: pd.Series) -> bool:
    fields = ["TITLE", "ABSTRACT", "KEYWORDS"]
    text = " ".join(str(row.get(field, "")) for field in fields)
    return bool(HOSHIN_PATTERN.search(text))


def source_group_from_filename(filename: str) -> str:
    name = str(filename).lower()
    if "sciencedirect" in name:
        return "ScienceDirect"
    if "scopus" in name:
        return "Scopus"
    if "webofscience" in name or "wos" in name:
        return "Web of Science"
    if "ieee" in name:
        return "IEEE Xplore"
    return Path(filename).stem


def create_source_intersection_upset(databases, file_names, biblio_dir: Path):
    """
    Creates an UpSetPlot showing overlap among bibliographic databases.

    This is more readable than Venn diagrams when there are more than three
    sources and is useful to document how Scopus, WoS, ScienceDirect, and IEEE
    overlap before deduplication.
    """
    rows = []
    title_to_sources = {}
    for df, filename in zip(databases, file_names):
        source = source_group_from_filename(filename)
        if "TITLE" not in df.columns:
            continue
        for title in df["TITLE"].fillna(""):
            normalized = lda.normalized_title(title)
            if not normalized:
                continue
            title_to_sources.setdefault(normalized, set()).add(source)

    for title_key, sources in title_to_sources.items():
        rows.append({
            "normalized_title": title_key,
            "source_combination": " + ".join(sorted(sources)),
            "n_sources": len(sources),
            **{source: source in sources for source in sorted({s for values in title_to_sources.values() for s in values})},
        })

    if not rows:
        return pd.DataFrame()

    upset_df = pd.DataFrame(rows)
    upset_df.to_excel(biblio_dir / "04_database_intersection_upsetplot.xlsx", index=False)

    summary = (
        upset_df.groupby(["source_combination", "n_sources"])
        .size()
        .reset_index(name="articles")
        .sort_values(["n_sources", "articles"], ascending=[False, False])
    )
    summary.to_excel(biblio_dir / "04_database_overlap_summary.xlsx", index=False)

    if not MATPLOTLIB_AVAILABLE:
        return upset_df

    img_dir = biblio_dir / "IMAGES_BIBLIOMETRICAS"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "04_database_intersection_upsetplot.png"

    memberships = [tuple(sorted(sources)) for sources in title_to_sources.values() if sources]
    try:
        from upsetplot import UpSet, from_memberships
        data = from_memberships(memberships)
        fig = plt.figure(figsize=(13, 7))
        upset = UpSet(
            data,
            subset_size="count",
            show_counts=True,
            sort_by="cardinality",
            sort_categories_by="cardinality",
            min_subset_size=1,
            facecolor="#2F5D8C",
            other_dots_color="#D0D0D0",
            shading_color="#F4F4F4",
        )
        upset.plot(fig=fig)
        fig.suptitle("Intersection of Bibliographic Sources Before Deduplication", fontweight="bold", y=1.02)
        plt.savefig(img_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        plot_custom_source_intersection(summary, img_path)
    return upset_df


def plot_custom_source_intersection(summary_df: pd.DataFrame, output_path: Path, top_n=18):
    if not MATPLOTLIB_AVAILABLE or summary_df.empty:
        return False
    plot_df = summary_df.sort_values("articles", ascending=False).head(top_n).copy()
    sources = sorted(set().union(*[set(str(combo).split(" + ")) for combo in plot_df["source_combination"]]))
    x = np.arange(len(plot_df))
    fig = plt.figure(figsize=(max(12, len(plot_df) * 0.55), 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.35], hspace=0.05)
    ax_bar = fig.add_subplot(gs[0])
    ax_mat = fig.add_subplot(gs[1], sharex=ax_bar)

    bars = ax_bar.bar(x, plot_df["articles"], color="#2F5D8C", edgecolor="white")
    ax_bar.set_title("Intersection of Bibliographic Sources Before Deduplication", fontweight="bold")
    ax_bar.set_ylabel("Articles")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.set_xticks([])
    for bar, value in zip(bars, plot_df["articles"]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(plot_df["articles"]) * 0.01, int(value), ha="center", va="bottom", fontsize=8)

    for row_idx, source in enumerate(sources):
        ax_mat.scatter(x, [row_idx] * len(x), s=34, color="#D8D8D8", zorder=1)
    for col_idx, combo in enumerate(plot_df["source_combination"]):
        active = set(str(combo).split(" + "))
        active_rows = [sources.index(source) for source in active if source in sources]
        if active_rows:
            ax_mat.scatter([col_idx] * len(active_rows), active_rows, s=78, color="#2F5D8C", zorder=3)
            if len(active_rows) > 1:
                ax_mat.plot([col_idx, col_idx], [min(active_rows), max(active_rows)], color="#2F5D8C", linewidth=1.6, zorder=2)

    ax_mat.set_yticks(range(len(sources)))
    ax_mat.set_yticklabels(sources)
    ax_mat.set_xlabel("Source combination, ordered by article count")
    ax_mat.set_ylim(-0.6, len(sources) - 0.4)
    ax_mat.invert_yaxis()
    ax_mat.grid(axis="x", alpha=0.08)
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)
        ax_mat.spines[spine].set_visible(False)
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def load_and_prepare_database(input_dir: Path, results_dir: Path):
    dirs = output_dirs(results_dir)
    biblio_dir = dirs["bibliometric"]
    bib_files = sorted(input_dir.glob("*.bib"))
    print(f"\n=== LOADING BIBTEX FILES ===")
    print(f"Input directory: {input_dir}")
    print(f"Found {len(bib_files)} .bib files")
    if not bib_files:
        raise FileNotFoundError(f"No .bib files found in {input_dir}")

    databases = []
    file_names = []
    for bib_file in bib_files:
        print(f"  Loading: {bib_file.name}")
        df = lda.load_bib_file_complete(str(bib_file))
        if not df.empty:
            databases.append(df)
            file_names.append(bib_file.name)
            print(f"    Records: {len(df)}")

    if not databases:
        raise RuntimeError("No valid records loaded from BibTeX files.")

    create_source_intersection_upset(databases, file_names, biblio_dir)

    merged, total_before, total_after, duplicates, removed, dup_stats = lda.analyze_duplicates(databases, file_names)

    df_summary = pd.DataFrame(
        [{"Source": name, "Loaded Articles": len(df)} for name, df in zip(file_names, databases)]
    )
    lda.create_word_report(df_summary, total_before, total_after, duplicates, removed, dup_stats, file_names)

    legacy_hoshin_signal = int(merged.apply(has_hoshin_kanri_signal, axis=1).sum())
    merged_filtered = merged.reset_index(drop=True)

    print(f"\n=== CORPUS AFTER DEDUPLICATION ===")
    print("Hoshin Kanri is not used as a bibliometric or topic-modeling axis in this script.")
    print(f"Records with inherited Hoshin/Kanri signal detected, but not removed: {legacy_hoshin_signal}")
    print(f"Records retained for LDA + BNMF/AviTech-Bayes: {len(merged_filtered)}")

    merged_filtered.to_excel(biblio_dir / "00_base_complete_deduplicated_for_topic_models.xlsx", index=False)
    merged_filtered.to_csv(biblio_dir / "00_base_complete_deduplicated_for_topic_models.csv", index=False, encoding="utf-8-sig")

    summary_stats = pd.DataFrame(
        {
            "Metric": [
                "Total before deduplication",
                "Total after deduplication",
                "Duplicate title + abstract keys",
                "Removed duplicate records",
                "Records with inherited Hoshin/Kanri signal detected, not removed",
                "Final records for analysis",
                "First publication year in deduplicated corpus",
                "Last publication year in deduplicated corpus",
                "Bibliometric time-window note",
            ],
            "Value": [
                total_before,
                total_after,
                duplicates,
                removed,
                legacy_hoshin_signal,
                len(merged_filtered),
                pd.to_numeric(merged_filtered.get("YEAR", pd.Series(dtype=float)), errors="coerce").min(),
                pd.to_numeric(merged_filtered.get("YEAR", pd.Series(dtype=float)), errors="coerce").max(),
                "Interpret bibliometric trends cautiously because the corpus covers a short publication window, expected around 2025-2026.",
            ],
        }
    )
    summary_stats.to_excel(biblio_dir / "00_deduplication_summary_for_topic_models.xlsx", index=False)

    return merged_filtered


def create_bibliometric_tables_and_images(df: pd.DataFrame, results_dir: Path):
    dirs = output_dirs(results_dir)
    biblio_dir = dirs["bibliometric"]
    img_dir = biblio_dir / "IMAGES_BIBLIOMETRICAS"
    img_dir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    work["year_numeric"] = pd.to_numeric(work.get("YEAR", pd.Series(dtype=float)), errors="coerce")
    work["source_clean"] = work.get("JOURNAL", pd.Series([""] * len(work))).fillna("").map(lambda x: str(x).strip())
    work["source_file"] = work.get("SOURCE_FILE", work.get("source_file", pd.Series(["Unknown"] * len(work)))).fillna("Unknown")

    publications_by_year = (
        work.dropna(subset=["year_numeric"])
        .assign(Year=lambda x: x["year_numeric"].astype(int))
        .groupby("Year")
        .size()
        .reset_index(name="Documents")
        .sort_values("Year")
    )
    top_sources = (
        work["source_clean"].replace("", np.nan).dropna().value_counts().head(20)
        .rename_axis("Source").reset_index(name="Documents")
    )
    records_by_file = (
        work["source_file"].value_counts().rename_axis("BibTeX_File").reset_index(name="Records").head(40)
    )

    author_counter = Counter()
    if "AUTHOR" in work.columns:
        for author_text in work["AUTHOR"].fillna(""):
            author_counter.update(lda.split_author_names(author_text))
    top_authors = pd.DataFrame(author_counter.most_common(30), columns=["Author", "Documents"])

    keyword_counter = Counter()
    if "KEYWORDS" in work.columns:
        for keyword_text in work["KEYWORDS"].fillna(""):
            keyword_counter.update(normalize_keyword_terms(lda.split_keywords(keyword_text)))
    top_keywords = pd.DataFrame(keyword_counter.most_common(40), columns=["Keyword", "Occurrences"])

    objective_term_counts = []
    text_columns = ["TITLE", "ABSTRACT", "KEYWORDS"]
    all_text = " ".join(
        " ".join(work[col].fillna("").astype(str).tolist())
        for col in text_columns if col in work.columns
    ).lower()
    all_text = apply_objective_phrase_replacements(all_text)
    tokens = Counter(tokenize_for_vocabulary(all_text))
    for group, terms in OBJECTIVE_GROUPS.items():
        for term in sorted(terms):
            objective_term_counts.append({
                "Group": group,
                "Term": term,
                "Occurrences": tokens[term],
            })
    objective_terms = pd.DataFrame(objective_term_counts).sort_values(["Group", "Occurrences"], ascending=[True, False])

    with pd.ExcelWriter(biblio_dir / "01_bibliometric_tables_2025_2026.xlsx") as writer:
        publications_by_year.to_excel(writer, sheet_name="Publications_by_Year", index=False)
        top_sources.to_excel(writer, sheet_name="Top_Sources", index=False)
        records_by_file.to_excel(writer, sheet_name="Records_by_BibTeX_File", index=False)
        top_authors.to_excel(writer, sheet_name="Top_Authors", index=False)
        top_keywords.to_excel(writer, sheet_name="Top_Keywords", index=False)
        objective_terms.to_excel(writer, sheet_name="Search_Objective_Terms", index=False)

    if not MATPLOTLIB_AVAILABLE:
        return

    if not publications_by_year.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(publications_by_year["Year"].astype(str), publications_by_year["Documents"], color="#4C78A8", edgecolor="black")
        ax.set_title("Publications by Year (Short Window)", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Documents")
        for bar, value in zip(bars, publications_by_year["Documents"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value), ha="center")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(img_dir / "01_publications_by_year_2025_2026.png", dpi=180, bbox_inches="tight")
        plt.close()

    if not top_sources.empty:
        plot_df = top_sources.head(15).sort_values("Documents", ascending=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(plot_df["Source"], plot_df["Documents"], color="#59A14F", edgecolor="black")
        ax.set_title("Top Journals / Sources", fontweight="bold")
        ax.set_xlabel("Documents")
        plt.tight_layout()
        plt.savefig(img_dir / "02_top_journals_sources.png", dpi=180, bbox_inches="tight")
        plt.close()

    if not top_keywords.empty:
        plot_df = top_keywords.head(20).sort_values("Occurrences", ascending=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(plot_df["Keyword"], plot_df["Occurrences"], color="#F28E2B", edgecolor="black")
        ax.set_title("Top Author Keywords", fontweight="bold")
        ax.set_xlabel("Occurrences")
        plt.tight_layout()
        plt.savefig(img_dir / "03_top_author_keywords.png", dpi=180, bbox_inches="tight")
        plt.close()

    objective_plot = objective_terms[objective_terms["Occurrences"] > 0].groupby("Group")["Occurrences"].sum().reset_index()
    if not objective_plot.empty:
        objective_plot["Axis"] = objective_plot["Group"].map({
            "poultry": "Poultry",
            "technology": "Technology",
            "economic": "Economic feasibility",
            "other": "Other",
        }).fillna(objective_plot["Group"].astype(str).str.title())
        fig, ax = plt.subplots(figsize=(8, 5))
        axis_colors = [
            THEME_COLORS.get(str(group), THEME_COLORS["other"])
            for group in objective_plot["Group"]
        ]
        ax.bar(objective_plot["Axis"], objective_plot["Occurrences"], color=axis_colors, edgecolor="black")
        ax.set_title("Search Objective Coverage in Corpus", fontweight="bold")
        ax.set_xlabel("Objective Axis")
        ax.set_ylabel("Term Occurrences")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(img_dir / "04_search_objective_axis_coverage.png", dpi=180, bbox_inches="tight")
        plt.close()


COUNTRY_ALIASES = {
    "usa": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "china": "China",
    "people's republic of china": "China",
    "pr china": "China",
    "brazil": "Brazil",
    "brasil": "Brazil",
    "spain": "Spain",
    "italy": "Italy",
    "france": "France",
    "germany": "Germany",
    "netherlands": "Netherlands",
    "india": "India",
    "canada": "Canada",
    "australia": "Australia",
    "portugal": "Portugal",
    "turkey": "Turkey",
    "japan": "Japan",
    "south korea": "South Korea",
    "korea": "South Korea",
    "iran": "Iran",
    "mexico": "Mexico",
    "poland": "Poland",
    "denmark": "Denmark",
    "sweden": "Sweden",
    "norway": "Norway",
    "finland": "Finland",
    "belgium": "Belgium",
    "switzerland": "Switzerland",
    "austria": "Austria",
    "ireland": "Ireland",
    "greece": "Greece",
    "egypt": "Egypt",
    "south africa": "South Africa",
    "pakistan": "Pakistan",
    "bangladesh": "Bangladesh",
    "indonesia": "Indonesia",
    "malaysia": "Malaysia",
    "thailand": "Thailand",
    "vietnam": "Vietnam",
    "saudi arabia": "Saudi Arabia",
}


def extract_record_text(row: pd.Series, columns: list) -> str:
    return " ".join(str(row.get(col, "")) for col in columns if col in row.index and pd.notna(row.get(col, "")))


def extract_countries(text: str) -> list:
    text_l = str(text or "").lower()
    found = []
    for alias, canonical in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text_l):
            found.append(canonical)
    return sorted(set(found))


def extract_institutions(text: str) -> list:
    text = str(text or "")
    if not text.strip():
        return []
    candidates = re.split(r";|\n|\band\b", text)
    institutions = []
    pattern = re.compile(
        r"([A-Z][A-Za-z&.\- ]{2,}\b(?:University|Institute|College|Laboratory|Lab|Center|Centre|School|Academy|Corporation|Company|Department|Faculty|Hospital|Research)[A-Za-z&.\- ]*)"
    )
    for candidate in candidates:
        for match in pattern.findall(candidate):
            clean = re.sub(r"\s+", " ", match).strip(" ,.;")
            if 4 <= len(clean) <= 120:
                institutions.append(clean)
    return sorted(set(institutions))


def record_affiliation_text(row: pd.Series) -> str:
    possible = [
        "AFFILIATION", "AFFILIATIONS", "ADDRESS", "ADDRESSES", "C1", "AD",
        "INSTITUTION", "ORGANIZATION", "ORGANISATION", "SCHOOL",
    ]
    return extract_record_text(row, possible)


def build_pair_counts(items_by_record, max_items_per_record=8):
    counter = Counter()
    for items in items_by_record:
        clean = sorted(set([str(item).strip() for item in items if str(item).strip()]))
        clean = clean[:max_items_per_record]
        for left, right in itertools.combinations(clean, 2):
            counter[(left, right)] += 1
    return pd.DataFrame(
        [{"source": k[0], "target": k[1], "weight": v} for k, v in counter.items()]
    ).sort_values("weight", ascending=False) if counter else pd.DataFrame(columns=["source", "target", "weight"])


def plot_interactive_network(graph, output_path: Path, title: str):
    html_path = output_path.with_suffix(".html")
    try:
        Network = importlib.import_module("pyvis.network").Network
        net = Network(height="820px", width="100%", bgcolor="#ffffff", font_color="#222222")
        net.force_atlas_2based(gravity=-45, central_gravity=0.01, spring_length=160, spring_strength=0.08)
        for node in graph.nodes():
            degree = graph.degree(node, weight="weight")
            net.add_node(
                node,
                label=str(node),
                title=f"{node}<br>Weighted degree: {degree:.2f}",
                value=max(1, degree),
            )
        for source, target, data in graph.edges(data=True):
            weight = float(data.get("weight", 1))
            net.add_edge(source, target, value=weight, title=f"Weight: {weight:.0f}")
        net.write_html(str(html_path), notebook=False)
        return True
    except Exception:
        pass

    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        return False

    pos = nx.spring_layout(graph, k=0.95, iterations=150, seed=42, weight="weight")
    edge_x, edge_y = [], []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, labels, hover, sizes = [], [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        degree = graph.degree(node, weight="weight")
        node_x.append(x)
        node_y.append(y)
        labels.append(str(node))
        hover.append(f"{node}<br>Weighted degree: {degree:.2f}")
        sizes.append(10 + min(40, degree * 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.7, color="#999999"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center",
        hovertext=hover, hoverinfo="text", marker=dict(size=sizes, color="#4C78A8", line=dict(width=1, color="white"))
    ))
    fig.update_layout(title=title, showlegend=False, margin=dict(l=10, r=10, t=50, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.write_html(str(html_path))
    return True


def plot_network(edge_df: pd.DataFrame, output_path: Path, title: str, max_edges=60, max_labels=22, min_weight=None, thematic=False):
    if not MATPLOTLIB_AVAILABLE or edge_df.empty:
        return
    try:
        import networkx as nx
    except ImportError:
        return

    plot_edges = filter_weighted_edges(edge_df, max_edges=max_edges, min_weight=min_weight)
    graph = nx.Graph()
    for _, row in plot_edges.iterrows():
        graph.add_edge(row["source"], row["target"], weight=float(row["weight"]))

    if graph.number_of_nodes() == 0:
        return

    degrees = dict(graph.degree(weight="weight"))
    communities = detect_graph_communities(graph)
    community_ids = {name: idx for idx, name in enumerate(sorted(set(communities.values())))}
    theme_palette = {
        "poultry": "#54A24B",
        "technology": "#4C78A8",
        "economic": "#F58518",
        "other": "#B279A2",
    }
    cluster_palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]
    pos = nx.spring_layout(graph, k=1.35, iterations=220, seed=42, weight="weight")
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    max_weight = max(weights) if weights else 1

    fig, ax = plt.subplots(figsize=(18, 13))
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=[0.8 + 4.0 * (w / max_weight) for w in weights],
        alpha=0.35,
        edge_color="#666666",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=[120 + 55 * degrees.get(node, 1) for node in graph.nodes()],
        node_color=[
            theme_palette.get(node_theme(node), "#B279A2")
            if thematic else cluster_palette[community_ids.get(communities.get(node, "cluster_1"), 0) % len(cluster_palette)]
            for node in graph.nodes()
        ],
        alpha=0.88,
        linewidths=0.8,
        edgecolors="white",
    )
    ranked_nodes = sorted(graph.nodes(), key=lambda node: degrees.get(node, 0), reverse=True)
    label_nodes = set(ranked_nodes[:max_labels])
    labels = {node: str(node) for node in label_nodes}
    text_items = nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_color="#111111",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, boxstyle="round,pad=0.18"),
    )
    try:
        from adjustText import adjust_text
        adjust_text(
            list(text_items.values()),
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4, alpha=0.55),
            expand_points=(1.2, 1.4),
            expand_text=(1.2, 1.4),
        )
    except Exception:
        pass
    ax.set_title(title, fontweight="bold")
    subtitle = f"Showing strongest {len(plot_edges)} edges | min weight: {plot_edges['weight'].min() if not plot_edges.empty else 0:g}"
    ax.text(0.01, 0.01, subtitle, transform=ax.transAxes, fontsize=9, color="#444444")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    plot_interactive_network(graph, output_path, title)


def export_network_cluster_tables(edge_df: pd.DataFrame, output_excel: Path, max_edges=80, min_weight=None, thematic=False):
    if edge_df.empty:
        pd.DataFrame().to_excel(output_excel, index=False)
        return
    try:
        import networkx as nx
    except ImportError:
        edge_df.to_excel(output_excel, index=False)
        return

    filtered = filter_weighted_edges(edge_df, max_edges=max_edges, min_weight=min_weight)
    graph = nx.Graph()
    for _, row in filtered.iterrows():
        graph.add_edge(row["source"], row["target"], weight=float(row["weight"]))
    communities = detect_graph_communities(graph)
    degree = dict(graph.degree(weight="weight"))
    node_rows = []
    for node in graph.nodes():
        node_rows.append({
            "node": node,
            "weighted_degree": degree.get(node, 0),
            "cluster": communities.get(node, ""),
            "theme": node_theme(node) if thematic else "",
        })
    node_df = pd.DataFrame(node_rows).sort_values("weighted_degree", ascending=False)
    edge_out = filtered.copy()
    edge_out["source_cluster"] = edge_out["source"].map(communities)
    edge_out["target_cluster"] = edge_out["target"].map(communities)
    if thematic:
        edge_out["source_theme"] = edge_out["source"].map(node_theme)
        edge_out["target_theme"] = edge_out["target"].map(node_theme)

    with pd.ExcelWriter(output_excel) as writer:
        edge_df.to_excel(writer, sheet_name="All_Edges", index=False)
        edge_out.to_excel(writer, sheet_name="Filtered_Strong_Edges", index=False)
        node_df.to_excel(writer, sheet_name="Nodes_Clusters", index=False)


def create_adjacency_heatmap(edge_df: pd.DataFrame, output_path: Path, title: str, top_n=25, min_weight=None):
    if not MATPLOTLIB_AVAILABLE or edge_df.empty:
        return
    filtered = filter_weighted_edges(edge_df, max_edges=top_n * 4, min_weight=min_weight)
    if filtered.empty:
        return
    nodes = Counter()
    for _, row in filtered.iterrows():
        nodes[row["source"]] += float(row["weight"])
        nodes[row["target"]] += float(row["weight"])
    top_nodes = [node for node, _ in nodes.most_common(top_n)]
    matrix = pd.DataFrame(0.0, index=top_nodes, columns=top_nodes)
    for _, row in filtered.iterrows():
        source, target, weight = row["source"], row["target"], float(row["weight"])
        if source in matrix.index and target in matrix.columns:
            matrix.loc[source, target] += weight
            matrix.loc[target, source] += weight
    if matrix.values.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix.values, cmap="YlGnBu")
    ax.set_xticks(range(len(top_nodes)))
    ax.set_yticks(range(len(top_nodes)))
    ax.set_xticklabels(top_nodes, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(top_nodes, fontsize=8)
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def create_top_edges_lollipop(edge_df: pd.DataFrame, output_path: Path, title: str, top_n=30, min_weight=None):
    if not MATPLOTLIB_AVAILABLE or edge_df.empty:
        return
    filtered = filter_weighted_edges(edge_df, max_edges=top_n, min_weight=min_weight).head(top_n)
    if filtered.empty:
        return
    filtered = filtered.sort_values("weight", ascending=True)
    labels = [f"{row.source} - {row.target}" for row in filtered.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(13, max(7, len(filtered) * 0.34)))
    ax.hlines(y=range(len(filtered)), xmin=0, xmax=filtered["weight"], color="#A0A0A0", linewidth=1.5)
    ax.scatter(filtered["weight"], range(len(filtered)), color="#4C78A8", s=70)
    ax.set_yticks(range(len(filtered)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Connection weight")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def create_cluster_bar_summary(edge_df: pd.DataFrame, output_path: Path, title: str, max_edges=100, min_weight=None, thematic=False):
    if not MATPLOTLIB_AVAILABLE or edge_df.empty:
        return
    try:
        import networkx as nx
    except ImportError:
        return
    filtered = filter_weighted_edges(edge_df, max_edges=max_edges, min_weight=min_weight)
    graph = nx.Graph()
    for _, row in filtered.iterrows():
        graph.add_edge(row["source"], row["target"], weight=float(row["weight"]))
    if graph.number_of_nodes() == 0:
        return
    communities = detect_graph_communities(graph)
    rows = []
    for cluster in sorted(set(communities.values())):
        members = [node for node, cl in communities.items() if cl == cluster]
        theme_counts = Counter(node_theme(node) for node in members)
        rows.append({
            "cluster": cluster,
            "nodes": len(members),
            "dominant_theme": theme_counts.most_common(1)[0][0] if thematic and theme_counts else cluster,
            "top_nodes": ", ".join(sorted(members, key=lambda n: graph.degree(n, weight="weight"), reverse=True)[:6]),
        })
    summary = pd.DataFrame(rows).sort_values("nodes", ascending=True)
    labels = [f"{row.cluster}\n{row.dominant_theme}" for row in summary.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(10, max(5, len(summary) * 0.65)))
    ax.barh(labels, summary["nodes"], color="#54A24B", edgecolor="black")
    ax.set_xlabel("Nodes in cluster")
    ax.set_title(title, fontweight="bold")
    for y, row in enumerate(summary.itertuples(index=False)):
        ax.text(row.nodes + 0.2, y, row.top_nodes, va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def create_interactive_chord_like(edge_df: pd.DataFrame, output_html: Path, title: str, top_n=30, min_weight=None):
    if edge_df.empty:
        return False
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    filtered = filter_weighted_edges(edge_df, max_edges=top_n * 3, min_weight=min_weight)
    nodes = Counter()
    for _, row in filtered.iterrows():
        nodes[row["source"]] += float(row["weight"])
        nodes[row["target"]] += float(row["weight"])
    top_nodes = [node for node, _ in nodes.most_common(top_n)]
    idx = {node: i for i, node in enumerate(top_nodes)}
    matrix = np.zeros((len(top_nodes), len(top_nodes)))
    for _, row in filtered.iterrows():
        source, target, weight = row["source"], row["target"], float(row["weight"])
        if source in idx and target in idx:
            matrix[idx[source], idx[target]] += weight
            matrix[idx[target], idx[source]] += weight
    if matrix.sum() == 0:
        return False
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=top_nodes,
        y=top_nodes,
        colorscale="Viridis",
        hovertemplate="%{y} - %{x}<br>Weight: %{z}<extra></extra>",
    ))
    fig.update_layout(title=title, height=850, margin=dict(l=180, b=180, t=60, r=30))
    fig.write_html(str(output_html))
    return True


def write_plotly_static_if_possible(fig, output_png: Path, width=1400, height=850):
    try:
        fig.write_image(str(output_png), width=width, height=height, scale=2)
        return True
    except Exception:
        return False


def readable_node_label(value, max_chars=34):
    label = clean_ris_value(value).replace("_", " ")
    for prefix in ["From: ", "To: ", "Source: ", "Target: ", "Source theme: ", "Target theme: "]:
        if label.startswith(prefix):
            label = label[len(prefix):]
    label = re.sub(r"\s+", " ", label).strip()
    return label if len(label) <= max_chars else label[: max_chars - 1].rstrip() + "…"


def hex_to_rgba(hex_color, alpha=0.35):
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(76,120,168,{alpha})"
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return f"rgba(76,120,168,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"


def hex_to_rgba_tuple(hex_color, alpha=0.35):
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return (76 / 255, 120 / 255, 168 / 255, alpha)
    try:
        r = int(color[0:2], 16) / 255
        g = int(color[2:4], 16) / 255
        b = int(color[4:6], 16) / 255
    except ValueError:
        return (76 / 255, 120 / 255, 168 / 255, alpha)
    return (r, g, b, alpha)


def display_edge_subset(edge_df: pd.DataFrame, top_n=22, min_weight=None, max_nodes=26):
    filtered = filter_weighted_edges(edge_df, max_edges=max(top_n * 4, top_n), min_weight=min_weight)
    if filtered.empty:
        return filtered
    strength = Counter()
    for _, row in filtered.iterrows():
        strength[str(row["source"])] += float(row["weight"])
        strength[str(row["target"])] += float(row["weight"])
    keep_nodes = {node for node, _ in strength.most_common(max_nodes)}
    filtered = filtered[
        filtered["source"].astype(str).isin(keep_nodes)
        & filtered["target"].astype(str).isin(keep_nodes)
    ].head(top_n)
    if filtered.empty:
        filtered = filter_weighted_edges(edge_df, max_edges=top_n, min_weight=min_weight).head(top_n)
    return filtered.reset_index(drop=True)


def mermaid_csv_value(value):
    text = readable_node_label(value, max_chars=42).replace('"', '""')
    return f'"{text}"'


def export_mermaid_sankey(edge_df: pd.DataFrame, output_mmd: Path, title: str, top_n=18, min_weight=None):
    filtered = display_edge_subset(edge_df, top_n=top_n, min_weight=min_weight, max_nodes=24)
    if filtered.empty:
        return False
    lines = ["sankey-beta"]
    for _, row in filtered.iterrows():
        lines.append(
            f"{mermaid_csv_value(row['source'])},{mermaid_csv_value(row['target'])},{float(row['weight']):.6g}"
        )
    output_mmd.write_text("\n".join(lines), encoding="utf-8")
    html_path = output_mmd.with_suffix(".html")
    html_path.write_text(
        "\n".join([
            "<!doctype html>",
            "<html lang=\"en\"><head><meta charset=\"utf-8\">",
            f"<title>{clean_ris_value(title)}</title>",
            "<style>body{font-family:Arial,sans-serif;margin:28px;background:#fafafa;color:#222}"
            ".wrap{background:white;padding:22px;border:1px solid #ddd;max-width:1200px}"
            "h1{font-size:20px;margin:0 0 12px}.note{color:#555;font-size:13px}</style>",
            "</head><body><div class=\"wrap\">",
            f"<h1>{clean_ris_value(title)}</h1>",
            "<p class=\"note\">Mermaid Sankey: source, target, value. Labels were shortened to improve readability.</p>",
            "<pre class=\"mermaid\">",
            "\n".join(lines),
            "</pre>",
            "</div>",
            "<script type=\"module\">",
            "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';",
            "mermaid.initialize({startOnLoad:true,securityLevel:'loose',sankey:{width:1100,height:620,linkColor:'gradient',nodeAlignment:'justify',labelStyle:'outlined',nodeWidth:14,nodePadding:18}});",
            "</script></body></html>",
        ]),
        encoding="utf-8",
    )
    return True


def create_static_pair_flow(edge_df: pd.DataFrame, output_png: Path, title: str, top_n=24, min_weight=None):
    if not MATPLOTLIB_AVAILABLE or edge_df.empty:
        return False
    filtered = display_edge_subset(edge_df, top_n=top_n, min_weight=min_weight, max_nodes=24)
    if filtered.empty:
        return False

    left_nodes = list(dict.fromkeys(filtered["source"].astype(str).tolist()))
    right_nodes = list(dict.fromkeys(filtered["target"].astype(str).tolist()))
    left_y = {node: idx for idx, node in enumerate(left_nodes)}
    right_y = {node: idx for idx, node in enumerate(right_nodes)}
    max_rows = max(len(left_nodes), len(right_nodes), 2)
    max_weight = max(float(filtered["weight"].max()), 1.0)

    fig, ax = plt.subplots(figsize=(15, max(7.5, max_rows * 0.42)))
    for _, row in filtered.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        weight = float(row["weight"])
        y0 = max_rows - 1 - left_y[source]
        y1 = max_rows - 1 - right_y[target]
        color = THEME_COLORS.get(node_theme(source), "#4C78A8")
        ax.plot([0.08, 0.92], [y0, y1], color=color, alpha=0.22 + 0.45 * weight / max_weight, linewidth=0.8 + 5 * weight / max_weight)

    for node, idx in left_y.items():
        y = max_rows - 1 - idx
        ax.scatter(0.06, y, s=90, color=THEME_COLORS.get(node_theme(node), "#9E9E9E"), edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(0.04, y, readable_node_label(node), ha="right", va="center", fontsize=8)
    for node, idx in right_y.items():
        y = max_rows - 1 - idx
        ax.scatter(0.94, y, s=90, color=THEME_COLORS.get(node_theme(node), "#9E9E9E"), edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(0.96, y, readable_node_label(node), ha="left", va="center", fontsize=8)

    ax.text(0.06, max_rows + 0.15, "Source", ha="center", fontweight="bold")
    ax.text(0.94, max_rows + 0.15, "Target", ha="center", fontweight="bold")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-1, max_rows + 0.7)
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close()
    return True


def create_static_theme_flow(edge_df: pd.DataFrame, output_png: Path, title: str, top_n=42, min_weight=None):
    if not MATPLOTLIB_AVAILABLE or edge_df.empty:
        return False
    filtered = filter_weighted_edges(edge_df, max_edges=top_n, min_weight=min_weight).head(top_n)
    if filtered.empty:
        return False

    theme_edges = Counter()
    for _, row in filtered.iterrows():
        theme_edges[(node_theme(row["source"]), node_theme(row["target"]))] += float(row["weight"])
    theme_df = pd.DataFrame([
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in theme_edges.items()
    ]).sort_values("weight", ascending=False)
    return create_static_pair_flow(theme_df, output_png, title, top_n=len(theme_df), min_weight=None)


def create_static_circular_interconnection(edge_df: pd.DataFrame, output_png: Path, title: str, top_n=32, min_weight=None):
    if not MATPLOTLIB_AVAILABLE or edge_df.empty:
        return False
    filtered = filter_weighted_edges(edge_df, max_edges=top_n, min_weight=min_weight).head(top_n)
    if filtered.empty:
        return False
    node_strength = Counter()
    for _, row in filtered.iterrows():
        node_strength[row["source"]] += float(row["weight"])
        node_strength[row["target"]] += float(row["weight"])
    nodes = [node for node, _ in node_strength.most_common(32)]
    if len(nodes) < 2:
        return False
    positions = {
        node: (
            math.cos(2 * math.pi * idx / len(nodes)),
            math.sin(2 * math.pi * idx / len(nodes)),
        )
        for idx, node in enumerate(nodes)
    }
    max_weight = max(float(filtered["weight"].max()), 1.0)
    max_strength = max(node_strength.values()) if node_strength else 1.0

    fig, ax = plt.subplots(figsize=(11, 11))
    for _, row in filtered.iterrows():
        source, target, weight = row["source"], row["target"], float(row["weight"])
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.plot([x0, x1], [y0, y1], color="#4C78A8", alpha=0.12 + 0.28 * weight / max_weight, linewidth=0.4 + 3 * weight / max_weight)

    for node in nodes:
        x, y = positions[node]
        ax.scatter(x, y, s=90 + 260 * node_strength[node] / max_strength, color=THEME_COLORS.get(node_theme(node), "#9E9E9E"), edgecolor="white", linewidth=1.2, zorder=3)
        if node_strength[node] >= sorted(node_strength.values(), reverse=True)[min(11, len(node_strength) - 1)]:
            ax.text(x * 1.16, y * 1.16, readable_node_label(node, max_chars=22), ha="center", va="center", fontsize=7)

    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close()
    return True


def create_pair_sankey(edge_df: pd.DataFrame, output_html: Path, title: str, top_n=40, min_weight=None, output_png: Path = None):
    if edge_df.empty:
        return False
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False

    filtered = display_edge_subset(edge_df, top_n=min(top_n, 22), min_weight=min_weight, max_nodes=26)
    if filtered.empty:
        return False

    left_labels = [f"From: {readable_node_label(value)}" for value in filtered["source"].astype(str)]
    right_labels = [f"To: {readable_node_label(value)}" for value in filtered["target"].astype(str)]
    labels = []
    for label in left_labels + right_labels:
        if label not in labels:
            labels.append(label)
    label_index = {label: idx for idx, label in enumerate(labels)}

    colors = []
    for label in labels:
        theme = node_theme(label)
        colors.append(THEME_COLORS.get(theme, "#9E9E9E"))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            pad=22,
            thickness=18,
            color=colors,
            line=dict(color="white", width=0.8),
        ),
        link=dict(
            source=[label_index[f"From: {readable_node_label(row.source)}"] for row in filtered.itertuples(index=False)],
            target=[label_index[f"To: {readable_node_label(row.target)}"] for row in filtered.itertuples(index=False)],
            value=[float(row.weight) for row in filtered.itertuples(index=False)],
            color=[
                hex_to_rgba(THEME_COLORS.get(node_theme(row.source), "#4C78A8"), 0.34)
                for row in filtered.itertuples(index=False)
            ],
        ),
    )])
    fig.update_layout(title_text=title, font_size=11, height=max(620, 26 * len(labels)), margin=dict(l=20, r=20, t=70, b=20))
    fig.write_html(str(output_html))
    if output_png is not None:
        if not write_plotly_static_if_possible(fig, output_png):
            create_static_pair_flow(edge_df, output_png, title, top_n=min(top_n, 28), min_weight=min_weight)
    return True


def create_theme_alluvial(edge_df: pd.DataFrame, output_html: Path, title: str, top_n=60, min_weight=None, output_png: Path = None):
    if edge_df.empty:
        return False
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False

    filtered = display_edge_subset(edge_df, top_n=min(top_n, 28), min_weight=min_weight, max_nodes=26)
    if filtered.empty:
        return False

    flows = Counter()
    for _, row in filtered.iterrows():
        source_theme = f"Source theme: {node_theme(row['source']).title()}"
        source_node = f"Source: {readable_node_label(row['source'])}"
        target_theme = f"Target theme: {node_theme(row['target']).title()}"
        target_node = f"Target: {readable_node_label(row['target'])}"
        weight = float(row["weight"])
        flows[(source_theme, source_node)] += weight
        flows[(source_node, target_theme)] += weight
        flows[(target_theme, target_node)] += weight

    labels = []
    for source, target in flows:
        if source not in labels:
            labels.append(source)
        if target not in labels:
            labels.append(target)
    label_index = {label: idx for idx, label in enumerate(labels)}
    node_colors = [THEME_COLORS.get(node_theme(label), "#9E9E9E") for label in labels]
    link_colors = ["rgba(84, 162, 75, 0.25)" if "theme" in source.lower() else "rgba(245, 133, 24, 0.25)" for source, _ in flows]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=20, thickness=16, color=node_colors, line=dict(color="white", width=0.8)),
        link=dict(
            source=[label_index[source] for source, _ in flows],
            target=[label_index[target] for _, target in flows],
            value=[value for value in flows.values()],
            color=link_colors,
        ),
    )])
    fig.update_layout(title_text=title, font_size=10, height=820, margin=dict(l=20, r=20, t=70, b=20))
    fig.write_html(str(output_html))
    if output_png is not None:
        if not write_plotly_static_if_possible(fig, output_png, height=900):
            create_static_theme_flow(edge_df, output_png, title, top_n=min(top_n, 48), min_weight=min_weight)
    return True


def create_circular_interconnection(edge_df: pd.DataFrame, output_html: Path, title: str, top_n=32, min_weight=None, output_png: Path = None):
    if edge_df.empty:
        return False
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False

    filtered = display_edge_subset(edge_df, top_n=min(top_n, 24), min_weight=min_weight, max_nodes=18)
    if filtered.empty:
        return False

    node_strength = Counter()
    for _, row in filtered.iterrows():
        node_strength[row["source"]] += float(row["weight"])
        node_strength[row["target"]] += float(row["weight"])
    nodes = [node for node, _ in node_strength.most_common(18)]
    if len(nodes) < 2:
        return False
    positions = {
        node: (
            math.cos(2 * math.pi * idx / len(nodes)),
            math.sin(2 * math.pi * idx / len(nodes)),
        )
        for idx, node in enumerate(nodes)
    }

    edge_traces = []
    max_weight = max(float(weight) for weight in filtered["weight"]) if not filtered.empty else 1.0
    for _, row in filtered.iterrows():
        source, target, weight = row["source"], row["target"], float(row["weight"])
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_traces.append(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=0.8 + 3.6 * weight / max_weight, color="rgba(76,120,168,0.24)"),
            hoverinfo="text",
            text=f"{source} - {target}<br>Weight: {weight}",
            showlegend=False,
        ))

    node_trace = go.Scatter(
        x=[positions[node][0] for node in nodes],
        y=[positions[node][1] for node in nodes],
        mode="markers+text",
        text=[readable_node_label(node, max_chars=24) for node in nodes],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=[18 + 24 * node_strength[node] / max(node_strength.values()) for node in nodes],
            color=[THEME_COLORS.get(node_theme(node), "#9E9E9E") for node in nodes],
            line=dict(width=1, color="white"),
        ),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title, height=850, plot_bgcolor="white", margin=dict(l=20, r=20, t=70, b=20))
    fig.write_html(str(output_html))
    if output_png is not None:
        if not write_plotly_static_if_possible(fig, output_png):
            create_static_circular_interconnection(edge_df, output_png, title, top_n=top_n, min_weight=min_weight)
    return True


def remove_obsolete_network_images(img_dir: Path, filenames):
    for filename in filenames:
        path = img_dir / filename
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def create_readable_network_representations(edge_df: pd.DataFrame, img_dir: Path, prefix: str, title: str, min_weight=None, thematic=False):
    create_adjacency_heatmap(
        edge_df,
        img_dir / f"{prefix}_adjacency_heatmap.png",
        f"{title} - Adjacency Heatmap",
        top_n=28,
        min_weight=min_weight,
    )
    create_top_edges_lollipop(
        edge_df,
        img_dir / f"{prefix}_top_connections_lollipop.png",
        f"{title} - Strongest Connections",
        top_n=32,
        min_weight=min_weight,
    )
    create_cluster_bar_summary(
        edge_df,
        img_dir / f"{prefix}_cluster_summary.png",
        f"{title} - Cluster Summary",
        max_edges=110,
        min_weight=min_weight,
        thematic=thematic,
    )
    create_interactive_chord_like(
        edge_df,
        img_dir / f"{prefix}_interactive_matrix.html",
        f"{title} - Interactive Matrix",
        top_n=35,
        min_weight=min_weight,
    )
    create_pair_sankey(
        edge_df,
        img_dir / f"{prefix}_top_connections_sankey.html",
        f"{title} - Sankey of Strongest Connections",
        top_n=20,
        min_weight=min_weight,
        output_png=img_dir / f"{prefix}_top_connections_sankey.png",
    )
    export_mermaid_sankey(
        edge_df,
        img_dir / f"{prefix}_top_connections_sankey_mermaid.mmd",
        f"{title} - Mermaid Sankey",
        top_n=16,
        min_weight=min_weight,
    )
    create_circular_interconnection(
        edge_df,
        img_dir / f"{prefix}_circular_interconnection.html",
        f"{title} - Circular Interconnection",
        top_n=22,
        min_weight=min_weight,
        output_png=img_dir / f"{prefix}_circular_interconnection.png",
    )




def create_keyword_correlation_heatmap(keyword_lists, output_path: Path, top_n=25):
    if not MATPLOTLIB_AVAILABLE:
        return pd.DataFrame()
    keyword_counter = Counter()
    for keywords in keyword_lists:
        keyword_counter.update(set(keywords))
    top_keywords = [word for word, _ in keyword_counter.most_common(top_n)]
    if len(top_keywords) < 2:
        return pd.DataFrame()

    matrix = []
    for keywords in keyword_lists:
        keyword_set = set(keywords)
        matrix.append([1 if word in keyword_set else 0 for word in top_keywords])
    data = np.asarray(matrix)
    if data.shape[0] < 2:
        return pd.DataFrame()

    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr_df = pd.DataFrame(corr, index=top_keywords, columns=top_keywords)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_keywords)))
    ax.set_yticks(range(len(top_keywords)))
    ax.set_xticklabels(top_keywords, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(top_keywords, fontsize=8)
    ax.set_title("Keyword Correlation Heatmap", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return corr_df


def create_world_map(country_counts: pd.DataFrame, output_html: Path):
    if country_counts.empty:
        return False
    try:
        import plotly.express as px
    except ImportError:
        return False
    fig = px.choropleth(
        country_counts,
        locations="Country",
        locationmode="country names",
        color="Documents",
        hover_name="Country",
        hover_data={"Institutions": True, "Documents": True},
        color_continuous_scale="Viridis",
        title="Institutions / Affiliations by Country",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=55, b=0))
    fig.write_html(str(output_html))
    return True


def create_variable_correlation_outputs(df: pd.DataFrame, keyword_lists, country_lists, results_dir: Path):
    dirs = output_dirs(results_dir)
    biblio_dir = dirs["bibliometric"]
    img_dir = biblio_dir / "IMAGES_BIBLIOMETRICAS"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, row in df.iterrows():
        title_tokens = tokenize_for_vocabulary(row.get("TITLE", ""))
        abstract_tokens = tokenize_for_vocabulary(row.get("ABSTRACT", ""))
        authors = lda.split_author_names(row.get("AUTHOR", "")) if "AUTHOR" in df.columns else []
        keywords = keyword_lists[idx] if idx < len(keyword_lists) else []
        countries = country_lists[idx] if idx < len(country_lists) else []
        text_tokens = title_tokens + abstract_tokens + keywords
        token_counter = Counter(text_tokens)
        group_counts = {
            group: sum(token_counter[term] for term in terms)
            for group, terms in OBJECTIVE_GROUPS.items()
        }
        rows.append({
            "year": pd.to_numeric(row.get("YEAR", np.nan), errors="coerce"),
            "author_count": len(authors),
            "keyword_count": len(set(keywords)),
            "country_count": len(set(countries)),
            "title_token_count": len(title_tokens),
            "abstract_token_count": len(abstract_tokens),
            "poultry_term_count": group_counts["poultry"],
            "technology_term_count": group_counts["technology"],
            "economic_term_count": group_counts["economic"],
            "objective_total_count": sum(group_counts.values()),
        })

    variable_df = pd.DataFrame(rows)
    corr_df = variable_df.corr(numeric_only=True).fillna(0)
    with pd.ExcelWriter(biblio_dir / "03_bibliometric_variable_correlations.xlsx") as writer:
        variable_df.to_excel(writer, sheet_name="Document_Variables", index=False)
        corr_df.to_excel(writer, sheet_name="Correlation_Matrix")

    if MATPLOTLIB_AVAILABLE and not corr_df.empty:
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_xticklabels(corr_df.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(corr_df.index, fontsize=8)
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Bibliometric Variable Correlation Matrix", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(img_dir / "12_bibliometric_variable_correlation_matrix.png", dpi=200, bbox_inches="tight")
        plt.close()


def create_three_field_sankey(df: pd.DataFrame, keyword_lists, output_html: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False

    source_counter = Counter()
    keyword_counter = Counter()
    year_counter = Counter()
    records = []
    for idx, row in df.iterrows():
        source = str(row.get("JOURNAL", "")).strip() or "Unknown source"
        year = str(row.get("YEAR", "")).strip() or "Unknown year"
        keywords = keyword_lists[idx] if idx < len(keyword_lists) else []
        objective_keywords = [kw for kw in keywords if kw in SEARCH_OBJECTIVE_KEEP_TERMS or "_" in kw]
        if not objective_keywords:
            objective_keywords = keywords[:3]
        for keyword in objective_keywords[:4]:
            records.append((source, keyword, year))
            source_counter[source] += 1
            keyword_counter[keyword] += 1
            year_counter[year] += 1

    top_sources = {item for item, _ in source_counter.most_common(12)}
    top_keywords = {item for item, _ in keyword_counter.most_common(18)}
    top_years = {item for item, _ in year_counter.most_common(5)}

    left_mid = Counter()
    mid_right = Counter()
    for source, keyword, year in records:
        if source in top_sources and keyword in top_keywords:
            left_mid[(source, keyword)] += 1
        if keyword in top_keywords and year in top_years:
            mid_right[(keyword, year)] += 1

    labels = []
    for values in [top_sources, top_keywords, top_years]:
        for value in sorted(values):
            if value not in labels:
                labels.append(value)
    index = {label: i for i, label in enumerate(labels)}

    sources = []
    targets = []
    values = []
    for (left, right), weight in list(left_mid.items()) + list(mid_right.items()):
        sources.append(index[left])
        targets.append(index[right])
        values.append(weight)

    if not values:
        return False

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=15),
        link=dict(source=sources, target=targets, value=values),
    )])
    fig.update_layout(title_text="Three-Field Bibliometric Map: Sources -> Keywords -> Years", font_size=10)
    fig.write_html(str(output_html))
    return True


def keyword_objective_axis(keyword: str) -> str:
    keyword = str(keyword).lower()
    keyword_tokens = set(re.split(r"[_\s-]+", keyword))
    for axis, terms in OBJECTIVE_GROUPS.items():
        if keyword in terms:
            return axis
    if any(term in keyword for term in ["poultry", "broiler", "layer", "chicken", "hen", "flock"]):
        return "poultry"
    if any(term in keyword for term in ["digital", "precision", "sensor", "vision", "learning", "automation", "blockchain"]):
        return "technology"
    if keyword_tokens.intersection({"ai", "iot"}):
        return "technology"
    if any(term in keyword for term in ["cost", "economic", "feasibility", "profit", "investment", "viability"]):
        return "economic"
    if keyword_tokens.intersection({"roi"}):
        return "economic"
    return "other"


def sankey_axis_color(label: str) -> str:
    text = str(label).lower()
    if "poultry" in text or "broiler" in text or "layer" in text:
        return THEME_COLORS["poultry"]
    if "technology" in text or "digital" in text or "sensor" in text or "learning" in text:
        return THEME_COLORS["technology"]
    if "economic" in text or "cost" in text or "roi" in text or "feasibility" in text:
        return THEME_COLORS["economic"]
    if text.startswith("year:"):
        return "#6B7280"
    if text.startswith("source:"):
        return "#8B5CF6"
    if text.startswith("author:"):
        return "#14B8A6"
    if text.startswith("keyword:"):
        return "#F97316"
    return THEME_COLORS["other"]


def create_static_axis_flow(records, output_png: Path, title: str, axis_order: list, top_edges=90):
    if not MATPLOTLIB_AVAILABLE or not records:
        return False
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    edge_counts = Counter(records)
    strongest = edge_counts.most_common(top_edges)
    if not strongest:
        return False

    node_counts = Counter()
    for (source, target), value in strongest:
        node_counts[source] += value
        node_counts[target] += value

    labels_by_axis = {}
    for axis_name in axis_order:
        prefix = f"{axis_name}:"
        labels_by_axis[axis_name] = [
            node for node, _ in node_counts.most_common()
            if str(node).startswith(prefix)
        ][:18]
    labels_kept = {node for nodes in labels_by_axis.values() for node in nodes}

    fig_width = max(12, 3.0 * len(axis_order))
    fig, ax = plt.subplots(figsize=(fig_width, 9))
    x_by_axis = {
        axis_name: idx / max(len(axis_order) - 1, 1)
        for idx, axis_name in enumerate(axis_order)
    }
    positions = {}
    for axis_name, nodes in labels_by_axis.items():
        x = x_by_axis[axis_name]
        for rank, node in enumerate(nodes):
            y = 1 - ((rank + 1) / (len(nodes) + 1))
            positions[node] = (x, y)
            color = sankey_axis_color(node)
            ax.scatter(x, y, s=90 + min(520, node_counts[node] * 8), color=color, edgecolor="white", linewidth=1.0, zorder=4)
            label = f"{readable_node_label(node, max_chars=24)} ({int(node_counts[node])})"
            ha = "right" if x > 0.75 else ("left" if x < 0.25 else "center")
            dx = -0.018 if ha == "right" else (0.018 if ha == "left" else 0)
            ax.text(x + dx, y, label, fontsize=7.5, va="center", ha=ha, zorder=5)
        ax.text(x, 1.035, axis_name, fontsize=11, fontweight="bold", ha="center")

    max_weight = max(value for _, value in strongest) if strongest else 1
    for (source, target), value in strongest:
        if source not in labels_kept or target not in labels_kept:
            continue
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        verts = [
            (x0, y0),
            (x0 + (x1 - x0) * 0.45, y0),
            (x0 + (x1 - x0) * 0.55, y1),
            (x1, y1),
        ]
        path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=hex_to_rgba_tuple(sankey_axis_color(source), 0.30),
            lw=0.35 + 5.0 * (value / max_weight),
            capstyle="round",
            zorder=2,
        )
        ax.add_patch(patch)

    ax.set_title(title, fontweight="bold", pad=18)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.04, 1.08)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def create_axis_sankey(records, output_html: Path, title: str, axis_order: list, output_png: Path = None, top_edges=140):
    """Create a ggalluvial-like Sankey with fixed conceptual axes."""
    if not records:
        return False
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False

    edge_counts = Counter(records)
    strongest = edge_counts.most_common(top_edges)
    if not strongest:
        return False

    node_counts = Counter()
    for (source, target), value in strongest:
        node_counts[source] += value
        node_counts[target] += value

    labels = []
    for axis_name in axis_order:
        prefix = f"{axis_name}:"
        labels.extend([
            node for node, _ in node_counts.most_common()
            if str(node).startswith(prefix) and node not in labels
        ])
    labels.extend([node for node, _ in node_counts.most_common() if node not in labels])
    label_index = {label: idx for idx, label in enumerate(labels)}

    axis_x = {
        axis_name: idx / max(len(axis_order) - 1, 1)
        for idx, axis_name in enumerate(axis_order)
    }
    grouped_by_axis = defaultdict(list)
    for label in labels:
        grouped_by_axis[str(label).split(":", 1)[0]].append(label)

    x_positions = []
    y_positions = []
    for label in labels:
        axis_name = str(label).split(":", 1)[0]
        group = grouped_by_axis.get(axis_name, [label])
        rank = group.index(label)
        x_positions.append(axis_x.get(axis_name, 0.5))
        y_positions.append((rank + 0.5) / max(len(group), 1))

    fig = go.Figure(data=[go.Sankey(
        arrangement="fixed",
        node=dict(
            label=[
                f"{readable_node_label(label, max_chars=34)} ({int(node_counts[label])})"
                for label in labels
            ],
            x=x_positions,
            y=y_positions,
            pad=18,
            thickness=17,
            color=[sankey_axis_color(label) for label in labels],
            line=dict(color="white", width=0.9),
        ),
        link=dict(
            source=[label_index[source] for (source, _), _ in strongest],
            target=[label_index[target] for (_, target), _ in strongest],
            value=[float(value) for _, value in strongest],
            color=[hex_to_rgba(sankey_axis_color(source), 0.30) for (source, _), _ in strongest],
        ),
    )])
    fig.update_layout(
        title_text=title,
        font_size=18,
        height=1150,
        margin=dict(l=45, r=45, t=95, b=35),
    )
    fig.write_html(str(output_html))
    if output_png is not None:
        if not write_plotly_static_if_possible(fig, output_png, height=1150):
            create_static_axis_flow(records, output_png, title, axis_order, top_edges=min(top_edges, 90))
    return True


def create_flow_records_excel(records, output_xlsx: Path):
    if not records:
        return
    pd.DataFrame([
        {"from": source, "to": target, "weight": weight}
        for (source, target), weight in Counter(records).most_common()
    ]).to_excel(output_xlsx, index=False)


def node_theme(node: str) -> str:
    node_l = str(node).lower()
    node_tokens = set(re.split(r"[_\s-]+", node_l))
    axis = keyword_objective_axis(node_l)
    if axis != "other":
        return axis
    if any(term in node_l for term in ["cost", "economic", "feasibility", "profit", "investment"]):
        return "economic"
    if node_tokens.intersection({"roi"}):
        return "economic"
    if any(term in node_l for term in ["poultry", "broiler", "layer", "hen", "chicken", "egg", "flock"]):
        return "poultry"
    if any(term in node_l for term in ["digital", "sensor", "learning", "vision", "automation", "precision", "blockchain"]):
        return "technology"
    if node_tokens.intersection({"ai", "iot"}):
        return "technology"
    return "other"


def filter_weighted_edges(edge_df: pd.DataFrame, max_edges=60, min_weight=None):
    if edge_df.empty:
        return edge_df.copy()
    work = edge_df.copy()
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(0)
    if min_weight is None:
        q75 = work["weight"].quantile(0.75)
        min_weight = max(2, int(q75))
    filtered = work[work["weight"] >= min_weight].sort_values("weight", ascending=False)
    if len(filtered) < min(12, len(work)):
        filtered = work.sort_values("weight", ascending=False).head(max_edges)
    else:
        filtered = filtered.head(max_edges)
    return filtered.reset_index(drop=True)


def detect_graph_communities(graph):
    try:
        import networkx as nx
        communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
        mapping = {}
        for idx, community in enumerate(communities, 1):
            for node in community:
                mapping[node] = f"cluster_{idx}"
        return mapping
    except Exception:
        return {node: "cluster_1" for node in graph.nodes()}


def create_creative_bibliometric_html(df: pd.DataFrame, keyword_lists, author_lists, results_dir: Path):
    dirs = output_dirs(results_dir)
    biblio_dir = dirs["bibliometric"]
    img_dir = biblio_dir / "IMAGES_BIBLIOMETRICAS"
    img_dir.mkdir(parents=True, exist_ok=True)
    remove_obsolete_network_images(img_dir, [
        "17_sankey_year_source_axis_keyword.html",
        "21_sankey_author_axis_keyword.html",
    ])

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        pd.DataFrame([{
            "message": "Plotly is not installed. Creative interactive bibliometric charts were skipped."
        }]).to_excel(biblio_dir / "06_interactive_charts_not_generated.xlsx", index=False)
        return

    rows = []
    for idx, row in df.iterrows():
        year = str(row.get("YEAR", "")).strip() or "Unknown year"
        source = str(row.get("JOURNAL", "")).strip() or "Unknown source"
        authors = author_lists[idx] if idx < len(author_lists) else []
        keywords = keyword_lists[idx] if idx < len(keyword_lists) else []
        if not keywords:
            keywords = [token for token in tokenize_for_vocabulary(row.get("TITLE", "")) if token in SEARCH_OBJECTIVE_KEEP_TERMS]
        keywords = normalize_keyword_terms(keywords)
        for keyword in sorted(set(keywords))[:10]:
            axis = keyword_objective_axis(keyword)
            axis_label = ALLUVIAL_AXIS_LABELS.get(axis, "Other")
            for author in (authors[:5] if authors else ["Unknown author"]):
                rows.append({
                    "Year": year,
                    "Source": source,
                    "Author": author,
                    "Keyword": keyword,
                    "Objective_Axis": axis_label,
                })

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return

    top_sources = set(long_df["Source"].value_counts().head(15).index)
    top_keywords = set(long_df["Keyword"].value_counts().head(24).index)
    for _, axis_df in long_df.groupby("Objective_Axis"):
        top_keywords.update(axis_df["Keyword"].value_counts().head(10).index)
    top_authors = set(long_df["Author"].value_counts().head(30).index)
    plot_df = long_df[
        long_df["Source"].isin(top_sources)
        & long_df["Keyword"].isin(top_keywords)
        & long_df["Author"].isin(top_authors)
    ].copy()

    # Alluvial-style Sankey 1: Year -> source -> objective axis -> keyword.
    sankey_records = []
    for _, row in plot_df.iterrows():
        sankey_records.extend([
            (f"Year: {row['Year']}", f"Source: {row['Source']}"),
            (f"Source: {row['Source']}", f"Research axis: {row['Objective_Axis']}"),
            (f"Research axis: {row['Objective_Axis']}", f"Keyword: {row['Keyword']}"),
        ])
    create_axis_sankey(
        sankey_records,
        img_dir / "17_alluvial_year_source_axis_keyword.html",
        "Alluvial Map: Year -> Source -> Research Axis -> Keyword",
        axis_order=["Year", "Source", "Research axis", "Keyword"],
        output_png=img_dir / "17_alluvial_year_source_axis_keyword.png",
        top_edges=130,
    )
    create_flow_records_excel(sankey_records, biblio_dir / "17_alluvial_year_source_axis_keyword_edges.xlsx")

    # Sunburst: objective axis -> keyword -> source.
    sun_df = (
        plot_df.groupby(["Objective_Axis", "Keyword", "Source"])
        .size()
        .reset_index(name="Occurrences")
    )
    fig = px.sunburst(
        sun_df,
        path=["Objective_Axis", "Keyword", "Source"],
        values="Occurrences",
        title="Sunburst: Objective Axis -> Keyword -> Source",
    )
    fig.write_html(str(img_dir / "18_sunburst_axis_keyword_source.html"))

    # Treemap: source -> objective axis -> keyword.
    tree_df = (
        plot_df.groupby(["Source", "Objective_Axis", "Keyword"])
        .size()
        .reset_index(name="Occurrences")
    )
    fig = px.treemap(
        tree_df,
        path=["Source", "Objective_Axis", "Keyword"],
        values="Occurrences",
        color="Objective_Axis",
        title="Treemap: Sources, Objective Axes and Keywords",
    )
    fig.write_html(str(img_dir / "19_sources_axes_keywords_treemap.html"))

    # Bubble evolution: keyword x year, sized by occurrence, colored by objective axis.
    bubble_df = (
        plot_df.groupby(["Year", "Keyword", "Objective_Axis"])
        .size()
        .reset_index(name="Occurrences")
    )
    bubble_df = bubble_df[bubble_df["Keyword"].isin(bubble_df.groupby("Keyword")["Occurrences"].sum().nlargest(25).index)]
    fig = px.scatter(
        bubble_df,
        x="Year",
        y="Keyword",
        size="Occurrences",
        color="Objective_Axis",
        hover_data=["Occurrences"],
        title="Thematic Bubble Map: Keywords by Year",
    )
    fig.write_html(str(img_dir / "20_bubble_keywords_by_year.html"))

    # Author -> objective axis -> keyword alluvial-style Sankey.
    author_plot = plot_df[plot_df["Author"].isin(top_authors)].copy()
    sankey_records = []
    for _, row in author_plot.iterrows():
        sankey_records.extend([
            (f"Author: {row['Author']}", f"Research axis: {row['Objective_Axis']}"),
            (f"Research axis: {row['Objective_Axis']}", f"Keyword: {row['Keyword']}"),
        ])
    create_axis_sankey(
        sankey_records,
        img_dir / "21_alluvial_author_axis_keyword.html",
        "Alluvial Map: Authors -> Research Axis -> Keywords",
        axis_order=["Author", "Research axis", "Keyword"],
        output_png=img_dir / "21_alluvial_author_axis_keyword.png",
        top_edges=120,
    )
    create_flow_records_excel(sankey_records, biblio_dir / "21_alluvial_author_axis_keyword_edges.xlsx")

    # Year -> objective axis -> author -> keyword, adapted from the ggalluvial idea.
    sankey_records = []
    year_author_plot = plot_df[
        plot_df["Author"].isin(plot_df["Author"].value_counts().head(18).index)
        & plot_df["Keyword"].isin(plot_df["Keyword"].value_counts().head(24).index)
    ].copy()
    for _, row in year_author_plot.iterrows():
        sankey_records.extend([
            (f"Year: {row['Year']}", f"Research axis: {row['Objective_Axis']}"),
            (f"Research axis: {row['Objective_Axis']}", f"Author: {row['Author']}"),
            (f"Author: {row['Author']}", f"Keyword: {row['Keyword']}"),
        ])
    create_axis_sankey(
        sankey_records,
        img_dir / "22_alluvial_year_axis_author_keyword.html",
        "Alluvial Map: Year -> Research Axis -> Author -> Keyword",
        axis_order=["Year", "Research axis", "Author", "Keyword"],
        output_png=img_dir / "22_alluvial_year_axis_author_keyword.png",
        top_edges=140,
    )
    create_flow_records_excel(sankey_records, biblio_dir / "22_alluvial_year_axis_author_keyword_edges.xlsx")

    # Year -> objective axis -> keyword -> source: cleaner thematic evolution.
    sankey_records = []
    year_keyword_plot = plot_df[
        plot_df["Keyword"].isin(plot_df["Keyword"].value_counts().head(24).index)
        & plot_df["Source"].isin(plot_df["Source"].value_counts().head(12).index)
    ].copy()
    for _, row in year_keyword_plot.iterrows():
        sankey_records.extend([
            (f"Year: {row['Year']}", f"Research axis: {row['Objective_Axis']}"),
            (f"Research axis: {row['Objective_Axis']}", f"Keyword: {row['Keyword']}"),
            (f"Keyword: {row['Keyword']}", f"Source: {row['Source']}"),
        ])
    create_axis_sankey(
        sankey_records,
        img_dir / "23_alluvial_year_axis_keyword_source.html",
        "Alluvial Map: Year -> Research Axis -> Keyword -> Source",
        axis_order=["Year", "Research axis", "Keyword", "Source"],
        output_png=img_dir / "23_alluvial_year_axis_keyword_source.png",
        top_edges=140,
    )
    create_flow_records_excel(sankey_records, biblio_dir / "23_alluvial_year_axis_keyword_source_edges.xlsx")

    long_df.to_excel(biblio_dir / "06_creative_interactive_chart_data.xlsx", index=False)


def create_country_collaboration_outputs(country_lists, results_dir: Path):
    dirs = output_dirs(results_dir)
    biblio_dir = dirs["bibliometric"]
    img_dir = biblio_dir / "IMAGES_BIBLIOMETRICAS"
    img_dir.mkdir(parents=True, exist_ok=True)

    country_docs = Counter()
    country_scp = Counter()
    country_mcp = Counter()
    collaboration_edges = Counter()

    for countries in country_lists:
        countries = sorted(set(countries))
        if not countries:
            continue
        for country in countries:
            country_docs[country] += 1
            if len(countries) == 1:
                country_scp[country] += 1
            else:
                country_mcp[country] += 1
        for left, right in itertools.combinations(countries, 2):
            collaboration_edges[(left, right)] += 1

    country_df = pd.DataFrame([
        {
            "Country": country,
            "Documents": country_docs[country],
            "SCP_single_country_publications": country_scp[country],
            "MCP_multiple_country_publications": country_mcp[country],
            "MCP_Ratio": country_mcp[country] / country_docs[country] if country_docs[country] else 0,
        }
        for country in sorted(country_docs)
    ]).sort_values("Documents", ascending=False) if country_docs else pd.DataFrame(columns=[
        "Country", "Documents", "SCP_single_country_publications", "MCP_multiple_country_publications", "MCP_Ratio"
    ])

    collab_df = pd.DataFrame([
        {"source": left, "target": right, "weight": weight}
        for (left, right), weight in collaboration_edges.items()
    ]).sort_values("weight", ascending=False) if collaboration_edges else pd.DataFrame(columns=["source", "target", "weight"])

    with pd.ExcelWriter(biblio_dir / "04_country_production_and_collaboration.xlsx") as writer:
        country_df.to_excel(writer, sheet_name="Country_Production_SCP_MCP", index=False)
        collab_df.to_excel(writer, sheet_name="Country_Collaboration_Edges", index=False)

    if not country_df.empty:
        create_world_map(
            country_df.assign(Institutions=0),
            img_dir / "13_mapa_mundi_producao_bibliografica.html",
        )
        if MATPLOTLIB_AVAILABLE:
            plot_df = country_df.head(20).sort_values("Documents", ascending=True)
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.barh(plot_df["Country"], plot_df["Documents"], color="#4C78A8", edgecolor="black", label="Total")
            ax.barh(plot_df["Country"], plot_df["MCP_multiple_country_publications"], color="#F58518", edgecolor="black", label="MCP")
            ax.set_title("Country Scientific Production and International Collaboration", fontweight="bold")
            ax.set_xlabel("Documents")
            ax.legend()
            plt.tight_layout()
            plt.savefig(img_dir / "14_producao_paises_scp_mcp.png", dpi=200, bbox_inches="tight")
            plt.close()

    return country_df, collab_df


def create_author_keyword_bipartite(author_lists, keyword_lists, results_dir: Path):
    dirs = output_dirs(results_dir)
    biblio_dir = dirs["bibliometric"]
    img_dir = biblio_dir / "IMAGES_BIBLIOMETRICAS"
    img_dir.mkdir(parents=True, exist_ok=True)

    edge_counter = Counter()
    for authors, keywords in zip(author_lists, keyword_lists):
        authors = authors[:6]
        keywords = normalize_keyword_terms(keywords)
        keywords = [
            kw for kw in keywords
            if keyword_objective_axis(kw) in OBJECTIVE_AXIS_LABELS
        ][:6]
        for author in authors:
            for keyword in keywords:
                edge_counter[(author, keyword)] += 1

    edge_df = pd.DataFrame([
        {"source": author, "target": keyword, "weight": weight}
        for (author, keyword), weight in edge_counter.items()
    ]).sort_values("weight", ascending=False) if edge_counter else pd.DataFrame(columns=["source", "target", "weight"])

    edge_df.to_excel(biblio_dir / "05_author_keyword_bipartite_network.xlsx", index=False)
    vos_dir = output_dirs(results_dir)["common"] / "VOSviewer_export"
    vos_dir.mkdir(parents=True, exist_ok=True)
    export_vosviewer_map_network(edge_df, vos_dir / "author_keyword_network", min_weight=2, max_edges=500, thematic=True)
    export_network_cluster_tables(
        edge_df,
        biblio_dir / "09_author_keyword_network_clusters_weighted.xlsx",
        max_edges=90,
        min_weight=2,
        thematic=True,
    )
    remove_obsolete_network_images(img_dir, [
        "16_author_keyword_network.png",
        "16_author_keyword_network.html",
        "16_author_keyword_alluvial_theme_connections.html",
        "16_author_keyword_alluvial_theme_connections.png",
    ])
    create_readable_network_representations(
        edge_df,
        img_dir,
        "16_author_keyword",
        "Author-Keyword Links",
        min_weight=2,
        thematic=True,
    )
    return edge_df


def create_advanced_bibliometric_networks(df: pd.DataFrame, results_dir: Path):
    dirs = output_dirs(results_dir)
    biblio_dir = dirs["bibliometric"]
    img_dir = biblio_dir / "IMAGES_BIBLIOMETRICAS"
    img_dir.mkdir(parents=True, exist_ok=True)

    keyword_lists = []
    for _, row in df.iterrows():
        keywords = []
        if "KEYWORDS" in df.columns and pd.notna(row.get("KEYWORDS", "")):
            keywords.extend(lda.split_keywords(row.get("KEYWORDS", "")))
        if not keywords:
            title_tokens = tokenize_for_vocabulary(row.get("TITLE", ""))
            keywords.extend([token for token in title_tokens if token in SEARCH_OBJECTIVE_KEEP_TERMS])
        keyword_lists.append(normalize_keyword_terms(keywords))

    author_lists = []
    if "AUTHOR" in df.columns:
        for author_text in df["AUTHOR"].fillna(""):
            author_lists.append(lda.split_author_names(author_text))

    affiliation_rows = []
    country_lists = []
    institution_lists = []
    for idx, row in df.iterrows():
        aff_text = record_affiliation_text(row)
        countries = extract_countries(aff_text)
        institutions = extract_institutions(aff_text)
        country_lists.append(countries)
        institution_lists.append(institutions)
        if countries or institutions:
            affiliation_rows.append({
                "record_index": idx,
                "title": row.get("TITLE", ""),
                "countries": "; ".join(countries),
                "institutions": "; ".join(institutions),
                "affiliation_text": aff_text,
            })

    keyword_edges = build_pair_counts(keyword_lists, max_items_per_record=8)
    author_edges = build_pair_counts(author_lists, max_items_per_record=8)
    country_edges = build_pair_counts(country_lists, max_items_per_record=6)
    institution_edges = build_pair_counts(institution_lists, max_items_per_record=6)
    vos_dir = output_dirs(results_dir)["common"] / "VOSviewer_export"
    vos_dir.mkdir(parents=True, exist_ok=True)
    export_vosviewer_map_network(keyword_edges, vos_dir / "keyword_cooccurrence", min_weight=4, max_edges=500, thematic=True)
    export_vosviewer_map_network(author_edges, vos_dir / "author_collaboration", min_weight=2, max_edges=500, thematic=False)
    export_vosviewer_map_network(country_edges, vos_dir / "country_collaboration", min_weight=1, max_edges=300, thematic=False)
    export_vosviewer_map_network(institution_edges, vos_dir / "institution_network", min_weight=1, max_edges=300, thematic=False)
    keyword_corr = create_keyword_correlation_heatmap(
        keyword_lists,
        img_dir / "08_keyword_correlation_matrix.png",
        top_n=25,
    )
    create_variable_correlation_outputs(df, keyword_lists, country_lists, results_dir)
    create_three_field_sankey(df, keyword_lists, img_dir / "15_three_field_map_sources_keywords_years.html")
    create_creative_bibliometric_html(df, keyword_lists, author_lists, results_dir)
    country_production_df, country_collaboration_df = create_country_collaboration_outputs(country_lists, results_dir)
    author_keyword_edges = create_author_keyword_bipartite(author_lists, keyword_lists, results_dir)

    country_counter = Counter()
    country_institution_counter = Counter()
    for countries, institutions in zip(country_lists, institution_lists):
        for country in countries:
            country_counter[country] += 1
            country_institution_counter[country] += len(institutions)
    country_counts = pd.DataFrame([
        {
            "Country": country,
            "Documents": documents,
            "Institutions": country_institution_counter.get(country, 0),
        }
        for country, documents in country_counter.items()
    ]).sort_values("Documents", ascending=False) if country_counter else pd.DataFrame(columns=["Country", "Documents", "Institutions"])

    with pd.ExcelWriter(biblio_dir / "02_bibliometric_networks_and_correlations.xlsx") as writer:
        keyword_edges.to_excel(writer, sheet_name="Keyword_Cooccurrence", index=False)
        author_edges.to_excel(writer, sheet_name="Author_Collaboration", index=False)
        country_edges.to_excel(writer, sheet_name="Country_Collaboration", index=False)
        institution_edges.to_excel(writer, sheet_name="Institution_Network", index=False)
        pd.DataFrame(affiliation_rows).to_excel(writer, sheet_name="Affiliations_Extracted", index=False)
        country_counts.to_excel(writer, sheet_name="Country_Counts", index=False)
        country_production_df.to_excel(writer, sheet_name="Country_SCP_MCP", index=False)
        country_collaboration_df.to_excel(writer, sheet_name="Country_Collab_Edges", index=False)
        author_keyword_edges.to_excel(writer, sheet_name="Author_Keyword_Edges", index=False)
        if not keyword_corr.empty:
            keyword_corr.to_excel(writer, sheet_name="Keyword_Correlation")

    export_network_cluster_tables(
        keyword_edges,
        biblio_dir / "07_keyword_network_clusters_weighted.xlsx",
        max_edges=70,
        min_weight=4,
        thematic=True,
    )
    export_network_cluster_tables(
        author_edges,
        biblio_dir / "08_author_network_clusters_weighted.xlsx",
        max_edges=70,
        min_weight=2,
        thematic=False,
    )
    remove_obsolete_network_images(img_dir, [
        "05_keyword_cooccurrence_network.png",
        "05_keyword_cooccurrence_network.html",
        "05_keywords_alluvial_theme_connections.html",
        "05_keywords_alluvial_theme_connections.png",
    ])
    create_readable_network_representations(
        keyword_edges,
        img_dir,
        "05_keywords",
        "Keyword Co-occurrence",
        min_weight=4,
        thematic=True,
    )
    remove_obsolete_network_images(img_dir, [
        "06_author_collaboration_network.png",
        "06_author_collaboration_network.html",
        "06_authors_alluvial_theme_connections.html",
        "06_authors_alluvial_theme_connections.png",
    ])
    create_readable_network_representations(
        author_edges,
        img_dir,
        "06_authors",
        "Author Collaboration",
        min_weight=2,
        thematic=False,
    )
    plot_network(country_edges, img_dir / "07_country_collaboration_network.png", "Country Collaboration Network")
    plot_network(institution_edges, img_dir / "09_institution_interconnection_network.png", "Institution Interconnection Network", max_edges=70, max_labels=24, min_weight=2)

    if not country_counts.empty:
        if MATPLOTLIB_AVAILABLE:
            plot_df = country_counts.head(20).sort_values("Documents", ascending=True)
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.barh(plot_df["Country"], plot_df["Documents"], color="#B07AA1", edgecolor="black")
            ax.set_title("Documents by Country Extracted from Affiliations", fontweight="bold")
            ax.set_xlabel("Documents")
            plt.tight_layout()
            plt.savefig(img_dir / "10_documents_by_affiliation_country.png", dpi=200, bbox_inches="tight")
            plt.close()
        create_world_map(country_counts, img_dir / "11_mapa_mundi_instituicoes.html")
    else:
        pd.DataFrame([{
            "message": "No affiliation/country metadata was found in the available BibTeX fields. World map could not be generated."
        }]).to_excel(biblio_dir / "02_world_map_not_generated.xlsx", index=False)


def clean_ris_value(value) -> str:
    value = "" if value is None or pd.isna(value) else str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace("\n", " ").replace("\r", " ")


def export_ris_for_vosviewer(df: pd.DataFrame, output_path: Path):
    """Exports a generic RIS file, which VOSviewer can read as a reference manager file."""
    lines = []
    for _, row in df.iterrows():
        lines.append("TY  - JOUR")
        title = clean_ris_value(row.get("TITLE", row.get("title", "")))
        if title:
            lines.append(f"TI  - {title}")
        year = clean_ris_value(row.get("YEAR", row.get("year", "")))
        if year:
            lines.append(f"PY  - {year}")
        journal = clean_ris_value(row.get("JOURNAL", row.get("journal", "")))
        if journal:
            lines.append(f"JO  - {journal}")
            lines.append(f"T2  - {journal}")
        doi = clean_ris_value(row.get("DOI", row.get("doi", "")))
        if doi:
            lines.append(f"DO  - {doi}")
        url = clean_ris_value(row.get("URL", row.get("url", "")))
        if url:
            lines.append(f"UR  - {url}")
        abstract = clean_ris_value(row.get("ABSTRACT", row.get("abstract", "")))
        if abstract:
            lines.append(f"AB  - {abstract}")
        keywords = row.get("KEYWORDS", row.get("keywords", ""))
        for keyword in lda.split_keywords(keywords):
            lines.append(f"KW  - {clean_ris_value(keyword)}")
        authors = lda.split_author_names(row.get("AUTHOR", row.get("author", "")))
        for author in authors:
            lines.append(f"AU  - {clean_ris_value(author)}")
        lines.append("ER  -")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_json_for_vosviewer(df: pd.DataFrame, output_path: Path):
    """
    Exports the same bibliographic corpus as structured JSON.

    VOSviewer itself primarily reads RIS, CSV, bibliographic databases, and its
    own map/network files. The JSON file is included as an interoperable backup
    for scripts, dashboards, and manual inspection of the exact corpus used to
    generate the VOSviewer-ready exports.
    """
    records = []
    for idx, row in df.reset_index(drop=True).iterrows():
        title = clean_ris_value(row.get("TITLE", row.get("title", "")))
        year = clean_ris_value(row.get("YEAR", row.get("year", "")))
        journal = clean_ris_value(row.get("JOURNAL", row.get("journal", "")))
        doi = clean_ris_value(row.get("DOI", row.get("doi", "")))
        url = clean_ris_value(row.get("URL", row.get("url", "")))
        abstract = clean_ris_value(row.get("ABSTRACT", row.get("abstract", "")))
        source_file = clean_ris_value(row.get("source_file", row.get("SOURCE_FILE", "")))
        authors = [
            clean_ris_value(author)
            for author in lda.split_author_names(row.get("AUTHOR", row.get("author", "")))
            if clean_ris_value(author)
        ]
        keywords = [
            clean_ris_value(keyword)
            for keyword in lda.split_keywords(row.get("KEYWORDS", row.get("keywords", "")))
            if clean_ris_value(keyword)
        ]
        raw_fields = {}
        for col, value in row.items():
            cleaned = clean_ris_value(value)
            if cleaned:
                raw_fields[str(col)] = cleaned
        records.append({
            "id": idx + 1,
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "doi": doi,
            "url": url,
            "abstract": abstract,
            "keywords": keywords,
            "source_file": source_file,
            "raw_fields": raw_fields,
        })
    payload = {
        "metadata": {
            "description": "Bibliographic corpus exported from BibTeX files after deduplication for LDA and AviTech-Bayes comparison.",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "record_count": len(records),
            "intended_use": "Companion JSON for VOSviewer-ready RIS, corpus, map, and network exports.",
        },
        "records": records,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_vosviewer_corpus_scores(df: pd.DataFrame, vos_dir: Path):
    """
    Exports VOSviewer corpus/scores/thesaurus files.

    According to the VOSviewer manual, a corpus file is plain text with one
    document per line, usually title + abstract. A scores file may be supplied
    with one score row for each corpus line. A thesaurus file has two columns:
    label and replace by.
    """
    corpus_lines = []
    score_rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        title = clean_ris_value(row.get("TITLE", ""))
        abstract = clean_ris_value(row.get("ABSTRACT", ""))
        text = f"{title}. {abstract}".strip()
        if not text:
            text = title or abstract or f"document_{idx + 1}"
        corpus_lines.append(text)

        year_raw = clean_ris_value(row.get("YEAR", ""))
        year_match = re.search(r"\d{4}", year_raw)
        year = int(year_match.group(0)) if year_match else np.nan
        title_tokens = len(tokenize_for_vocabulary(title))
        abstract_tokens = len(tokenize_for_vocabulary(abstract))
        objective_hits = 0
        all_tokens = set(tokenize_for_vocabulary(f"{title} {abstract}"))
        for terms in OBJECTIVE_GROUPS.values():
            objective_hits += len(all_tokens & terms)
        score_rows.append({
            "score<Publication year>": year,
            "score<Title tokens>": title_tokens,
            "score<Abstract tokens>": abstract_tokens,
            "score<Objective search hits>": objective_hits,
        })

    (vos_dir / "corpus_title_abstract.txt").write_text("\n".join(corpus_lines), encoding="utf-8")
    pd.DataFrame(score_rows).to_csv(vos_dir / "scores_title_abstract.txt", sep="\t", index=False, encoding="utf-8")

    thesaurus_rows = [
        {"label": "precision livestock farming", "replace by": "precision_livestock_farming"},
        {"label": "digital transformation", "replace by": "digital_transformation"},
        {"label": "return on investment", "replace by": "return_on_investment"},
        {"label": "economic feasibility", "replace by": "economic_feasibility"},
        {"label": "computer vision", "replace by": "computer_vision"},
        {"label": "machine learning", "replace by": "machine_learning"},
        {"label": "deep learning", "replace by": "deep_learning"},
        {"label": "artificial intelligence", "replace by": "artificial_intelligence"},
        {"label": "supply chain", "replace by": "supply_chain"},
        {"label": "value chain", "replace by": "value_chain"},
        {"label": "food chain", "replace by": "food_chain"},
        {"label": "production chain", "replace by": "production_chain"},
        {"label": "chain", "replace by": ""},
    ]
    pd.DataFrame(thesaurus_rows).to_csv(vos_dir / "thesaurus_terms.txt", sep="\t", index=False, encoding="utf-8")


def export_vosviewer_map_network(edge_df: pd.DataFrame, output_prefix: Path, min_weight=2, max_edges=300, thematic=False):
    """Exports VOSviewer map/network files for 'Create a map based on network data'."""
    if edge_df.empty:
        return None
    try:
        import networkx as nx
    except ImportError:
        return None

    filtered = filter_weighted_edges(edge_df, max_edges=max_edges, min_weight=min_weight)
    graph = nx.Graph()
    for _, row in filtered.iterrows():
        graph.add_edge(str(row["source"]), str(row["target"]), weight=float(row["weight"]))
    if graph.number_of_nodes() < 2:
        return None

    communities = detect_graph_communities(graph)
    cluster_ids = {cluster: idx + 1 for idx, cluster in enumerate(sorted(set(communities.values())))}
    degree = dict(graph.degree(weight="weight"))
    occurrence = Counter()
    for _, row in filtered.iterrows():
        occurrence[str(row["source"])] += float(row["weight"])
        occurrence[str(row["target"])] += float(row["weight"])
    pos = nx.spring_layout(graph, k=1.2, iterations=300, seed=42, weight="weight")
    node_ids = {node: idx + 1 for idx, node in enumerate(sorted(graph.nodes()))}

    map_rows = []
    for node in sorted(graph.nodes()):
        x, y = pos[node]
        map_rows.append({
            "id": node_ids[node],
            "label": node,
            "x": round(float(x), 6),
            "y": round(float(y), 6),
            "cluster": cluster_ids.get(communities.get(node, "cluster_1"), 1),
            "weight<Occurrences>": round(float(occurrence.get(node, 0)), 6),
            "weight<Total link strength>": round(float(degree.get(node, 0)), 6),
            "score<Theme>": node_theme(node) if thematic else "",
        })
    map_df = pd.DataFrame(map_rows)

    network_rows = []
    for source, target, data in graph.edges(data=True):
        network_rows.append({
            "source": node_ids[source],
            "target": node_ids[target],
            "weight": round(float(data.get("weight", 1)), 6),
        })
    network_df = pd.DataFrame(network_rows)

    map_path = output_prefix.with_name(output_prefix.name + "_map.txt")
    network_path = output_prefix.with_name(output_prefix.name + "_network.txt")
    map_df.to_csv(map_path, sep="\t", index=False, encoding="utf-8")
    network_df.to_csv(network_path, sep="\t", index=False, encoding="utf-8")
    return map_path, network_path


def export_vosviewer_like_html(vos_dir: Path):
    """
    Builds publication-oriented VOSviewer-like maps from map/network files.

    The HTML keeps all item names available through hover, while the PNG labels
    only the most influential nodes to avoid overlap in article figures.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return []

    cluster_palette = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
        "#FF9DA6", "#9D755D", "#BAB0AC", "#8CD17D", "#B6992D", "#499894",
    ]

    def cluster_color(cluster):
        try:
            idx = int(cluster) - 1
        except Exception:
            idx = 0
        if idx < 0:
            return "#B0B0B0"
        return cluster_palette[idx % len(cluster_palette)]

    def scale_values(values, low, high):
        values = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
        if float(values.max()) > float(values.min()):
            return low + (high - low) * (values - values.min()) / (values.max() - values.min())
        return pd.Series([low + (high - low) / 2] * len(values), index=values.index)

    def make_publication_png(map_df, network_df, png_path, title, label_limit):
        if not MATPLOTLIB_AVAILABLE:
            return False
        fig = plt.figure(figsize=(14, 10), dpi=180)
        ax = fig.add_axes([0.04, 0.08, 0.72, 0.84])
        side_ax = fig.add_axes([0.78, 0.12, 0.19, 0.76])
        side_ax.axis("off")

        node_lookup = map_df.set_index("id").to_dict("index")
        edge_weights = pd.to_numeric(network_df["weight"], errors="coerce").fillna(0).astype(float)
        max_edge = max(float(edge_weights.max()), 1.0)
        draw_edges = network_df.sort_values("weight", ascending=False).head(220)
        for _, edge in draw_edges.iterrows():
            source = node_lookup.get(int(edge["source"]))
            target = node_lookup.get(int(edge["target"]))
            if not source or not target:
                continue
            weight = float(edge["weight"])
            ax.plot(
                [source["x"], target["x"]],
                [source["y"], target["y"]],
                color="#9AA6B2",
                alpha=0.08 + 0.22 * weight / max_edge,
                linewidth=0.25 + 2.4 * weight / max_edge,
                zorder=1,
            )

        cluster_plot_col = "display_cluster" if "display_cluster" in map_df.columns else "cluster"
        for cluster in sorted(map_df[cluster_plot_col].unique()):
            cluster_df = map_df[map_df[cluster_plot_col] == cluster]
            cluster_label = "Other clusters" if int(cluster) == 0 else f"Cluster {int(cluster)}"
            ax.scatter(
                cluster_df["x"],
                cluster_df["y"],
                s=cluster_df["plot_size"],
                color=cluster_color(cluster),
                edgecolor="white",
                linewidth=0.8,
                alpha=0.92,
                label=cluster_label,
                zorder=3,
            )

        labeled = map_df.sort_values("label_weight", ascending=False).head(label_limit).reset_index(drop=True)
        for marker_id, (_, node) in enumerate(labeled.iterrows(), 1):
            ax.text(
                node["x"],
                node["y"],
                str(marker_id),
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#111111",
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="#263238", lw=0.6, alpha=0.92),
                zorder=5,
            )

        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)

        cluster_summary = (
            map_df.groupby("cluster")
            .agg(
                items=("label", "count"),
                total_link_strength=("label_weight", "sum"),
            )
            .reset_index()
            .sort_values("total_link_strength", ascending=False)
        )
        top_terms = []
        for cluster in cluster_summary["cluster"]:
            names = (
                map_df[map_df["cluster"] == cluster]
                .sort_values("label_weight", ascending=False)["label"]
                .head(4)
                .map(lambda x: readable_node_label(x, max_chars=23))
                .tolist()
            )
            top_terms.append(", ".join(names))
        cluster_summary["representative_items"] = top_terms

        side_lines = [
            "Network summary",
            f"Items: {len(map_df)}",
            f"Links shown: {len(draw_edges)} of {len(network_df)}",
            f"Clusters: {map_df['cluster'].nunique()}",
            "",
            "Legend - main names",
            "(numbered markers on map)",
        ]
        for marker_id, (_, node) in enumerate(labeled.iterrows(), 1):
            side_lines.append(
                f"{marker_id}. {readable_node_label(node['label'], max_chars=31)}"
                f" (C{int(node['cluster'])}; TLS {float(node['label_weight']):.1f})"
            )
        side_lines.extend([
            "",
            "Main clusters",
        ])
        for _, row in cluster_summary.head(6).iterrows():
            side_lines.extend([
                f"C{int(row['cluster'])}: {int(row['items'])} items",
                f"  TLS: {row['total_link_strength']:.1f}",
                f"  {row['representative_items']}",
            ])
        side_ax.text(
            0,
            1,
            "\n".join(side_lines),
            va="top",
            ha="left",
            fontsize=8.8,
            color="#263238",
            linespacing=1.35,
        )
        ax.set_aspect("equal", adjustable="datalim")
        fig.text(
            0.04,
            0.025,
            "Node size = total link strength/occurrences; color = cluster; numbered markers identify the strongest nodes.",
            fontsize=8.5,
            color="#546E7A",
        )
        plt.savefig(png_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        return True

    generated = []
    map_files = sorted(vos_dir.glob("*_map.txt"))
    for map_path in map_files:
        prefix = map_path.name[:-8]
        network_path = vos_dir / f"{prefix}_network.txt"
        if not network_path.exists():
            continue
        try:
            map_df = pd.read_csv(map_path, sep="\t")
            network_df = pd.read_csv(network_path, sep="\t")
        except Exception:
            continue
        required_map = {"id", "label", "x", "y", "cluster"}
        required_network = {"source", "target", "weight"}
        if not required_map.issubset(map_df.columns) or not required_network.issubset(network_df.columns):
            continue
        if map_df.empty or network_df.empty:
            continue

        map_df = map_df.copy()
        network_df = network_df.copy()
        map_df["id"] = map_df["id"].astype(int)
        map_df["cluster"] = map_df["cluster"].astype(int)
        network_df["source"] = network_df["source"].astype(int)
        network_df["target"] = network_df["target"].astype(int)
        network_df["weight"] = pd.to_numeric(network_df["weight"], errors="coerce").fillna(0).astype(float)

        weight_col = "weight<Total link strength>" if "weight<Total link strength>" in map_df.columns else None
        occurrence_col = "weight<Occurrences>" if "weight<Occurrences>" in map_df.columns else weight_col
        if weight_col:
            map_df["label_weight"] = pd.to_numeric(map_df[weight_col], errors="coerce").fillna(0).astype(float)
        elif occurrence_col:
            map_df["label_weight"] = pd.to_numeric(map_df[occurrence_col], errors="coerce").fillna(0).astype(float)
        else:
            map_df["label_weight"] = 1.0
        map_df["plot_size"] = scale_values(map_df["label_weight"], 42, 420)
        map_df["interactive_size"] = scale_values(map_df["label_weight"], 10, 36)
        map_df["label_rank"] = map_df["label_weight"].rank(method="first", ascending=False)
        if prefix in {"author_collaboration", "keyword_cooccurrence"}:
            label_limit = min(14, max(10, int(len(map_df) * 0.08)))
        else:
            label_limit = min(10, max(5, int(len(map_df) * 0.06)))
        cluster_strength = map_df.groupby("cluster")["label_weight"].sum().sort_values(ascending=False)
        if len(cluster_strength) > 14:
            top_display_clusters = set(cluster_strength.head(10).index.tolist())
            map_df["display_cluster"] = map_df["cluster"].where(map_df["cluster"].isin(top_display_clusters), 0)
        else:
            map_df["display_cluster"] = map_df["cluster"]

        node_lookup = map_df.set_index("id").to_dict("index")
        edge_x, edge_y = [], []
        top_edges = network_df.sort_values("weight", ascending=False).head(320)
        for _, edge in top_edges.iterrows():
            source = node_lookup.get(int(edge["source"]))
            target = node_lookup.get(int(edge["target"]))
            if not source or not target:
                continue
            edge_x += [source["x"], target["x"], None]
            edge_y += [source["y"], target["y"], None]

        map_df["hover_text"] = map_df.apply(
            lambda node: "<br>".join([
                f"<b>{clean_ris_value(node['label'])}</b>",
                f"Cluster: {int(node['cluster'])}",
                f"Occurrences: {node.get('weight<Occurrences>', '')}",
                f"Total link strength: {node.get('weight<Total link strength>', '')}",
                f"Theme: {node.get('score<Theme>', '')}",
            ]),
            axis=1,
        )

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.65, color="rgba(120,128,140,0.20)"),
            hoverinfo="skip",
            showlegend=False,
            name="Links",
        ))
        for cluster in sorted(map_df["display_cluster"].unique()):
            cluster_df = map_df[map_df["display_cluster"] == cluster]
            cluster_label = "Other clusters" if int(cluster) == 0 else f"Cluster {int(cluster)}"
            fig.add_trace(go.Scattergl(
                x=cluster_df["x"],
                y=cluster_df["y"],
                mode="markers",
                marker=dict(
                    size=cluster_df["interactive_size"],
                    color=cluster_color(cluster),
                    line=dict(width=0.9, color="white"),
                    opacity=0.9,
                ),
                customdata=cluster_df["hover_text"],
                hovertemplate="%{customdata}<extra></extra>",
                name=cluster_label,
            ))

        annotations = [dict(
            text="Hover over nodes for complete names and metrics. Publication PNG uses numbered markers to avoid overlap.",
            x=0.01,
            y=0.01,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=12, color="#546E7A"),
        )]

        title = f"VOSviewer-like clustered map - {prefix.replace('_', ' ')}"
        fig.update_layout(
            title=title,
            template="plotly_white",
            width=1500,
            height=980,
            margin=dict(l=20, r=20, t=75, b=25),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            annotations=annotations,
        )

        html_path = vos_dir / f"vosviewer_like_{prefix}.html"
        png_path = vos_dir / f"vosviewer_like_{prefix}_PUBLICATION.png"
        nodes_path = vos_dir / f"vosviewer_like_{prefix}_nodes.xlsx"
        edges_path = vos_dir / f"vosviewer_like_{prefix}_edges.xlsx"
        clusters_path = vos_dir / f"vosviewer_like_{prefix}_cluster_summary.xlsx"
        fig.write_html(html_path, include_plotlyjs="cdn")
        make_publication_png(map_df, network_df, png_path, title, label_limit)

        cluster_summary = (
            map_df.groupby("cluster")
            .agg(
                items=("label", "count"),
                total_link_strength=("label_weight", "sum"),
                mean_link_strength=("label_weight", "mean"),
            )
            .reset_index()
            .sort_values("total_link_strength", ascending=False)
        )
        cluster_summary["top_items"] = [
            "; ".join(
                map_df[map_df["cluster"] == cluster]
                .sort_values("label_weight", ascending=False)["label"]
                .head(10)
                .map(clean_ris_value)
                .tolist()
            )
            for cluster in cluster_summary["cluster"]
        ]
        map_df.sort_values(["cluster", "label_rank"]).to_excel(nodes_path, index=False)
        network_df.sort_values("weight", ascending=False).to_excel(edges_path, index=False)
        cluster_summary.to_excel(clusters_path, index=False)
        generated.extend([html_path, png_path, nodes_path, edges_path, clusters_path])

    html_files = [path for path in generated if path.suffix.lower() == ".html"]
    png_files = [path for path in generated if path.name.endswith("_PUBLICATION.png")]
    if html_files or png_files:
        link_rows = []
        for path in html_files:
            label = path.stem.replace("vosviewer_like_", "").replace("_", " ").title()
            link_rows.append(f"<li><a href=\"{path.name}\">{label} - interactive HTML</a></li>")
        for path in png_files:
            label = path.stem.replace("vosviewer_like_", "").replace("_PUBLICATION", "").replace("_", " ").title()
            link_rows.append(f"<li><a href=\"{path.name}\">{label} - publication PNG</a></li>")
        index_path = vos_dir / "vosviewer_like_INDEX.html"
        index_path.write_text(
            "\n".join([
                "<!doctype html>",
                "<html lang=\"en\">",
                "<head><meta charset=\"utf-8\"><title>VOSviewer-like maps</title>",
                "<style>body{font-family:Arial,sans-serif;margin:36px;line-height:1.5;color:#1f2933}"
                "a{color:#0b5cad;text-decoration:none}a:hover{text-decoration:underline}"
                "li{margin:8px 0}.note{color:#546e7a;max-width:900px}</style></head>",
                "<body>",
                "<h1>VOSviewer-like clustered maps</h1>",
                "<p class=\"note\">Publication-oriented companion maps. PNG files use numbered markers and a side legend with the main names; HTML files keep all details available through hover.</p>",
                "<ul>",
                "\n".join(link_rows),
                "</ul>",
                "</body></html>",
            ]),
            encoding="utf-8",
        )
        generated.append(index_path)
    return generated


def export_vosviewer_files(df: pd.DataFrame, results_dir: Path):
    dirs = output_dirs(results_dir)
    vos_dir = dirs["common"] / "VOSviewer_export"
    vos_dir.mkdir(parents=True, exist_ok=True)
    ris_long_name = vos_dir / "corpus_topic_modeling_for_vosviewer.ris"
    ris_short_name = vos_dir / "corpus_vosviewer.ris"
    json_name = vos_dir / "corpus_vosviewer.json"
    export_ris_for_vosviewer(df, ris_long_name)
    shutil.copy2(ris_long_name, ris_short_name)
    export_json_for_vosviewer(df, json_name)
    export_vosviewer_corpus_scores(df, vos_dir)
    export_vosviewer_like_html(vos_dir)
    pd.DataFrame([
        {
            "file": "corpus_topic_modeling_for_vosviewer.ris",
            "how_to_open": "VOSviewer > Create > Create a map based on bibliographic data > Read data from reference manager files > RIS",
        },
        {
            "file": "corpus_vosviewer.ris",
            "how_to_open": "Same RIS database with a shorter file name; use this one if VOSviewer has trouble locating the longer name.",
        },
        {
            "file": "corpus_vosviewer.json",
            "how_to_open": "Structured JSON version of the same cleaned corpus; use for scripts, dashboards, checking fields, or conversion to other tools.",
        },
        {
            "file": "corpus_title_abstract.txt + scores_title_abstract.txt",
            "how_to_open": "VOSviewer > Create > Create a map based on text data > Read data from VOSviewer files > corpus file; optionally add scores file",
        },
        {
            "file": "thesaurus_terms.txt",
            "how_to_open": "Use as thesaurus file in the Create Map wizard to merge/ignore terms.",
        },
        {
            "file": "*_map.txt + *_network.txt",
            "how_to_open": "VOSviewer > Create > Create a map based on network data > Read from VOSviewer files > select map and network",
        },
        {
            "file": "vosviewer_like_*.html",
            "how_to_open": "Python-generated VOSviewer-like interactive maps with clusters, weighted nodes, and hover labels; open directly in a browser.",
        },
        {
            "file": "vosviewer_like_*_nodes.xlsx + vosviewer_like_*_edges.xlsx",
            "how_to_open": "Companion node/edge tables for interpreting the clusters and strongest links in each interactive map.",
        },
    ]).to_excel(vos_dir / "README_VOSviewer.xlsx", index=False)
    return vos_dir


def preprocess_for_topic_models(df: pd.DataFrame, results_dir: Path, create_images: bool = True):
    dirs = output_dirs(results_dir)
    common_dir = dirs["common"]
    if "ABSTRACT" not in df.columns:
        raise KeyError("ABSTRACT column was not found in the BibTeX files.")

    print("\n=== TEXT PREPROCESSING FOR LDA + AVITECH-BAYES ===")
    abstracts = df["ABSTRACT"].fillna("").tolist()
    processed_tokens, valid_indices = lda.preprocess_abstracts(abstracts)
    processed_df = df.iloc[valid_indices].reset_index(drop=True)

    if len(processed_tokens) < 2:
        raise RuntimeError("Not enough valid documents after preprocessing.")

    all_tokens = [token for doc in processed_tokens for token in doc]
    word_freq = pd.DataFrame(Counter(all_tokens).items(), columns=["word", "frequency"])
    word_freq = word_freq.sort_values("frequency", ascending=False)
    word_freq.to_excel(common_dir / "01_word_frequency_topic_model_corpus.xlsx", index=False)

    if create_images:
        article_img_dir = common_dir / "COMMON_IMAGES"
        article_img_dir.mkdir(parents=True, exist_ok=True)

        lda.create_round_wordcloud(word_freq, str(common_dir / "01_round_word_cloud_topic_model_corpus.png"))
        lda.create_round_wordcloud(word_freq, str(article_img_dir / "01_round_word_cloud.png"))
        bnmf.create_round_wordcloud(
            word_freq,
            article_img_dir / "01_article_ready_round_word_cloud.png",
            title="Word Cloud - LDA + AviTech-Bayes Corpus",
            dpi=300,
            size=2400,
        )
        remove_obsolete_network_images(article_img_dir, [
            "01_publication_word_cloud_rectangular.png",
            "01_word_cloud_round.png",
            "01_publication_word_cloud_round.png",
        ])

    print("\n=== CREATING SHARED DTM ===")
    dtm, vectorizer, feature_names = lda.create_dtm_simple(processed_tokens)
    return processed_df, processed_tokens, word_freq, dtm, vectorizer, feature_names


def run_lda_pipeline(processed_df, processed_tokens, word_freq, dtm, vectorizer, feature_names, results_dir: Path):
    print("\n=== LDA PIPELINE ===")
    dirs = output_dirs(results_dir)
    lda_dir = dirs["lda"]
    max_topics = min(lda.config.max_topics_cap, len(processed_tokens) // lda.config.min_docs_per_topic)
    if max_topics < 3:
        max_topics = 3 if len(processed_tokens) >= lda.config.min_docs_per_topic * 3 else 2
        if max_topics == 2:
            print("Warning: too few documents to safely avoid K=2. Using K=2 as fallback.")

    best_k, k_metrics, premise_applied = select_optimal_k_fast(dtm, feature_names, max_topics)
    if not k_metrics.empty:
        k_metrics.to_excel(lda_dir / "02_lda_k_metrics.xlsx", index=False)
        lda_k_plot = lda_dir / "02_lda_topic_evaluation.png"
        lda.create_topic_evaluation_plot(k_metrics, best_k, str(lda_k_plot))
        article_img_dir = lda_dir / "IMAGES_LDA"
        article_img_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lda_k_plot, article_img_dir / "02_lda_optimal_k.png")

    lda_model, doc_topic = run_lda_simple_quiet(dtm, best_k)
    topic_words = lda.get_top_words(lda_model, feature_names, lda.config.n_words_per_topic)
    df_priority = lda.calculate_topic_priority(doc_topic, topic_words, dtm, feature_names)
    topic_labels = lda.generate_topic_labels(topic_words, df_priority)

    dominant_topic = np.argmax(doc_topic, axis=1) + 1
    topic_prob = np.max(doc_topic, axis=1)

    results_df = pd.DataFrame(
        {
            "title": processed_df["TITLE"].values,
            "author": processed_df["AUTHOR"].values if "AUTHOR" in processed_df.columns else "",
            "year": processed_df["YEAR"].values if "YEAR" in processed_df.columns else "",
            "journal": processed_df["JOURNAL"].values if "JOURNAL" in processed_df.columns else "",
            "doi": processed_df["DOI"].values if "DOI" in processed_df.columns else "",
            "keywords": processed_df["KEYWORDS"].values if "KEYWORDS" in processed_df.columns else "",
            "publisher": processed_df["PUBLISHER"].values if "PUBLISHER" in processed_df.columns else "",
            "abstract": processed_df["ABSTRACT"].values,
            "main_topic": dominant_topic,
            "topic_label": [topic_labels.get(t, f"Topic {t}") for t in dominant_topic],
            "topic_probability": topic_prob,
        }
    )

    lda_workbook = lda_dir / "03_lda_results_topic_model_corpus.xlsx"
    legacy_lda_workbook = lda_dir / "03_lda_results_without_hoshin_kanri.xlsx"
    with pd.ExcelWriter(lda_workbook) as writer:
        results_df.to_excel(writer, sheet_name="Documents_with_LDA_Topics", index=False)
        pd.DataFrame(
            [(k, ", ".join(v)) for k, v in topic_words.items()],
            columns=["Topic", "Top_Words"],
        ).to_excel(writer, sheet_name="LDA_Words", index=False)
        df_priority.to_excel(writer, sheet_name="LDA_Topic_Priority", index=False)
        word_freq.head(1000).to_excel(writer, sheet_name="Word_Frequency", index=False)
    shutil.copy2(lda_workbook, legacy_lda_workbook)

    lda.create_topic_priority_chart(df_priority, str(lda_dir / "03_lda_topic_priority_chart.png"))
    lda.create_topic_distribution_pie(df_priority, str(lda_dir / "03_lda_topic_distribution_pie.png"))
    lda.create_priority_summary_chart(df_priority, str(lda_dir / "03_lda_priority_summary.png"))
    lda.create_topic_correlation_heatmap(doc_topic, str(lda_dir / "03_lda_topic_correlation.png"))
    lda.create_pyldavis(lda_model, dtm, vectorizer, str(lda_dir / "03_lda_pyldavis_visualization.html"))
    previous_lda_results_dir = lda.config.results_dir
    lda.config.results_dir = lda_dir
    try:
        lda.generate_report(df_priority, topic_labels, best_k, premise_applied, word_freq, processed_tokens, k_metrics)
    finally:
        lda.config.results_dir = previous_lda_results_dir

    return {
        "best_k": best_k,
        "model": lda_model,
        "doc_topic": doc_topic,
        "topic_words": topic_words,
        "feature_names": feature_names,
        "priority": df_priority,
        "labels": topic_labels,
        "results_df": results_df,
        "premise": premise_applied,
    }


def load_existing_lda_result(results_dir: Path, feature_names):
    dirs = output_dirs(results_dir)
    lda_path = dirs["lda"] / "03_lda_results_topic_model_corpus.xlsx"
    legacy_lda_path = dirs["lda"] / "03_lda_results_without_hoshin_kanri.xlsx"
    metrics_path = dirs["lda"] / "02_lda_k_metrics.xlsx"
    if not lda_path.exists() and legacy_lda_path.exists():
        lda_path = legacy_lda_path
    if not lda_path.exists():
        raise FileNotFoundError(f"LDA result workbook not found: {lda_path}")

    results_df = pd.read_excel(lda_path, sheet_name="Documents_with_LDA_Topics")
    words_df = pd.read_excel(lda_path, sheet_name="LDA_Words")
    priority = pd.read_excel(lda_path, sheet_name="LDA_Topic_Priority")
    topic_words = {}
    for _, row in words_df.iterrows():
        topic_raw = str(row["Topic"])
        match = re.search(r"(\d+)", topic_raw)
        topic_key = f"Topic_{match.group(1)}" if match else topic_raw
        topic_words[topic_key] = [word.strip() for word in str(row["Top_Words"]).split(",") if word.strip()]

    if metrics_path.exists():
        metrics = pd.read_excel(metrics_path)
        score_col = "Composite_Score" if "Composite_Score" in metrics.columns else metrics.columns[-1]
        best_k = int(metrics.sort_values(score_col, ascending=False).iloc[0]["K"])
    else:
        best_k = int(results_df["main_topic"].max())

    labels = {}
    if "Topic" in priority.columns and "Label" in priority.columns:
        labels = dict(zip(priority["Topic"], priority["Label"]))

    return {
        "best_k": best_k,
        "model": None,
        "doc_topic": None,
        "topic_words": topic_words,
        "feature_names": feature_names,
        "priority": priority,
        "labels": labels,
        "results_df": results_df,
        "premise": "Loaded from existing LDA workbook.",
    }


def create_pyldavis_from_distributions(topic_term, doc_topic, dtm, feature_names, output_html: Path, title: str):
    """
    Creates a pyLDAvis HTML from generic topic distributions.

    For LDA this is native. For AviTech-Bayes, this is a pyLDAvis-style
    projection: W is used as document-topic distribution and H as topic-term
    distribution. The model remains BNMF/Bayesian posterior NMF; pyLDAvis is
    used only as an interactive topic exploration interface.
    """
    try:
        import pyLDAvis
    except ImportError:
        print(f"pyLDAvis not installed; skipped {title}.")
        return False

    topic_term = np.asarray(topic_term, dtype=float)
    doc_topic = np.asarray(doc_topic, dtype=float)
    topic_term = np.maximum(topic_term, 1e-12)
    doc_topic = np.maximum(doc_topic, 1e-12)
    topic_term = topic_term / (topic_term.sum(axis=1, keepdims=True) + 1e-12)
    doc_topic = doc_topic / (doc_topic.sum(axis=1, keepdims=True) + 1e-12)

    try:
        from scipy import sparse
        if sparse.issparse(dtm):
            doc_lengths = np.asarray(dtm.sum(axis=1)).ravel()
            term_frequency = np.asarray(dtm.sum(axis=0)).ravel()
        else:
            doc_lengths = np.asarray(dtm).sum(axis=1)
            term_frequency = np.asarray(dtm).sum(axis=0)
    except Exception:
        doc_lengths = np.asarray(dtm.sum(axis=1)).ravel()
        term_frequency = np.asarray(dtm.sum(axis=0)).ravel()

    doc_lengths = np.maximum(np.asarray(doc_lengths, dtype=float), 1.0)
    term_frequency = np.maximum(np.asarray(term_frequency, dtype=float), 1.0)
    vocab = [str(term) for term in feature_names]

    try:
        prepared = pyLDAvis.prepare(
            topic_term_dists=topic_term,
            doc_topic_dists=doc_topic,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency,
            sort_topics=False,
        )
        pyLDAvis.save_html(prepared, str(output_html))
        output_html.write_text(
            output_html.read_text(encoding="utf-8").replace(
                "<body>",
                f"<body><div style='font-family:Arial,sans-serif;padding:10px 14px;background:#f7f7f7;border-bottom:1px solid #ddd'>"
                f"<b>{clean_ris_value(title)}</b><br>"
                f"<span style='font-size:12px;color:#555'>For AviTech-Bayes, pyLDAvis is used as an exploratory interface over W/H distributions, not as evidence that the model is LDA.</span>"
                f"</div>",
                1,
            ),
            encoding="utf-8",
        )
        print(f"pyLDAvis-style visualization saved: {output_html}")
        return True
    except Exception as exc:
        print(f"Could not create pyLDAvis visualization for {title}: {exc}")
        return False


def build_avitech_bayes_matrix(dtm, feature_names):
    """
    Builds the optimized matrix used by AviTech-Bayes.

    LDA keeps using the raw count DTM because that is the standard input for
    count-based topic modeling. AviTech-Bayes uses a different representation:
        1. TF-IDF weighting to reduce the dominance of very frequent generic
           words.
        2. Extra weights for terms directly tied to the search string:
           poultry/broiler/layer, PLF/digital transformation, feasibility/ROI/cost.
        3. Down-weighting of generic terms that tend to make BNMF topics less
           exclusive.
        4. L2 row normalization to make document-topic assignments sharper and
           more comparable across abstracts of different lengths.

    This is intended to improve the new method without changing the corpus or
    deleting evidence. It makes the BNMF representation more aligned with the
    article objective.
    """
    try:
        from scipy import sparse
    except ImportError:
        sparse = None

    x = dtm.astype(float).copy()
    n_docs = x.shape[0]
    if sparse is not None and sparse.issparse(x):
        df = np.asarray((x > 0).sum(axis=0)).ravel()
        idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        x = x.multiply(idf)
    else:
        df = np.count_nonzero(x > 0, axis=0)
        idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        x = x * idf

    objective_terms = set().union(*OBJECTIVE_GROUPS.values()) | SEARCH_OBJECTIVE_KEEP_TERMS
    feature_weights = np.ones(len(feature_names), dtype=float)
    for idx, feature in enumerate(feature_names):
        word = str(feature).lower()
        if word in GENERIC_EXCLUDE_TERMS:
            feature_weights[idx] *= 0.25
        if word in objective_terms:
            feature_weights[idx] *= 2.60
        if "_" in word:
            feature_weights[idx] *= 1.45
        if word in OBJECTIVE_GROUPS["economic"]:
            feature_weights[idx] *= 1.25
        if word in OBJECTIVE_GROUPS["technology"]:
            feature_weights[idx] *= 1.15
        if word in OBJECTIVE_GROUPS["poultry"]:
            feature_weights[idx] *= 1.15

    if sparse is not None and sparse.issparse(x):
        x = x.multiply(feature_weights)
    else:
        x = x * feature_weights
    x = normalize(x, norm="l2", axis=1, copy=False)
    return x, pd.DataFrame({"term": feature_names, "avitech_weight": feature_weights})


class FastAviTechBayesModel:
    """
    Fast implementation of AviTech-Bayes.

    The numerical factorization is estimated with an optimized NMF solver from
    scikit-learn so the script can run on a large BibTeX corpus. The Bayesian
    part is not the optimizer itself; it is the posterior layer estimated after
    W and H are obtained. This is equivalent to using NMF as the non-negative
    latent structure and then estimating Bayesian uncertainty around that
    structure.

    Why it is Bayesian:
        1. W and H are treated as positive latent intensities.
        2. Gamma posterior samples are drawn around W and H.
        3. The samples are normalized to document-topic and topic-word
           distributions.
        4. The model reports posterior mean, standard deviation, credibility
           intervals, and ARD inactive-topic probability.

    This is a Bayesian posterior layer over NMF, hence the article-friendly
    name AviTech-Bayes.
    """
    def __init__(self, n_components, max_iter=180, tol=1e-4, posterior_samples=60, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.posterior_samples = posterior_samples
        self.posterior_concentration = float(getattr(bnmf.Config, "BAYESIAN_POSTERIOR_CONCENTRATION", 120.0))
        self.random_state = random_state
        self.W = None
        self.H = None
        self.raw_W = None
        self.raw_H = None
        self.reconstruction_error_ = []
        self.log_posterior_ = []
        self.posterior_summary_ = None
        self.model = None

    def fit(self, X):
        # Step 1: deterministic non-negative matrix factorization.
        # X ~= raw_W x raw_H, where raw_W represents document-topic intensity
        # and raw_H represents topic-word intensity.
        self.model = SklearnNMF(
            n_components=self.n_components,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self.raw_W = self.model.fit_transform(X)
        self.raw_H = self.model.components_

        # Step 2: normalize W and H so they can be interpreted like
        # probability distributions. W rows sum to 1 for each document, and H
        # rows sum to 1 for each topic.
        self.W = self.raw_W / (self.raw_W.sum(axis=1, keepdims=True) + 1e-10)
        self.H = self.raw_H / (self.raw_H.sum(axis=1, keepdims=True) + 1e-10)
        denom = max(1, X.shape[0] * X.shape[1])
        rec = float((self.model.reconstruction_err_ ** 2) / denom)
        self.reconstruction_error_ = [rec]
        self.log_posterior_ = [-rec]
        return self

    def get_topic_assignments(self):
        return np.argmax(self.W, axis=1) + 1

    def get_top_words(self, terms, n_words=15):
        terms = list(terms)
        return [[terms[i] for i in topic.argsort()[-n_words:][::-1]] for topic in self.H]

    def estimate_posterior_uncertainty(self, random_state=None):
        # Step 3: Bayesian posterior approximation.
        # The Gamma distribution is appropriate here because W and H are
        # strictly non-negative. Around each estimated positive weight, the
        # sampler creates plausible alternative W/H matrices. These samples
        # express uncertainty in the topic structure.
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        eps = 1e-10
        n_samples = int(max(1, self.posterior_samples))
        concentration = max(float(self.posterior_concentration), 1.0)
        topic_activity_samples = []
        doc_max_prob_samples = []
        h_mean_accum = np.zeros_like(self.H)
        h_sq_accum = np.zeros_like(self.H)
        w_mean_accum = np.zeros_like(self.W)
        w_sq_accum = np.zeros_like(self.W)

        for _ in range(n_samples):
            # Gamma sampling around normalized W and H. The concentration
            # controls how tightly the posterior samples stay around the fitted
            # NMF solution. Higher concentration = lower posterior dispersion.
            w_sample = rng.gamma(shape=np.maximum(self.W * concentration, eps), scale=1.0 / concentration)
            h_sample = rng.gamma(shape=np.maximum(self.H * concentration, eps), scale=1.0 / concentration)

            # Re-normalize each posterior draw so the samples remain
            # interpretable as probabilities.
            w_sample = w_sample / (w_sample.sum(axis=1, keepdims=True) + eps)
            h_sample = h_sample / (h_sample.sum(axis=1, keepdims=True) + eps)
            topic_activity_samples.append(w_sample.mean(axis=0))
            doc_max_prob_samples.append(w_sample.max(axis=1))
            h_mean_accum += h_sample
            h_sq_accum += h_sample ** 2
            w_mean_accum += w_sample
            w_sq_accum += w_sample ** 2

        topic_activity_samples = np.asarray(topic_activity_samples)
        doc_max_prob_samples = np.asarray(doc_max_prob_samples)
        h_mean = h_mean_accum / n_samples
        h_std = np.sqrt(np.maximum((h_sq_accum / n_samples) - h_mean ** 2, 0.0))
        w_mean = w_mean_accum / n_samples
        w_std = np.sqrt(np.maximum((w_sq_accum / n_samples) - w_mean ** 2, 0.0))
        self.posterior_summary_ = {
            "topic_activity_mean": topic_activity_samples.mean(axis=0),
            "topic_activity_std": topic_activity_samples.std(axis=0),
            "topic_activity_ci_low": np.quantile(topic_activity_samples, 0.025, axis=0),
            "topic_activity_ci_high": np.quantile(topic_activity_samples, 0.975, axis=0),
            # Bayesian-ARD diagnostic: probability that each topic has activity
            # below the relevance threshold. High values indicate weak or
            # redundant topic candidates.
            "inactive_topic_probability": (topic_activity_samples < bnmf.Config.ARD_ACTIVITY_THRESHOLD).mean(axis=0),
            "doc_max_prob_mean": doc_max_prob_samples.mean(axis=0),
            "doc_max_prob_std": doc_max_prob_samples.std(axis=0),
            "doc_max_prob_ci_low": np.quantile(doc_max_prob_samples, 0.025, axis=0),
            "doc_max_prob_ci_high": np.quantile(doc_max_prob_samples, 0.975, axis=0),
            "H_mean": h_mean,
            "H_std": h_std,
            "W_mean": w_mean,
            "W_std": w_std,
        }
        return self.posterior_summary_


def select_optimal_k_bnmf_fast(dtm, feature_names, min_k: int, max_k: int):
    """
    Selects K for AviTech-Bayes.

    This is the analogue of the LDA "best K" graph, but adapted to BNMF. The
    best K is not chosen by perplexity. It is chosen by a composite criterion:
        - reconstruction_norm: how well W x H reconstructs X.
        - entropy_norm: how balanced the topics are across documents.
        - exclusivity_norm: how exclusive the top words are to each topic.
        - ard_norm: Bayesian-ARD activity, i.e., whether the fitted topics are
          active instead of being weak/redundant components.

    This criterion is why the AviTech-Bayes K plot is methodologically
    different from the LDA K plot.
    """
    max_k = min(max_k, 6)
    rows = []
    terms = list(feature_names)
    print(f"\n[FAST {NEW_METHOD_NAME} K] Testing K from {min_k} to {max_k}...")
    for k in range(min_k, max_k + 1):
        try:
            model = FastAviTechBayesModel(
                n_components=k,
                max_iter=min(int(getattr(bnmf.Config, "NMF_MAX_ITER", 100)), 90),
                tol=bnmf.Config.NMF_TOL,
                posterior_samples=40,
            )
            model.fit(dtm)
            reconstruction_error = float(model.reconstruction_error_[-1])
            topic_dist = model.W.mean(axis=0)
            topic_dist = topic_dist / (topic_dist.sum() + 1e-10)
            topic_entropy = float(-np.sum(topic_dist * np.log(topic_dist + 1e-10)) / np.log(k))
            words = model.get_top_words(terms, n_words=10)
            exclusivity = float(topic_exclusivity(words))
            active_topics_ard = int(np.sum(topic_dist >= bnmf.Config.ARD_ACTIVITY_THRESHOLD))
            ard_activity_score = float(active_topics_ard / k)
            rows.append(
                {
                    "k": k,
                    "reconstruction_error": reconstruction_error,
                    "topic_entropy": topic_entropy,
                    "exclusivity": exclusivity,
                    "active_topics_ard": active_topics_ard,
                    "ard_activity_score": ard_activity_score,
                    "log_posterior": model.log_posterior_[-1] if model.log_posterior_ else np.nan,
                    "converged": True,
                }
            )
            print(f"   K={k}: error={reconstruction_error:.5f}; entropy={topic_entropy:.3f}; exclusivity={exclusivity:.3f}; ARD={active_topics_ard}/{k}")
        except Exception as exc:
            print(f"   K={k}: failed - {str(exc)[:90]}")

    if not rows:
        raise ValueError(f"Could not calculate {NEW_METHOD_NAME} metrics for any K.")

    df_metrics = pd.DataFrame(rows)
    df_metrics["reconstruction_norm"] = 1 - (
        (df_metrics["reconstruction_error"] - df_metrics["reconstruction_error"].min())
        / (df_metrics["reconstruction_error"].max() - df_metrics["reconstruction_error"].min() + 1e-10)
    )
    df_metrics["entropy_norm"] = df_metrics["topic_entropy"] / (df_metrics["topic_entropy"].max() + 1e-10)
    df_metrics["exclusivity_norm"] = df_metrics["exclusivity"]
    df_metrics["ard_norm"] = df_metrics["ard_activity_score"]
    df_metrics["composite_score"] = (
        0.35 * df_metrics["reconstruction_norm"]
        + 0.25 * df_metrics["entropy_norm"]
        + 0.25 * df_metrics["exclusivity_norm"]
        + 0.15 * df_metrics["ard_norm"]
    )
    optimal_k = int(df_metrics.loc[df_metrics["composite_score"].idxmax(), "k"])
    print(f"{NEW_METHOD_NAME} best K selected: {optimal_k}")
    return optimal_k, df_metrics


def run_bnmf_pipeline(shared_results_df, dtm, feature_names, results_dir: Path):
    print("\n=== BATH-SEP-BNMF PIPELINE ===")
    dirs = output_dirs(results_dir)
    bnmf_dir = dirs["bnmf"]
    bnmf_dir.mkdir(parents=True, exist_ok=True)

    max_k_allowed = min(bnmf.Config.MAX_K, max(bnmf.Config.MIN_K, dtm.shape[0] // 5))
    if max_k_allowed < bnmf.Config.MIN_K:
        max_k_allowed = bnmf.Config.MIN_K

    avitech_dtm, avitech_weights = build_avitech_bayes_matrix(dtm, feature_names)
    avitech_weights.to_excel(bnmf_dir / "04_avitech_feature_weights.xlsx", index=False)

    optimal_k, k_metrics = select_optimal_k_bnmf_fast(avitech_dtm, feature_names, bnmf.Config.MIN_K, max_k_allowed)
    k_metrics.to_excel(bnmf_dir / "04_bnmf_k_selection_metrics.xlsx", index=False)
    bnmf.plot_k_selection(k_metrics, bnmf_dir / "04_bnmf_k_selection_plot.png")
    plot_avitech_k_criteria(k_metrics, bnmf_dir / "04_avitech_k_selection_criteria.png")
    pd.DataFrame([
        {"criterion": "reconstruction_norm", "weight": 0.35, "meaning": "Higher means lower reconstruction error."},
        {"criterion": "entropy_norm", "weight": 0.25, "meaning": "Higher means more balanced topic distribution."},
        {"criterion": "exclusivity_norm", "weight": 0.25, "meaning": "Higher means top words are more exclusive to each topic."},
        {"criterion": "ard_norm", "weight": 0.15, "meaning": "Higher means Bayesian-ARD activity supports the selected topic structure."},
        {"criterion": "composite_score", "weight": 1.00, "meaning": "Weighted score used to choose the best K."},
        {"criterion": "avitech_matrix", "weight": np.nan, "meaning": "BNMF is fitted on TF-IDF + objective-weighted DTM to emphasize poultry, technology and economic feasibility terms."},
    ]).to_excel(bnmf_dir / "04_avitech_k_selection_criteria_explained.xlsx", index=False)

    model = FastAviTechBayesModel(
        n_components=optimal_k,
        max_iter=min(int(getattr(bnmf.Config, "NMF_MAX_ITER", 120)), 120),
        tol=bnmf.Config.NMF_TOL,
        posterior_samples=60,
    )
    model.fit(avitech_dtm)

    topic_assignments = model.get_topic_assignments()
    topic_probs = model.W
    enriched_df = shared_results_df.copy()
    enriched_df["topic_bnmf"] = topic_assignments
    enriched_df["topic_bnmf_prob_map"] = np.max(topic_probs, axis=1)

    topic_counts = enriched_df["topic_bnmf"].value_counts().to_dict()
    bnmf.plot_topic_distribution(topic_counts, bnmf_dir / "04_bnmf_topic_distribution.png")
    bnmf.plot_topic_correlation(topic_probs, bnmf_dir / "04_bnmf_topic_correlation.png")
    create_pyldavis_from_distributions(
        model.H,
        model.W,
        dtm,
        feature_names,
        bnmf_dir / "04_avitech_bayes_pyldavis_style.html",
        f"{NEW_METHOD_NAME} Topic Exploration via pyLDAvis-style W/H Interface",
    )

    terms = list(feature_names)
    top_words_per_topic = model.get_top_words(terms, n_words=15)
    topic_words_rows = []
    for topic_id, words in enumerate(top_words_per_topic, 1):
        for rank, word in enumerate(words, 1):
            topic_words_rows.append({"topic": topic_id, "rank": rank, "word": word})
    pd.DataFrame(topic_words_rows).to_excel(bnmf_dir / "04_bnmf_topic_words.xlsx", index=False)

    df_importance = bnmf.compute_topic_importance(model, terms, topic_counts)
    df_importance.to_excel(bnmf_dir / "04_bnmf_topic_importance.xlsx", index=False)
    bnmf.plot_importance_scores(df_importance, bnmf_dir / "04_bnmf_topic_importance_plot.png")
    plot_avitech_preponderant_topics(
        df_importance,
        top_words_per_topic,
        bnmf_dir / "04_avitech_preponderant_topics.png",
    )
    plot_avitech_priority_matrix(
        df_importance,
        top_words_per_topic,
        bnmf_dir / "04_avitech_topic_priority_matrix.png",
    )

    topic_uncertainty, word_uncertainty, doc_uncertainty = bnmf.export_bayesian_posterior(
        model,
        terms,
        enriched_df.rename(columns={"topic_bnmf_prob_map": "topic_prob", "topic_bnmf": "topic_nmf"}),
        bnmf_dir / "04_bnmf_bayesian_posterior_uncertainty.xlsx",
    )

    enriched_df["topic_bnmf_prob_posterior_mean"] = doc_uncertainty["topic_prob_posterior_mean"].values
    enriched_df["topic_bnmf_prob_posterior_std"] = doc_uncertainty["topic_prob_posterior_std"].values
    enriched_df["topic_bnmf_prob_ci_2_5"] = doc_uncertainty["topic_prob_ci_2_5"].values
    enriched_df["topic_bnmf_prob_ci_97_5"] = doc_uncertainty["topic_prob_ci_97_5"].values
    for topic_idx in range(optimal_k):
        enriched_df[f"bnmf_topic_{topic_idx + 1}_prob"] = topic_probs[:, topic_idx]

    enriched_df.to_excel(bnmf_dir / "04_bnmf_documents_with_topics.xlsx", index=False)
    enriched_df.to_csv(bnmf_dir / "04_bnmf_documents_with_topics.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        {
            "metric": [
                "method",
                "optimal_k",
                "active_topics_ard",
                "gamma_prior_w_shape_rate",
                "gamma_prior_h_shape_rate",
                "posterior_samples",
                "convergence_iterations",
                "final_log_posterior",
            ],
            "value": [
                "BATH-SEP-BNMF",
                optimal_k,
                int((topic_uncertainty["ard_status"] == "active").sum()),
                f"{bnmf.Config.BAYESIAN_SHAPE_W}/{bnmf.Config.BAYESIAN_RATE_W}",
                f"{bnmf.Config.BAYESIAN_SHAPE_H}/{bnmf.Config.BAYESIAN_RATE_H}",
                bnmf.Config.BAYESIAN_POSTERIOR_SAMPLES,
                len(model.reconstruction_error_),
                model.log_posterior_[-1] if model.log_posterior_ else np.nan,
            ],
        }
    )
    summary.to_excel(bnmf_dir / "04_bnmf_summary.xlsx", index=False)
    create_new_method_tables_and_images(
        model,
        top_words_per_topic,
        topic_probs,
        df_importance,
        topic_uncertainty,
        bnmf_dir,
    )
    export_avitech_labeling_workbook(
        {
            "topic_words": top_words_per_topic,
            "importance": df_importance,
            "topic_uncertainty": topic_uncertainty,
            "results_df": enriched_df,
        },
        bnmf_dir / "04_avitech_topic_labeling_workbook.xlsx",
    )

    return {
        "best_k": optimal_k,
        "model": model,
        "results_df": enriched_df,
        "topic_probs": topic_probs,
        "topic_words": top_words_per_topic,
        "feature_names": feature_names,
        "importance": df_importance,
        "topic_uncertainty": topic_uncertainty,
        "word_uncertainty": word_uncertainty,
        "doc_uncertainty": doc_uncertainty,
    }


def create_new_method_tables_and_images(model, top_words_per_topic, topic_probs, df_importance, topic_uncertainty, bnmf_dir: Path):
    method_dir = bnmf_dir / f"{NEW_METHOD_NAME}_evidence"
    method_dir.mkdir(parents=True, exist_ok=True)
    img_dir = method_dir / "IMAGES"
    img_dir.mkdir(parents=True, exist_ok=True)

    _, alignment_df = topic_objective_alignment(top_words_per_topic)
    alignment_df["method"] = NEW_METHOD_NAME
    with pd.ExcelWriter(method_dir / f"{NEW_METHOD_NAME}_tables.xlsx") as writer:
        alignment_df.to_excel(writer, sheet_name="Objective_Alignment", index=False)
        df_importance.to_excel(writer, sheet_name="Topic_Importance", index=False)
        topic_uncertainty.to_excel(writer, sheet_name="Posterior_Uncertainty", index=False)

    if not MATPLOTLIB_AVAILABLE:
        return

    plot_df = alignment_df.copy().sort_values("objective_alignment_score", ascending=True)
    objective_cols = ["poultry_hits", "technology_hits", "economic_hits"]
    objective_labels = {
        "poultry_hits": "Poultry",
        "technology_hits": "Technology",
        "economic_hits": "Economic feasibility",
    }
    objective_colors = {
        "poultry_hits": THEME_COLORS["poultry"],
        "technology_hits": THEME_COLORS["technology"],
        "economic_hits": THEME_COLORS["economic"],
    }
    plot_df["objective_total_hits"] = plot_df[objective_cols].sum(axis=1)
    plot_df["dominant_objective"] = plot_df[objective_cols].idxmax(axis=1).map(objective_labels)

    fig, ax = plt.subplots(figsize=(13, max(5.5, len(plot_df) * 0.62)))
    y_labels = [f"T{int(t)}" for t in plot_df["topic"]]
    left = np.zeros(len(plot_df), dtype=float)
    for col in objective_cols:
        values = plot_df[col].astype(float).values
        ax.barh(
            y_labels,
            values,
            left=left,
            color=objective_colors[col],
            edgecolor="white",
            linewidth=0.8,
            label=objective_labels[col],
        )
        left += values

    xmax = max(float(left.max()), 1.0)
    ax.set_xlim(0, xmax * 1.45)
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        dominant = row["dominant_objective"] if row["objective_total_hits"] > 0 else "No objective hits"
        ax.text(
            row["objective_total_hits"] + xmax * 0.025,
            idx,
            f"{dominant} | score={row['objective_alignment_score']:.2f}",
            va="center",
            fontsize=9,
            color="#263238",
        )

    ax.set_title(f"{NEW_METHOD_NAME} - Topic Alignment with the Research Objective", fontweight="bold")
    ax.set_xlabel("Number of objective terms among top topic words")
    ax.set_ylabel("Topic")
    ax.legend(
        title="Objective axis",
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.text(
        0,
        -0.14,
        "Higher stacked bars indicate stronger alignment with the search objective; the annotation names the dominant objective axis for each topic.",
        transform=ax.transAxes,
        fontsize=9,
        color="#546E7A",
    )
    plt.tight_layout()
    plt.savefig(img_dir / "01_topic_alignment_with_research_objective.png", dpi=180, bbox_inches="tight")
    plt.close()

    heatmap_data = alignment_df.set_index("topic")[["poultry_hits", "technology_hits", "economic_hits"]]
    fig, ax = plt.subplots(figsize=(8, max(5, len(heatmap_data) * 0.45)))
    im = ax.imshow(heatmap_data.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(heatmap_data.shape[1]))
    ax.set_xticklabels(["Poultry", "Technology", "Economic feasibility"])
    ax.set_yticks(range(heatmap_data.shape[0]))
    ax.set_yticklabels([f"T{int(t)}" for t in heatmap_data.index])
    ax.set_title(f"{NEW_METHOD_NAME} - Objective Axis Heatmap", fontweight="bold")
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            ax.text(j, i, int(heatmap_data.values[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(img_dir / "02_research_axis_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()

    uncertainty_plot = topic_uncertainty.copy().sort_values("topic")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        uncertainty_plot["topic"],
        uncertainty_plot["activity_mean"],
        yerr=uncertainty_plot["activity_std"],
        fmt="o",
        color="#F58518",
        ecolor="#777777",
        capsize=4,
    )
    ax.set_title(f"{NEW_METHOD_NAME} - Posterior Topic Activity", fontweight="bold")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Posterior activity")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(img_dir / "03_posterior_topic_activity.png", dpi=180, bbox_inches="tight")
    plt.close()

    corr = np.corrcoef(topic_probs.T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"{NEW_METHOD_NAME} - Topic Correlation", fontweight="bold")
    ax.set_xticks(range(corr.shape[0]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels([f"T{i+1}" for i in range(corr.shape[0])])
    ax.set_yticklabels([f"T{i+1}" for i in range(corr.shape[0])])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(img_dir / "04_avitech_bayes_topic_correlation.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_avitech_preponderant_topics(df_importance, top_words_per_topic, output_path: Path):
    if not MATPLOTLIB_AVAILABLE or df_importance.empty:
        return
    plot_df = df_importance.copy().sort_values("total_score", ascending=False).reset_index(drop=True)
    plot_df["rank"] = np.arange(1, len(plot_df) + 1)
    plot_df["objective_alignment"] = pd.cut(
        plot_df["rank"],
        bins=[0, 2, 4, np.inf],
        labels=["HIGH ALIGNMENT", "MODERATE ALIGNMENT", "LOW ALIGNMENT"],
        include_lowest=True,
    )
    alignment_colors = {
        "HIGH ALIGNMENT": "#2CA02C",
        "MODERATE ALIGNMENT": "#F58518",
        "LOW ALIGNMENT": "#9E9E9E",
    }
    plot_df["color"] = plot_df["objective_alignment"].astype(str).map(alignment_colors).fillna("#4C78A8")
    plot_df_asc = plot_df.sort_values("total_score", ascending=True)
    labels = []
    for _, row in plot_df_asc.iterrows():
        topic_id = int(row["topic"])
        words = ", ".join(top_words_per_topic[topic_id - 1][:5]) if topic_id - 1 < len(top_words_per_topic) else ""
        labels.append(f"#{int(row['rank'])}  T{topic_id}: {words}")

    fig, ax = plt.subplots(figsize=(15, max(7, len(plot_df_asc) * 0.72)))
    bars = ax.barh(labels, plot_df_asc["total_score"], color=plot_df_asc["color"], edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Objective-alignment score")
    ax.set_title(f"{NEW_METHOD_NAME} - Topics Ranked by Alignment with the Research Objective", fontweight="bold", fontsize=15)
    ax.grid(axis="x", alpha=0.25)
    xmax = float(plot_df_asc["total_score"].max()) if not plot_df_asc.empty else 1.0
    ax.set_xlim(0, xmax * 1.42)
    for bar, row in zip(bars, plot_df_asc.itertuples(index=False)):
        note = "  <-- used for weights" if str(row.objective_alignment) == "HIGH ALIGNMENT" else ""
        ax.text(
            bar.get_width() + xmax * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row.documents)} docs | {row.percentage:.1f}% | {row.objective_alignment}{note}",
            va="center",
            fontsize=9,
            fontweight="bold" if str(row.objective_alignment) == "HIGH ALIGNMENT" else "normal",
        )
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in alignment_colors.items()
    ]
    ax.legend(handles=legend_handles, title="Objective alignment", loc="lower right")
    ax.text(
        0,
        -0.13,
        "Scores represent alignment with the research objective and are used to derive statistical weights for the no-expert multicriteria model.",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    priority_table_path = output_path.with_name(output_path.stem + "_objective_alignment.xlsx")
    export_df = plot_df.copy()
    export_df["top_words_for_labeling"] = export_df["topic"].map(
        lambda t: ", ".join(top_words_per_topic[int(t) - 1][:15]) if int(t) - 1 < len(top_words_per_topic) else ""
    )
    export_df[
        ["rank", "topic", "objective_alignment", "documents", "percentage", "total_score", "top_words_for_labeling"]
    ].to_excel(priority_table_path, index=False)


def plot_avitech_priority_matrix(df_importance, top_words_per_topic, output_path: Path):
    if not MATPLOTLIB_AVAILABLE or df_importance.empty:
        return
    df = df_importance.copy().sort_values("total_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    score_cols = [col for col in ["documents", "percentage", "total_score"] if col in df.columns]
    matrix = df[score_cols].astype(float).copy()
    for col in score_cols:
        col_min, col_max = matrix[col].min(), matrix[col].max()
        matrix[col] = 1.0 if np.isclose(col_min, col_max) else (matrix[col] - col_min) / (col_max - col_min)

    labels = []
    for _, row in df.iterrows():
        topic_id = int(row["topic"])
        words = ", ".join(top_words_per_topic[topic_id - 1][:3]) if topic_id - 1 < len(top_words_per_topic) else ""
        labels.append(f"#{int(row['rank'])} T{topic_id}: {words}")

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.55)))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels(["Documents", "Corpus %", "Alignment score"], fontsize=10)
    ax.set_title(f"{NEW_METHOD_NAME} - Topic Alignment Matrix", fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized alignment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_avitech_k_criteria(k_metrics: pd.DataFrame, output_path: Path):
    if not MATPLOTLIB_AVAILABLE or k_metrics.empty:
        return
    cols = ["reconstruction_norm", "entropy_norm", "exclusivity_norm", "ard_norm", "composite_score"]
    available = [col for col in cols if col in k_metrics.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in available:
        ax.plot(k_metrics["k"], k_metrics[col], marker="o", linewidth=2, label=col.replace("_", " ").title())
    best_k = int(k_metrics.loc[k_metrics["composite_score"].idxmax(), "k"]) if "composite_score" in k_metrics.columns else int(k_metrics["k"].iloc[-1])
    ax.axvline(best_k, color="#E45756", linestyle="--", linewidth=2, label=f"Best K = {best_k}")
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("Normalized criterion")
    ax.set_title(f"{NEW_METHOD_NAME} - K Selection Criteria", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def export_avitech_labeling_workbook(bnmf_result, output_path: Path):
    def clean_export_text(value):
        """Keep exported tables free of broken glyphs and non-ASCII artifacts."""
        import unicodedata

        if pd.isna(value):
            return value
        text = str(value)
        replacements = {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u00a0": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_export_dataframe(df):
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].map(clean_export_text)
        return df

    def avitech_topic_label_and_description(words):
        word_set = set(words)
        joined = " ".join(words)
        if {"detection", "accuracy", "machine_learning", "deep_learning", "monitoring"} & word_set and {"image", "sensor", "behavior"} & word_set:
            return (
                "Computer vision for poultry monitoring",
                "Studies in this group examine image-based monitoring, sensor data, and learning algorithms for detecting poultry behavior, welfare conditions, production patterns, and performance signals.",
            )
        if {"iot", "internet_of_things", "sensors", "automation"} & word_set:
            return (
                "Smart poultry houses and connected sensors",
                "The theme covers IoT devices, smart sensors, automation, and connected management systems used for real-time monitoring, environmental control, and operational decision-making in poultry houses.",
            )
        if {"digital", "economic", "innovation", "digital twin"} & word_set or "digital twin" in joined:
            return (
                "Digital transformation and sustainable agribusiness",
                "The documents associated with this theme relate digital transformation, digital twins, sustainability, platforms, and infrastructure to innovation and economic modernization in poultry and agribusiness.",
            )
        if {"ai", "artificial_intelligence", "automation", "organizational", "adoption"} & word_set:
            return (
                "Artificial intelligence in management and investment decisions",
                "This theme emphasizes the managerial adoption of artificial intelligence and automation, including organizational readiness, ethical concerns, innovation processes, and investment decisions.",
            )
        if {"digital_transformation", "firms", "investment", "supply_chain", "smes"} & word_set:
            return (
                "Digital readiness in agri-food supply chains",
                "The theme is centered on firms, SMEs, investment, strategy, and supply chains, indicating the organizational and economic conditions required for digital technology adoption.",
            )
        if {"blockchain", "supply_chain", "transparency", "traceability"} & word_set:
            return (
                "Blockchain-based traceability in agri-food chains",
                "The documents in this theme discuss blockchain and decentralized records as mechanisms for traceability, transparency, trust, information security, cost control, and supply-chain management.",
            )
        return (
            "Digital technology and economic feasibility",
            "This residual theme combines terms related to digital technology, production, innovation, and economic feasibility in poultry and related agri-food systems.",
        )

    topic_rows = []
    topic_words = bnmf_result["topic_words"]
    importance = bnmf_result["importance"].copy()
    uncertainty = bnmf_result["topic_uncertainty"].copy()
    for topic_id, words in enumerate(topic_words, 1):
        imp = importance[importance["topic"] == topic_id]
        unc = uncertainty[uncertainty["topic"] == topic_id]
        suggested_label, description = avitech_topic_label_and_description(words[:15])
        topic_rows.append({
            "topic": topic_id,
            "keywords": ", ".join(words[:15]),
            "description": description,
            "suggested_label": suggested_label,
            "top_words_15": ", ".join(words[:15]),
            "documents": int(imp.iloc[0]["documents"]) if not imp.empty else 0,
            "percentage": float(imp.iloc[0]["percentage"]) if not imp.empty else 0.0,
            "preponderance_score": float(imp.iloc[0]["total_score"]) if not imp.empty else 0.0,
            "priority": imp.iloc[0]["priority"] if not imp.empty else "",
            "posterior_activity_mean": float(unc.iloc[0]["activity_mean"]) if not unc.empty else np.nan,
            "inactive_probability_ard": float(unc.iloc[0]["inactive_probability_ard"]) if not unc.empty else np.nan,
        })
    topic_labeling = clean_export_dataframe(pd.DataFrame(topic_rows).sort_values("preponderance_score", ascending=False))
    publication_table = topic_labeling[["topic", "keywords", "description", "suggested_label"]].rename(
        columns={"suggested_label": "label"}
    )
    with pd.ExcelWriter(output_path) as writer:
        topic_labeling.to_excel(writer, sheet_name="Topic_Labeling", index=False)
        publication_table.to_excel(writer, sheet_name="Publication_Table", index=False)
        clean_export_dataframe(importance).to_excel(writer, sheet_name="Preponderance", index=False)
        clean_export_dataframe(uncertainty).to_excel(writer, sheet_name="Bayesian_Uncertainty", index=False)
        clean_export_dataframe(bnmf_result["results_df"]).to_excel(writer, sheet_name="Documents", index=False)


def export_combined_summary(lda_result, bnmf_result, results_dir: Path):
    dirs = output_dirs(results_dir)
    common_dir = dirs["common"]
    comparison = pd.DataFrame(
        {
            "metric": [
                "analysis_name",
                "analysis_date",
                "lda_optimal_k",
                f"{NEW_METHOD_NAME}_optimal_k",
                "documents_analyzed",
                "legacy_hoshin_kanri_analysis",
            ],
            "value": [
                f"LDA + {NEW_METHOD_NAME} for poultry technology and economic feasibility",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                lda_result["best_k"],
                bnmf_result["best_k"],
                len(lda_result["results_df"]),
                "Not part of this script; only the inherited LDA topic model is reused.",
            ],
        }
    )

    topic_crosswalk = (
        bnmf_result["results_df"]
        .groupby(["main_topic", "topic_label", "topic_bnmf"])
        .size()
        .reset_index(name="documents")
        .sort_values(["main_topic", "documents"], ascending=[True, False])
    )

    with pd.ExcelWriter(common_dir / "05_combined_lda_bnmf_summary_topic_model_corpus.xlsx") as writer:
        comparison.to_excel(writer, sheet_name="Summary", index=False)
        topic_crosswalk.to_excel(writer, sheet_name="LDA_BNMF_Crosswalk", index=False)
        lda_result["priority"].to_excel(writer, sheet_name="LDA_Topic_Priority", index=False)
        bnmf_result["importance"].to_excel(writer, sheet_name="AviTech_Topic_Importance", index=False)
        bnmf_result["topic_uncertainty"].to_excel(writer, sheet_name="AviTech_Uncertainty", index=False)


def topic_exclusivity(topic_words):
    all_words = []
    for words in topic_words:
        all_words.extend(words[:10])
    if not all_words:
        return 0.0
    counts = Counter(all_words)
    unique_count = sum(1 for word in all_words if counts[word] == 1)
    return unique_count / len(all_words)


def topic_objective_alignment(topic_words):
    """Measures alignment with poultry + digital/PLF + economic feasibility."""
    rows = []
    scores = []
    for topic_id, words in enumerate(topic_words, 1):
        normalized_words = {str(word).lower() for word in words[:15]}
        group_hits = {
            group: len(normalized_words & terms)
            for group, terms in OBJECTIVE_GROUPS.items()
        }
        group_presence = {
            group: 1 if hits > 0 else 0
            for group, hits in group_hits.items()
        }
        coverage_score = sum(group_presence.values()) / len(OBJECTIVE_GROUPS)
        density_score = min(sum(group_hits.values()) / 9.0, 1.0)
        score = (0.70 * coverage_score) + (0.30 * density_score)
        scores.append(score)
        rows.append({
            "topic": topic_id,
            "objective_alignment_score": score,
            "poultry_hits": group_hits["poultry"],
            "technology_hits": group_hits["technology"],
            "economic_hits": group_hits["economic"],
            "covers_poultry": group_presence["poultry"],
            "covers_technology": group_presence["technology"],
            "covers_economic": group_presence["economic"],
            "top_words": ", ".join(words[:15]),
        })
    return float(np.mean(scores)) if scores else 0.0, pd.DataFrame(rows)


def normalized_topic_entropy(assignments, n_topics):
    counts = pd.Series(assignments).value_counts().reindex(range(1, n_topics + 1), fill_value=0).values
    probs = counts / counts.sum() if counts.sum() else np.zeros_like(counts, dtype=float)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy / np.log(n_topics)) if n_topics > 1 else 1.0


def mean_topic_coherence(topic_words, dtm, feature_names):
    coherences = []
    for words in topic_words:
        coherences.append(lda.calculate_topic_coherence_for_topic(feature_names, dtm, words[:10]))
    return float(np.mean(coherences)) if coherences else 0.0


def normalize_metric(values, higher_is_better=True):
    values = np.asarray(values, dtype=float)
    if np.allclose(values.max(), values.min()):
        return np.ones_like(values)
    scaled = (values - values.min()) / (values.max() - values.min())
    return scaled if higher_is_better else 1.0 - scaled


def build_method_comparison(lda_result, bnmf_result, dtm, feature_names):
    lda_words = [words for _, words in sorted(lda_result["topic_words"].items())]
    bnmf_words = bnmf_result["topic_words"]
    lda_alignment, lda_alignment_topics = topic_objective_alignment(lda_words)
    bnmf_alignment, bnmf_alignment_topics = topic_objective_alignment(bnmf_words)

    if lda_result.get("doc_topic") is not None:
        lda_assignments = np.argmax(lda_result["doc_topic"], axis=1) + 1
        lda_confidence = float(np.max(lda_result["doc_topic"], axis=1).mean())
    else:
        lda_assignments = lda_result["results_df"]["main_topic"].values
        lda_confidence = float(lda_result["results_df"]["topic_probability"].mean())
    bnmf_assignments = bnmf_result["results_df"]["topic_bnmf"].values

    raw = pd.DataFrame(
        [
            {
                "method": "LDA",
                "optimal_k": lda_result["best_k"],
                "topic_coherence": mean_topic_coherence(lda_words, dtm, feature_names),
                "word_exclusivity": topic_exclusivity(lda_words),
                "objective_alignment": lda_alignment,
                "topic_balance": normalized_topic_entropy(lda_assignments, lda_result["best_k"]),
                "assignment_confidence": lda_confidence,
                "posterior_uncertainty": np.nan,
                "uncertainty_quantification": 0.0,
            },
            {
                "method": NEW_METHOD_NAME,
                "optimal_k": bnmf_result["best_k"],
                "topic_coherence": mean_topic_coherence(bnmf_words, dtm, feature_names),
                "word_exclusivity": topic_exclusivity(bnmf_words),
                "objective_alignment": bnmf_alignment,
                "topic_balance": normalized_topic_entropy(bnmf_assignments, bnmf_result["best_k"]),
                "assignment_confidence": float(bnmf_result["results_df"]["topic_bnmf_prob_map"].mean()),
                "posterior_uncertainty": float(bnmf_result["doc_uncertainty"]["topic_prob_posterior_std"].mean()),
                "uncertainty_quantification": 1.0,
            },
        ]
    )

    raw["coherence_norm"] = normalize_metric(raw["topic_coherence"])
    raw["exclusivity_norm"] = normalize_metric(raw["word_exclusivity"])
    raw["alignment_norm"] = normalize_metric(raw["objective_alignment"])
    raw["balance_norm"] = normalize_metric(raw["topic_balance"])
    raw["confidence_norm"] = normalize_metric(raw["assignment_confidence"])
    raw["uncertainty_norm"] = raw["uncertainty_quantification"].astype(float)
    raw["composite_score"] = (
        0.30 * raw["alignment_norm"]
        + 0.20 * raw["coherence_norm"]
        + 0.20 * raw["exclusivity_norm"]
        + 0.10 * raw["balance_norm"]
        + 0.10 * raw["confidence_norm"]
        + 0.10 * raw["uncertainty_norm"]
    )
    raw["rank"] = raw["composite_score"].rank(ascending=False, method="min").astype(int)
    raw.attrs["topic_alignment"] = {
        "LDA": lda_alignment_topics.assign(method="LDA"),
        NEW_METHOD_NAME: bnmf_alignment_topics.assign(method=NEW_METHOD_NAME),
    }
    return raw.sort_values("rank").reset_index(drop=True)


def plot_method_comparison(comparison_df, output_dir: Path):
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available; comparison images skipped.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["objective_alignment", "topic_coherence", "word_exclusivity", "topic_balance", "assignment_confidence", "uncertainty_quantification", "composite_score"]
    labels = ["Search Alignment", "Coherence", "Exclusivity", "Balance", "Confidence", "Bayesian Uncertainty", "Composite"]
    plot_df = comparison_df.set_index("method").loc[:, metrics]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    width = 0.36
    methods = plot_df.index.tolist()
    colors = ["#4C78A8", "#F58518"]
    for idx, method in enumerate(methods):
        ax.bar(x + (idx - 0.5) * width, plot_df.loc[method].values, width, label=method, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score / Metric value")
    ax.set_title(f"Method Comparison: LDA vs {NEW_METHOD_NAME}", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "05_method_comparison_scores.png", dpi=180, bbox_inches="tight")
    plt.close()

    radar_metrics = ["alignment_norm", "coherence_norm", "exclusivity_norm", "balance_norm", "confidence_norm", "uncertainty_norm"]
    radar_labels = ["Search Alignment", "Coherence", "Exclusivity", "Balance", "Confidence", "Bayesian Uncertainty"]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for idx, row in comparison_df.iterrows():
        values = row[radar_metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["method"], color=colors[idx])
        ax.fill(angles, values, alpha=0.12, color=colors[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Normalized Performance Profile", fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
    plt.tight_layout()
    plt.savefig(output_dir / "05_method_comparison_radar.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_best_method_decision(comparison_df, output_path: Path):
    if not MATPLOTLIB_AVAILABLE or comparison_df.empty:
        return

    plot_df = comparison_df.copy().sort_values("composite_score", ascending=True)
    winner = comparison_df.sort_values("composite_score", ascending=False).iloc[0]
    colors = ["#9E9E9E" if method != winner["method"] else "#2CA02C" for method in plot_df["method"]]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(plot_df["method"], plot_df["composite_score"], color=colors, edgecolor="black")
    ax.set_xlim(0, max(1.0, float(plot_df["composite_score"].max()) * 1.18))
    ax.set_xlabel("Composite performance score")
    ax.set_title("Which Method Performed Better?", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for bar, score, rank in zip(bars, plot_df["composite_score"], plot_df["rank"]):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"rank {int(rank)} | {score:.3f}",
            va="center",
            fontsize=10,
        )
    ax.text(
        0.01,
        0.02,
        f"Winner: {winner['method']} | Criteria: alignment + coherence + exclusivity + balance + confidence + Bayesian uncertainty",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#DDDDDD", alpha=0.95),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def copy_if_exists(src: Path, dst: Path, copied_rows: list):
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_rows.append(
            {
                "source": str(src),
                "destination": str(dst),
                "type": src.suffix.lower().lstrip("."),
                "size_kb": round(dst.stat().st_size / 1024, 1),
            }
        )


def export_best_method_named_package(best_method, lda_result, bnmf_result, comparison_df, best_topic, results_dir: Path):
    dirs = output_dirs(results_dir)
    safe_method = re.sub(r"[^A-Za-z0-9_-]+", "_", str(best_method)).strip("_") or "best_method"
    package_dir = dirs["best"] / safe_method
    img_dir = package_dir / "IMAGES"
    table_dir = package_dir / "TABLES"
    labeling_dir = package_dir / "TOPIC_LABELING"
    img_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    labeling_dir.mkdir(parents=True, exist_ok=True)

    copied_rows = []
    common_files = [
        (dirs["common"] / "05_method_comparison_metrics.xlsx", table_dir / "01_method_comparison_metrics.xlsx"),
        (dirs["common"] / "05_topic_alignment_by_method.xlsx", table_dir / "02_topic_alignment_with_search_objective.xlsx"),
        (dirs["common"] / "06_preponderant_topics_lda_bnmf.xlsx", table_dir / "03_preponderant_topics_lda_vs_avitech.xlsx"),
        (dirs["common"] / "05_method_comparison_scores.png", img_dir / "01_method_comparison_bar_chart.png"),
        (dirs["common"] / "05_method_comparison_radar.png", img_dir / "02_method_comparison_radar.png"),
        (dirs["common"] / "05_best_method_decision.png", img_dir / "03_best_method_decision.png"),
        (dirs["common"] / "05_topic_alignment_with_research_objective.png", img_dir / "04_topic_alignment_with_research_objective.png"),
    ]

    if best_method == "LDA":
        method_files = [
            (dirs["lda"] / "03_lda_results_topic_model_corpus.xlsx", table_dir / "10_lda_documents_with_topics.xlsx"),
            (dirs["lda"] / "03_lda_topic_priority_chart.png", img_dir / "10_lda_preponderant_topics.png"),
            (dirs["lda"] / "03_lda_topic_distribution_pie.png", img_dir / "11_lda_topic_distribution.png"),
            (dirs["lda"] / "03_lda_topic_correlation.png", img_dir / "12_lda_correlation_matrix_between_topics.png"),
            (dirs["lda"] / "IMAGES_LDA" / "02_lda_optimal_k.png", img_dir / "13_lda_best_k.png"),
            (dirs["lda"] / "02_lda_topic_evaluation.png", img_dir / "14_lda_k_selection_criteria.png"),
            (dirs["lda"] / "03_lda_pyldavis_visualization.html", package_dir / "INTERACTIVE" / "15_lda_pyldavis_visualization.html"),
        ]
        topic_words = [words for _, words in sorted(lda_result["topic_words"].items())]
        docs_df = lda_result["results_df"].copy()
        topic_col = "main_topic"
        prob_col = "topic_probability"
    else:
        method_files = [
            (dirs["bnmf"] / "04_avitech_topic_labeling_workbook.xlsx", labeling_dir / "10_avitech_topic_labeling_workbook.xlsx"),
            (dirs["bnmf"] / "04_bnmf_topic_words.xlsx", table_dir / "11_avitech_topic_words.xlsx"),
            (dirs["bnmf"] / "04_bnmf_topic_importance.xlsx", table_dir / "12_avitech_topic_preponderance.xlsx"),
            (dirs["bnmf"] / "04_bnmf_bayesian_posterior_uncertainty.xlsx", table_dir / "13_avitech_bayesian_posterior_uncertainty.xlsx"),
            (dirs["bnmf"] / "04_avitech_k_selection_criteria_explained.xlsx", table_dir / "14_avitech_k_selection_criteria_explained.xlsx"),
            (dirs["bnmf"] / "04_avitech_feature_weights.xlsx", table_dir / "14b_avitech_feature_weights.xlsx"),
            (dirs["bnmf"] / "04_bnmf_documents_with_topics.xlsx", table_dir / "15_avitech_documents_with_topics.xlsx"),
            (dirs["bnmf"] / "04_avitech_preponderant_topics.png", img_dir / "10_avitech_most_preponderant_topics.png"),
            (dirs["bnmf"] / "04_avitech_topic_priority_matrix.png", img_dir / "10b_avitech_topic_priority_matrix.png"),
            (dirs["bnmf"] / "04_avitech_preponderant_topics_labeling_priority.xlsx", labeling_dir / "10b_avitech_labeling_priority.xlsx"),
            (dirs["bnmf"] / "04_avitech_k_selection_criteria.png", img_dir / "11_avitech_best_k_criteria.png"),
            (dirs["bnmf"] / "04_bnmf_k_selection_plot.png", img_dir / "12_avitech_best_k_composite.png"),
            (dirs["bnmf"] / "04_bnmf_topic_correlation.png", img_dir / "13_avitech_correlation_matrix_between_topics.png"),
            (dirs["bnmf"] / "04_bnmf_topic_importance_plot.png", img_dir / "14_avitech_topic_importance_score.png"),
            (dirs["bnmf"] / "04_bnmf_topic_distribution.png", img_dir / "15_avitech_topic_distribution.png"),
            (dirs["bnmf"] / "04_avitech_bayes_pyldavis_style.html", package_dir / "INTERACTIVE" / "17_avitech_bayes_pyldavis_style.html"),
            (dirs["bnmf"] / "AviTech-Bayes_evidence" / "AviTech-Bayes_tables.xlsx", table_dir / "16_avitech_article_evidence_tables.xlsx"),
        ]
        evidence_img_dir = dirs["bnmf"] / "AviTech-Bayes_evidence" / "IMAGES"
        if evidence_img_dir.exists():
            for src in sorted(evidence_img_dir.glob("*.png")):
                method_files.append((src, img_dir / f"20_{src.name}"))
        topic_words = bnmf_result["topic_words"]
        docs_df = bnmf_result["results_df"].copy()
        topic_col = "topic_bnmf"
        prob_col = "topic_bnmf_prob_map"

    for src, dst in common_files + method_files:
        copy_if_exists(src, dst, copied_rows)

    topic_rows = []
    for topic_id, words in enumerate(topic_words, 1):
        topic_docs = docs_df[docs_df[topic_col] == topic_id] if topic_col in docs_df.columns else pd.DataFrame()
        topic_rows.append(
            {
                "topic": topic_id,
                "suggested_label_user_fill": "",
                "priority_note": "MOST_PREPONDERANT_TOPIC" if topic_id == best_topic else "",
                "documents": int(len(topic_docs)),
                "mean_probability_or_weight": float(topic_docs[prob_col].mean()) if prob_col in topic_docs.columns and not topic_docs.empty else np.nan,
                "top_words_20": ", ".join(words[:20]),
                "interpretation_notes_user_fill": "",
            }
        )
    pd.DataFrame(topic_rows).sort_values(["priority_note", "documents"], ascending=[False, False]).to_excel(
        labeling_dir / "00_topic_labeling_workbook.xlsx",
        index=False,
    )

    probability_note = pd.DataFrame(
        [
            {
                "question": "Do both methods produce probabilities?",
                "answer": (
                    "Yes. LDA estimates document-topic and topic-word probabilities. "
                    f"{NEW_METHOD_NAME} uses normalized non-negative weights as document-topic distributions "
                    "and adds Bayesian posterior uncertainty for topic activity and assignment confidence."
                ),
            },
            {
                "question": "Why can the criterion favor the new method when posterior uncertainty is available?",
                "answer": (
                    f"Because the methodological goal of {NEW_METHOD_NAME} is not only to generate topics, "
                    "but also to quantify topic uncertainty and relevance through a Bayesian/ARD layer. "
                    "The classical LDA implementation used here produces probabilities, but does not export "
                    "comparable ARD posterior diagnostics."
                ),
            },
            {
                "question": f"Which criterion selects K in {NEW_METHOD_NAME}?",
                "answer": (
                    "K is selected by the highest composite score, combining lower reconstruction error, "
                    "higher word exclusivity, topic balance/entropy, and an ARD penalty for low-activity topics."
                ),
            },
            {
                "question": "How should this folder be used?",
                "answer": "Use TOPIC_LABELING/00_topic_labeling_workbook.xlsx to name the topics and IMAGES for article-ready figures.",
            },
        ]
    )
    with pd.ExcelWriter(package_dir / "README_best_method.xlsx") as writer:
        probability_note.to_excel(writer, sheet_name="Questions", index=False)
        comparison_df.to_excel(writer, sheet_name="Method_Comparison", index=False)
        pd.DataFrame(copied_rows).to_excel(writer, sheet_name="Copied_Files", index=False)

    return package_dir


def sync_legacy_bath_sep_results(results_dir: Path):
    """
    Keeps the old BATH_SEP_NMF_RESULTS folder useful.

    The unified script writes the new method outputs to 3_BNMF_results. The
    original BNMF script used BATH_SEP_NMF_RESULTS. This function creates a
    lightweight compatibility folder with a README and copies the main
    AviTech-Bayes/BNMF outputs there, so the old folder is not empty.
    """
    dirs = output_dirs(results_dir)
    legacy_dir = results_dir / "BATH_SEP_NMF_RESULTS"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    readme = legacy_dir / "README.txt"
    readme.write_text(
        "\n".join([
            "This folder is kept for compatibility with the older BNMF script.",
            "In the unified script, the official AviTech-Bayes/BNMF results are saved in:",
            str(dirs["bnmf"]),
            "",
            "Main files copied to this folder:",
            "- K selection",
            "- preponderant topics",
            "- topic words",
            "- Bayesian posterior uncertainty",
            "- documents with BNMF topics",
        ]),
        encoding="utf-8",
    )
    copy_pairs = [
        ("04_avitech_k_selection_criteria.png", "04_avitech_k_selection_criteria.png"),
        ("04_avitech_preponderant_topics.png", "04_avitech_preponderant_topics.png"),
        ("04_bnmf_k_selection_plot.png", "04_bnmf_k_selection_plot.png"),
        ("04_bnmf_topic_correlation.png", "04_bnmf_topic_correlation.png"),
        ("04_bnmf_topic_distribution.png", "04_bnmf_topic_distribution.png"),
        ("04_bnmf_topic_words.xlsx", "04_bnmf_topic_words.xlsx"),
        ("04_avitech_feature_weights.xlsx", "04_avitech_feature_weights.xlsx"),
        ("04_bnmf_topic_importance.xlsx", "04_bnmf_topic_importance.xlsx"),
        ("04_bnmf_summary.xlsx", "04_bnmf_summary.xlsx"),
        ("04_bnmf_bayesian_posterior_uncertainty.xlsx", "04_bnmf_bayesian_posterior_uncertainty.xlsx"),
        ("04_bnmf_documents_with_topics.xlsx", "04_bnmf_documents_with_topics.xlsx"),
        ("04_avitech_topic_labeling_workbook.xlsx", "04_avitech_topic_labeling_workbook.xlsx"),
    ]
    for src_name, dst_name in copy_pairs:
        src = dirs["bnmf"] / src_name
        if src.exists():
            shutil.copy2(src, legacy_dir / dst_name)
    return legacy_dir


def plot_best_method_results(best_method, lda_result, bnmf_result, results_dir: Path):
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available; best-method images skipped.")
        return

    dirs = output_dirs(results_dir)
    output_dir = dirs["best"] / "BEST_METHOD_IMAGES"
    output_dir.mkdir(parents=True, exist_ok=True)

    if best_method == "LDA":
        df = lda_result["results_df"].copy()
        topic_col = "main_topic"
        prob_col = "topic_probability"
        topic_words = [words for _, words in sorted(lda_result["topic_words"].items())]
        title_prefix = "Best Method: LDA"
    else:
        df = bnmf_result["results_df"].copy()
        topic_col = "topic_bnmf"
        prob_col = "topic_bnmf_prob_map"
        topic_words = bnmf_result["topic_words"]
        title_prefix = f"Best Method: {NEW_METHOD_NAME}"

    counts = df[topic_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#54A24B", edgecolor="black")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Documents")
    ax.set_title(f"{title_prefix} - Topic Distribution", fontweight="bold")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value), ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "06_best_method_topic_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    top_rows = []
    for topic_id, words in enumerate(topic_words, 1):
        for rank, word in enumerate(words[:8], 1):
            top_rows.append({"topic": topic_id, "rank": rank, "word": word, "score": 9 - rank})
    top_df = pd.DataFrame(top_rows)
    fig, axes = plt.subplots(len(topic_words), 1, figsize=(11, max(4, len(topic_words) * 1.7)), sharex=True)
    if len(topic_words) == 1:
        axes = [axes]
    for ax, topic_id in zip(axes, range(1, len(topic_words) + 1)):
        subset = top_df[top_df["topic"] == topic_id].sort_values("rank", ascending=False)
        ax.barh(subset["word"], subset["score"], color="#4C78A8")
        ax.set_title(f"Topic {topic_id}", loc="left", fontsize=10, fontweight="bold")
        ax.set_xticks([])
    fig.suptitle(f"{title_prefix} - Top Words by Topic", fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "06_best_method_top_words.png", dpi=180, bbox_inches="tight")
    plt.close()

    if "year" in df.columns:
        df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")
        yearly = (
            df.dropna(subset=["year_numeric"])
            .assign(Year=lambda x: x["year_numeric"].astype(int))
            .pivot_table(index="Year", columns=topic_col, values="title", aggfunc="count", fill_value=0)
        )
        if not yearly.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            yearly.plot(kind="area", stacked=True, alpha=0.78, ax=ax, colormap="tab20")
            ax.set_xlabel("Year")
            ax.set_ylabel("Documents")
            ax.set_title(f"{title_prefix} - Topic Evolution Over Time", fontweight="bold")
            ax.legend(title="Topic", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(output_dir / "06_best_method_topic_evolution.png", dpi=180, bbox_inches="tight")
            plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[prob_col].dropna(), bins=20, color="#E45756", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Dominant topic probability")
    ax.set_ylabel("Documents")
    ax.set_title(f"{title_prefix} - Assignment Confidence", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "06_best_method_assignment_confidence.png", dpi=180, bbox_inches="tight")
    plt.close()


def build_topic_preponderance_tables(lda_result, bnmf_result):
    lda_table = lda_result["priority"].copy()
    lda_table = lda_table.rename(
        columns={
            "Topic": "topic",
            "Documents": "documents",
            "Percentage": "percentage",
            "Composite_Score": "preponderance_score",
            "Priority_Level": "priority",
            "Top_Words": "top_words",
        }
    )
    lda_table["method"] = "LDA"
    lda_table["topic_label"] = lda_table["topic"].map(lda_result["labels"]).fillna("")
    lda_table = lda_table[
        ["method", "topic", "topic_label", "documents", "percentage", "preponderance_score", "priority", "top_words"]
    ].sort_values("preponderance_score", ascending=False)

    word_map = {
        topic_id: ", ".join(words[:8])
        for topic_id, words in enumerate(bnmf_result["topic_words"], 1)
    }
    bnmf_table = bnmf_result["importance"].copy()
    bnmf_table = bnmf_table.rename(columns={"total_score": "preponderance_score"})
    bnmf_table["method"] = "BATH-SEP-BNMF"
    bnmf_table["topic_label"] = bnmf_table["topic"].map(lambda t: f"BNMF Topic {int(t)}")
    bnmf_table["top_words"] = bnmf_table["topic"].map(word_map)
    bnmf_table = bnmf_table[
        ["method", "topic", "topic_label", "documents", "percentage", "preponderance_score", "priority", "top_words"]
    ].sort_values("preponderance_score", ascending=False)

    return lda_table.reset_index(drop=True), bnmf_table.reset_index(drop=True)


def plot_topic_preponderance(preponderance_df, method_name, output_path):
    if not MATPLOTLIB_AVAILABLE or preponderance_df.empty:
        return

    plot_df = preponderance_df.head(12).sort_values("preponderance_score", ascending=True)
    rank_map = {
        topic: rank
        for rank, topic in enumerate(
            preponderance_df.sort_values("preponderance_score", ascending=False)["topic"].tolist(),
            1,
        )
    }
    def priority_from_rank(topic):
        rank = rank_map.get(topic, 99)
        if rank <= 2:
            return "EXPLORE FIRST"
        if rank <= 4:
            return "SECONDARY"
        return "MONITOR"
    color_map = {"EXPLORE FIRST": "#2CA02C", "SECONDARY": "#F58518", "MONITOR": "#9E9E9E"}
    priorities = [priority_from_rank(topic) for topic in plot_df["topic"]]
    labels = [
        f"#{rank_map.get(row.topic, 0)} T{int(row.topic)} - {str(row.top_words).split(',')[0]}"
        for row in plot_df.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(13, max(6, len(plot_df) * 0.60)))
    bars = ax.barh(labels, plot_df["preponderance_score"], color=[color_map[p] for p in priorities], edgecolor="black")
    ax.set_xlabel("Preponderance score")
    ax.set_title(f"{method_name} - Most Preponderant Topics / Exploration Priority", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    xmax = float(plot_df["preponderance_score"].max()) if not plot_df.empty else 1.0
    ax.set_xlim(0, xmax * 1.38)
    for bar, docs, pct, priority in zip(bars, plot_df["documents"], plot_df["percentage"], priorities):
        ax.text(
            bar.get_width() + xmax * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{int(docs)} docs | {pct:.1f}% | {priority}",
            va="center",
            fontsize=9,
            fontweight="bold" if priority == "EXPLORE FIRST" else "normal",
        )
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=color, edgecolor="black", label=label) for label, color in color_map.items()],
        title="Priority",
        loc="lower right",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def create_best_topic_wordcloud(best_method, lda_result, bnmf_result, results_dir: Path):
    dirs = output_dirs(results_dir)
    article_img_dir = dirs["best"] / "BEST_METHOD_IMAGES"
    article_img_dir.mkdir(parents=True, exist_ok=True)

    if best_method == "LDA":
        preponderance = lda_result["priority"].sort_values("Composite_Score", ascending=False)
        topic_raw = preponderance.iloc[0]["Topic"]
        match = re.search(r"(\d+)", str(topic_raw))
        best_topic = int(match.group(1)) if match else int(topic_raw)
        if lda_result.get("model") is not None:
            weights = lda_result["model"].components_[best_topic - 1]
            terms = lda_result["feature_names"]
            top_idx = np.argsort(weights)[-150:][::-1]
            freq_df = pd.DataFrame({"word": [terms[i] for i in top_idx], "frequency": weights[top_idx]})
        else:
            words = lda_result["topic_words"].get(f"Topic_{best_topic}", [])
            freq_df = pd.DataFrame(
                {
                    "word": words,
                    "frequency": list(range(len(words), 0, -1)),
                }
            )
    else:
        preponderance = bnmf_result["importance"].sort_values("total_score", ascending=False)
        best_topic = int(preponderance.iloc[0]["topic"])
        weights = bnmf_result["model"].H[best_topic - 1]
        terms = bnmf_result["feature_names"]
        top_idx = np.argsort(weights)[-150:][::-1]
        freq_df = pd.DataFrame({"word": [terms[i] for i in top_idx], "frequency": weights[top_idx]})

    bnmf.create_round_wordcloud(
        freq_df,
        article_img_dir / "07_preponderant_topic_word_cloud.png",
        title=f"Preponderant Topic - {best_method} - T{best_topic}",
        dpi=300,
        size=2200,
    )
    return best_topic


SUBTHEME_TERMS = {
    "1_poultry_production_system": {
        "poultry", "poultry_farming", "broiler", "broilers", "layer", "layers",
        "hen", "hens", "chicken", "chickens", "flock", "flocks", "egg", "eggs",
        "bird", "birds", "production", "housing", "feed", "feeding",
    },
    "2_plf_and_digital_transformation": {
        "precision_livestock_farming", "plf", "digital_transformation", "digital",
        "technology", "smart_technology", "smart_manufacturing",
        "smart_farming", "smart_agriculture", "iot", "internet_of_things", "automation",
    },
    "3_sensors_and_monitoring": {
        "sensor", "sensors", "monitoring", "camera", "cameras", "thermal",
        "image", "images", "imaging", "environment", "weight", "mortality",
    },
    "4_ai_and_computer_vision": {
        "ai", "artificial_intelligence", "artificial", "intelligence",
        "machine_learning", "deep_learning", "machine", "learning",
        "neural_networks", "artificial_neural_networks", "convolutional_neural_networks",
        "computer_vision", "computer", "vision", "algorithm", "algorithms",
        "detection", "accuracy",
    },
    "5_economic_feasibility": {
        "feasibility", "economic_feasibility", "economic", "economics",
        "cost", "costs", "roi", "return_on_investment", "profitability",
        "investment", "viability",
    },
    "6_adoption_barriers_and_management": {
        "adoption", "barrier", "barriers", "challenge", "challenges",
        "farmer", "farmers", "farm", "farms", "management", "risk",
        "decision", "support",
    },
    "7_performance_and_welfare": {
        "performance", "efficiency", "welfare", "behavior", "behaviour",
        "health", "biosecurity", "quality", "productivity", "environmental",
    },
}

POULTRY_TECHNOLOGY_TERMS = {
    "Machine learning and AI": {
        "ai", "artificial_intelligence", "machine_learning", "deep_learning",
        "neural_networks", "artificial_neural_networks", "convolutional_neural_networks",
        "algorithm", "algorithms", "detection", "accuracy",
    },
    "Sensors and monitoring": {
        "sensor", "sensors", "monitoring", "camera", "cameras", "thermal",
        "image", "images", "imaging",
    },
    "Computer vision": {
        "computer_vision", "computer", "vision",
    },
    "Automation and robotics": {
        "automation", "automated", "robot", "robots", "robotics",
    },
    "Internet of Things": {
        "iot", "internet_of_things", "smart_technology", "smart_manufacturing",
        "smart_farming", "smart_agriculture",
    },
    "Digital transformation and PLF": {
        "digital_transformation", "precision_livestock_farming", "plf",
        "digital", "technology",
    },
    "Blockchain and supply chain": {
        "blockchain", "supply_chain", "value_chain", "food_chain", "production_chain",
    },
    "Digital twin": {
        "digital_twin",
    },
}


def get_best_topic_documents(best_method, lda_result, bnmf_result, best_topic):
    if best_method == "LDA":
        docs = lda_result["results_df"].copy()
        topic_col = "main_topic"
        prob_col = "topic_probability"
    else:
        docs = bnmf_result["results_df"].copy()
        topic_col = "topic_bnmf"
        prob_col = "topic_bnmf_prob_map"
    selected = docs[docs[topic_col] == best_topic].copy()
    selected["selected_method"] = best_method
    selected["selected_topic"] = best_topic
    selected["selected_topic_probability"] = selected[prob_col]
    return selected


def map_poultry_technologies(best_method, lda_result, bnmf_result, best_topic, results_dir: Path):
    """Map digital technologies detected in poultry-related records.

    The map is intentionally article-oriented: it connects the winning method's
    most preponderant/aligned topic with concrete technology families used in
    poultry, broiler, layer, flock, egg, welfare, feeding, housing, or mortality
    contexts. This prevents the technology map from being only a generic digital
    transformation inventory.
    """
    dirs = output_dirs(results_dir)
    best_dir = dirs["best"]
    img_dir = best_dir / "BEST_METHOD_IMAGES"
    map_dir = best_dir / "POULTRY_TECHNOLOGY_MAPPING"
    map_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    docs = get_best_topic_documents(best_method, lda_result, bnmf_result, best_topic)
    rows = []
    tech_counters = {
        tech: {"documents": 0, "term_hits": 0, "topic_probability_sum": 0.0, "matched_terms": Counter()}
        for tech in POULTRY_TECHNOLOGY_TERMS
    }
    poultry_terms = OBJECTIVE_GROUPS["poultry"]
    economic_terms = OBJECTIVE_GROUPS["economic"]

    for idx, row in docs.iterrows():
        text = " ".join(str(row.get(col, "")) for col in ["title", "abstract", "keywords"])
        normalized = apply_objective_phrase_replacements(text)
        tokens = tokenize_for_vocabulary(normalized)
        token_counter = Counter(tokens)
        poultry_hits = sum(token_counter.get(term, 0) for term in poultry_terms)
        economic_hits = sum(token_counter.get(term, 0) for term in economic_terms)
        detected = []
        matched_parts = []
        for tech, terms in POULTRY_TECHNOLOGY_TERMS.items():
            hits = {term: token_counter.get(term, 0) for term in terms if token_counter.get(term, 0) > 0}
            if hits and poultry_hits > 0:
                detected.append(tech)
                tech_counters[tech]["documents"] += 1
                tech_counters[tech]["term_hits"] += int(sum(hits.values()))
                tech_counters[tech]["topic_probability_sum"] += float(row.get("selected_topic_probability", 0) or 0)
                tech_counters[tech]["matched_terms"].update(hits)
                matched_parts.append(f"{tech}: {', '.join(sorted(hits))}")
        if detected:
            rows.append(
                {
                    "document_index": idx,
                    "title": row.get("title", ""),
                    "year": row.get("year", ""),
                    "journal": row.get("journal", ""),
                    "selected_method": best_method,
                    "selected_topic": best_topic,
                    "selected_topic_probability": row.get("selected_topic_probability", np.nan),
                    "poultry_context_hits": poultry_hits,
                    "economic_context_hits": economic_hits,
                    "technologies_detected": "; ".join(detected),
                    "matched_terms": " | ".join(matched_parts),
                }
            )

    summary_rows = []
    for tech, data in tech_counters.items():
        documents = data["documents"]
        if documents == 0:
            continue
        summary_rows.append(
            {
                "technology_family": tech,
                "documents": documents,
                "term_hits": data["term_hits"],
                "mean_topic_probability": data["topic_probability_sum"] / documents,
                "top_matched_terms": ", ".join(term for term, _ in data["matched_terms"].most_common(12)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["documents", "mean_topic_probability", "term_hits"],
        ascending=False,
    )
    doc_map = pd.DataFrame(rows)

    with pd.ExcelWriter(map_dir / "01_poultry_technology_mapping.xlsx") as writer:
        summary.to_excel(writer, sheet_name="Technology_Summary", index=False)
        doc_map.to_excel(writer, sheet_name="Documents_with_Technologies", index=False)
        pd.DataFrame(
            [{"technology_family": key, "terms": ", ".join(sorted(value))} for key, value in POULTRY_TECHNOLOGY_TERMS.items()]
        ).to_excel(writer, sheet_name="Technology_Dictionary", index=False)

    if MATPLOTLIB_AVAILABLE and not summary.empty:
        plot_df = summary.sort_values("documents", ascending=True)
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0.18, 0.82, len(plot_df)))
        bars = ax.barh(plot_df["technology_family"], plot_df["documents"], color=colors, edgecolor="#263238")
        ax.set_title(
            f"Poultry Technology Map - {best_method} Topic {best_topic}",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xlabel("Documents in poultry-related context")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        max_value = float(plot_df["documents"].max())
        for bar, prob in zip(bars, plot_df["mean_topic_probability"]):
            ax.text(
                bar.get_width() + max(0.5, max_value * 0.015),
                bar.get_y() + bar.get_height() / 2,
                f"p={prob:.2f}",
                va="center",
                fontsize=9,
            )
        legend_text = "Label p = mean topic probability among documents where the technology family was detected."
        ax.text(0.0, -0.12, legend_text, transform=ax.transAxes, fontsize=9, color="#374151")
        plt.tight_layout()
        out_png = map_dir / "01_poultry_technology_map.png"
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.savefig(img_dir / "09_poultry_technology_map.png", dpi=220, bbox_inches="tight")
        plt.close()

    print(f"Poultry technology mapping saved in: {map_dir}")
    return summary, doc_map


def classify_document_subthemes(text: str):
    tokens = tokenize_for_vocabulary(apply_objective_phrase_replacements(text))
    token_counter = Counter(tokens)
    scores = {}
    matched = {}
    for subtheme, terms in SUBTHEME_TERMS.items():
        hits = {term: token_counter[term] for term in terms if token_counter[term] > 0}
        scores[subtheme] = sum(hits.values())
        matched[subtheme] = ", ".join(sorted(hits))
    best_subtheme = max(scores, key=scores.get) if scores else ""
    if scores.get(best_subtheme, 0) == 0:
        best_subtheme = "8_other_related_terms"
    return best_subtheme, scores, matched, tokens


def analyze_best_topic_subthemes(best_method, lda_result, bnmf_result, best_topic, results_dir: Path):
    dirs = output_dirs(results_dir)
    best_dir = dirs["best"]
    img_dir = best_dir / "BEST_METHOD_IMAGES"
    subtheme_dir = best_dir / "PREPONDERANT_TOPIC_SUBTHEMES"
    subtheme_img_dir = subtheme_dir / "SUBTHEME_IMAGES"
    subtheme_dir.mkdir(parents=True, exist_ok=True)
    subtheme_img_dir.mkdir(parents=True, exist_ok=True)

    docs = get_best_topic_documents(best_method, lda_result, bnmf_result, best_topic)
    rows = []
    subtheme_word_counters = {name: Counter() for name in SUBTHEME_TERMS}
    subtheme_word_counters["8_other_related_terms"] = Counter()

    for idx, row in docs.iterrows():
        text = " ".join(str(row.get(col, "")) for col in ["title", "abstract", "keywords"])
        primary, scores, matched, tokens = classify_document_subthemes(text)
        subtheme_word_counters[primary].update(tokens)
        out = {
            "document_index": idx,
            "title": row.get("title", ""),
            "year": row.get("year", ""),
            "journal": row.get("journal", ""),
            "selected_method": best_method,
            "selected_topic": best_topic,
            "selected_topic_probability": row.get("selected_topic_probability", np.nan),
            "primary_subtheme": primary,
            "matched_terms_primary": matched.get(primary, ""),
        }
        for subtheme in list(SUBTHEME_TERMS) + ["8_other_related_terms"]:
            out[f"score_{subtheme}"] = scores.get(subtheme, 0)
            out[f"matched_{subtheme}"] = matched.get(subtheme, "")
        rows.append(out)

    doc_subthemes = pd.DataFrame(rows)
    if doc_subthemes.empty:
        summary = pd.DataFrame(columns=["subtheme", "documents", "percentage", "avg_topic_probability"])
    else:
        summary = (
            doc_subthemes.groupby("primary_subtheme")
            .agg(
                documents=("title", "count"),
                avg_topic_probability=("selected_topic_probability", "mean"),
            )
            .reset_index()
        )
        summary["percentage"] = (summary["documents"] / len(doc_subthemes) * 100).round(2)
        summary = summary.sort_values(["documents", "avg_topic_probability"], ascending=False)

    top_terms_rows = []
    for subtheme, counter in subtheme_word_counters.items():
        for rank, (word, freq) in enumerate(counter.most_common(30), 1):
            if word not in GENERIC_EXCLUDE_TERMS:
                top_terms_rows.append({
                    "subtheme": subtheme,
                    "rank": rank,
                    "word": word,
                    "frequency": freq,
                })
    top_terms = pd.DataFrame(top_terms_rows)

    with pd.ExcelWriter(subtheme_dir / "01_preponderant_topic_subthemes.xlsx") as writer:
        summary.to_excel(writer, sheet_name="Subtheme_Summary", index=False)
        doc_subthemes.to_excel(writer, sheet_name="Documents_Classified", index=False)
        top_terms.to_excel(writer, sheet_name="Top_Terms_by_Subtheme", index=False)
        pd.DataFrame(
            [{"subtheme": key, "terms": ", ".join(sorted(value))} for key, value in SUBTHEME_TERMS.items()]
        ).to_excel(writer, sheet_name="Subtheme_Dictionary", index=False)

    if MATPLOTLIB_AVAILABLE and not summary.empty:
        plot_df = summary.sort_values("documents", ascending=True)
        fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.55)))
        bars = ax.barh(plot_df["primary_subtheme"], plot_df["documents"], color="#4C78A8", edgecolor="black")
        ax.set_title(f"Subthemes within Most Preponderant Topic - {best_method} T{best_topic}", fontweight="bold")
        ax.set_xlabel("Documents")
        for bar, pct in zip(bars, plot_df["percentage"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(subtheme_img_dir / "01_preponderant_topic_subtheme_distribution.png", dpi=200, bbox_inches="tight")
        plt.savefig(img_dir / "08_preponderant_topic_subtheme_distribution.png", dpi=200, bbox_inches="tight")
        plt.close()

        score_cols = [col for col in doc_subthemes.columns if col.startswith("score_")]
        heatmap_data = doc_subthemes[score_cols].copy()
        heatmap_data.columns = [col.replace("score_", "") for col in score_cols]
        heatmap_data = heatmap_data.loc[:, (heatmap_data.sum(axis=0) > 0)]
        if not heatmap_data.empty:
            matrix = heatmap_data.head(80).values
            fig, ax = plt.subplots(figsize=(14, max(6, min(80, len(heatmap_data)) * 0.18)))
            im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
            ax.set_xticks(range(heatmap_data.shape[1]))
            ax.set_xticklabels(heatmap_data.columns, rotation=60, ha="right", fontsize=8)
            ax.set_yticks([])
            ax.set_title("Document x Subtheme Intensity Matrix", fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(subtheme_img_dir / "02_document_subtheme_matrix.png", dpi=200, bbox_inches="tight")
            plt.close()

    for subtheme, counter in subtheme_word_counters.items():
        filtered = [(word, freq) for word, freq in counter.most_common(80) if word not in GENERIC_EXCLUDE_TERMS]
        if not filtered:
            continue
        freq_df = pd.DataFrame(filtered, columns=["word", "frequency"])
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", subtheme)
        bnmf.create_round_wordcloud(
            freq_df,
            subtheme_img_dir / f"03_word_cloud_{safe_name}.png",
            title=subtheme.replace("_", " ").title(),
            dpi=300,
            size=1800,
        )

    print(f"Subtheme analysis saved in: {subtheme_dir}")
    return summary, doc_subthemes, top_terms


def export_article_ready_outputs(lda_result, bnmf_result, comparison_df, best_method, results_dir: Path):
    dirs = output_dirs(results_dir)
    common_dir = dirs["common"]
    best_dir = dirs["best"]
    common_img_dir = common_dir / "COMMON_IMAGES"
    best_img_dir = best_dir / "BEST_METHOD_IMAGES"
    common_img_dir.mkdir(parents=True, exist_ok=True)
    best_img_dir.mkdir(parents=True, exist_ok=True)

    lda_preponderance, bnmf_preponderance = build_topic_preponderance_tables(lda_result, bnmf_result)
    with pd.ExcelWriter(common_dir / "06_preponderant_topics_lda_bnmf.xlsx") as writer:
        lda_preponderance.to_excel(writer, sheet_name="LDA_Preponderance", index=False)
        bnmf_preponderance.to_excel(writer, sheet_name="BNMF_Preponderance", index=False)
        comparison_df.to_excel(writer, sheet_name="Method_Comparison", index=False)

    plot_topic_preponderance(
        lda_preponderance,
        "LDA",
        common_img_dir / "04_lda_preponderant_topics.png",
    )
    plot_topic_preponderance(
        bnmf_preponderance,
        NEW_METHOD_NAME,
        common_img_dir / "04_avitech_preponderant_topics.png",
    )

    files_to_copy = [
        (dirs["lda"] / "03_lda_topic_distribution_pie.png", common_img_dir / "03_lda_topic_distribution.png"),
        (dirs["lda"] / "IMAGES_LDA" / "02_lda_optimal_k.png", common_img_dir / "02_lda_optimal_k.png"),
        (dirs["bnmf"] / "04_bnmf_k_selection_plot.png", common_img_dir / "02_avitech_optimal_k.png"),
        (dirs["bnmf"] / "04_bnmf_topic_distribution.png", common_img_dir / "03_avitech_topic_distribution.png"),
        (common_dir / "05_method_comparison_scores.png", common_img_dir / "05_method_comparison_bar_chart.png"),
        (common_dir / "05_method_comparison_radar.png", common_img_dir / "05_method_comparison_radar.png"),
    ]
    for src, dst in files_to_copy:
        if src.exists():
            shutil.copy2(src, dst)

    best_topic = create_best_topic_wordcloud(best_method, lda_result, bnmf_result, results_dir)
    analyze_best_topic_subthemes(best_method, lda_result, bnmf_result, best_topic, results_dir)
    map_poultry_technologies(best_method, lda_result, bnmf_result, best_topic, results_dir)
    best_package_dir = export_best_method_named_package(best_method, lda_result, bnmf_result, comparison_df, best_topic, results_dir)

    manifest_rows = []
    for path in sorted(results_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".xlsx", ".csv", ".png", ".html", ".docx"}:
            manifest_rows.append(
                {
                    "file": str(path.relative_to(results_dir)),
                    "type": path.suffix.lower().lstrip("."),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                }
            )
    pd.DataFrame(manifest_rows).to_excel(common_dir / "00_generated_files_manifest.xlsx", index=False)

    print("\n=== ARTICLE-READY OUTPUTS ===")
    print(f"Best method: {best_method}")
    print(f"Most preponderant topic in best method: Topic {best_topic}")
    print(f"Common images saved in: {common_img_dir}")
    print(f"Best-method images saved in: {best_img_dir}")
    print(f"Best-method named package: {best_package_dir}")
    print(f"Generated-files manifest: {common_dir / '00_generated_files_manifest.xlsx'}")
    return lda_preponderance, bnmf_preponderance, best_topic


def compare_methods_and_plot(lda_result, bnmf_result, dtm, feature_names, results_dir: Path):
    dirs = output_dirs(results_dir)
    common_dir = dirs["common"]
    comparison_df = build_method_comparison(lda_result, bnmf_result, dtm, feature_names)
    comparison_df.to_excel(common_dir / "05_method_comparison_metrics.xlsx", index=False)
    alignment_tables = comparison_df.attrs.get("topic_alignment", {})
    if alignment_tables:
        with pd.ExcelWriter(common_dir / "05_topic_alignment_by_method.xlsx") as writer:
            for method, table in alignment_tables.items():
                sheet = re.sub(r"[^A-Za-z0-9_]", "_", method)[:31]
                table.to_excel(writer, sheet_name=sheet, index=False)
    plot_method_comparison(comparison_df, common_dir)
    plot_best_method_decision(comparison_df, common_dir / "05_best_method_decision.png")
    plot_alignment_comparison(alignment_tables, common_dir)

    best_method = comparison_df.iloc[0]["method"]
    plot_best_method_results(best_method, lda_result, bnmf_result, results_dir)

    print("\n=== METHOD COMPARISON ===")
    print(comparison_df[["method", "composite_score", "topic_coherence", "word_exclusivity", "topic_balance", "assignment_confidence"]])
    print(f"\nBest method by composite score: {best_method}")
    return comparison_df, best_method


def plot_alignment_comparison(alignment_tables, common_dir: Path):
    if not MATPLOTLIB_AVAILABLE or not alignment_tables:
        return

    rows = []
    for method, table in alignment_tables.items():
        temp = table.copy()
        temp["method"] = method
        rows.append(temp)
    df = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    methods = list(alignment_tables.keys())
    width = 0.35
    max_topics = int(df["topic"].max()) if not df.empty else 0
    x = np.arange(1, max_topics + 1)
    colors = ["#4C78A8", "#F58518"]
    for idx, method in enumerate(methods):
        subset = df[df["method"] == method].set_index("topic")
        values = subset["objective_alignment_score"].reindex(x).fillna(0).astype(float).values
        ax.bar(x + (idx - 0.5) * width, values, width, label=method, color=colors[idx % len(colors)])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Search objective alignment")
    ax.set_title("Topic Alignment with Poultry + Technology + Economic Feasibility", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(common_dir / "05_topic_alignment_with_research_objective.png", dpi=180, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run unified LDA + AviTech-Bayes bibliometric analysis for poultry technology and economic feasibility."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing .bib files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the same folder as --input-dir.",
    )
    parser.add_argument(
        "--resume-from-lda",
        action="store_true",
        help="Skip bibliometric/LDA regeneration and continue from existing LDA outputs to run BNMF, comparison and best-method package.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else input_dir
    )
    configure_modules(input_dir, results_dir)

    print("=" * 72)
    print("UNIFIED LDA + BATH-SEP-BNMF BIBLIOMETRIC ANALYSIS")
    print("Using the inherited LDA pipeline only for comparison with AviTech-Bayes")
    print("No Hoshin Kanri bibliometric or strategic-analysis module is executed")
    print("=" * 72)

    database = load_and_prepare_database(input_dir, results_dir)
    if args.resume_from_lda:
        print("\n=== RESUME MODE: USING EXISTING LDA OUTPUTS ===")
        processed_df, processed_tokens, word_freq, dtm, vectorizer, feature_names = preprocess_for_topic_models(
            database,
            results_dir,
            create_images=False,
        )
        lda_result = load_existing_lda_result(results_dir, feature_names)
    else:
        export_vosviewer_files(database, results_dir)
        create_bibliometric_tables_and_images(database, results_dir)
        create_advanced_bibliometric_networks(database, results_dir)
        adapt_vocabulary_to_bibtex(database, results_dir)
        processed_df, processed_tokens, word_freq, dtm, vectorizer, feature_names = preprocess_for_topic_models(
            database,
            results_dir,
        )
        lda_result = run_lda_pipeline(processed_df, processed_tokens, word_freq, dtm, vectorizer, feature_names, results_dir)
    bnmf_result = run_bnmf_pipeline(lda_result["results_df"], dtm, feature_names, results_dir)
    sync_legacy_bath_sep_results(results_dir)
    comparison_df, best_method = compare_methods_and_plot(
        lda_result,
        bnmf_result,
        dtm,
        feature_names,
        results_dir,
    )

    if not args.resume_from_lda:
        print("\n=== BIBLIOMETRIC ANALYSIS FOR THE TOPIC-MODELING CORPUS ===")
        lda.run_bibliographic_analysis(bnmf_result["results_df"])
    export_combined_summary(lda_result, bnmf_result, results_dir)
    lda_preponderance, bnmf_preponderance, best_topic = export_article_ready_outputs(
        lda_result,
        bnmf_result,
        comparison_df,
        best_method,
        results_dir,
    )

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETED")
    print(f"LDA K: {lda_result['best_k']}")
    print(f"BNMF K: {bnmf_result['best_k']}")
    print(f"Best method: {best_method}")
    print(f"Most preponderant topic in best method: Topic {best_topic}")
    dirs = output_dirs(results_dir)
    print(f"Bibliometric results: {dirs['bibliometric']}")
    print(f"LDA results: {dirs['lda']}")
    print(f"BNMF results: {dirs['bnmf']}")
    print(f"Best-method results: {dirs['best']}")
    print(f"Common results: {dirs['common']}")
    print(f"Documents analyzed: {len(lda_result['results_df'])}")
    print(f"Results saved in: {results_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
