#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V9.12 automatic full-text economic-value extraction for the no-expert
multicriteria poultry technology manuscript.

The script is deliberately conservative. It extracts candidate values from the
30 full-text PDFs only when a numeric value appears inside a local context that
also contains economic, operational-efficiency, productivity, ROI, payback,
cost, investment, or savings language. The outputs are therefore suitable as a
computational screening layer, not as manually audited financial evidence.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import pdfplumber
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pdfplumber is required for PDF text extraction.") from exc

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.shared import Cm, Pt
    from docx.text.paragraph import Paragraph
except ImportError as exc:  # pragma: no cover
    raise SystemExit("python-docx is required for manuscript updating.") from exc


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "full_texts_for_multicriteria"
EXTRACTION_DIR = BASE_DIR / "8_full_text_economic_extraction_30_pdfs"
MCDM_TABLE_DIR = BASE_DIR / "6_multicriteria" / "without_expert_topic_weights" / "TABLES"
SOURCE_DOCX = BASE_DIR / "V9.11 _Art 2 _ Manuscript_Heliyon_v71_figure6_restored.docx"
REFERENCE_DOCX = BASE_DIR / "V9.7 _Art 2 _ Manuscript_Heliyon_equations_in_methods.docx"
OUTPUT_DOCX = BASE_DIR / "V9.12 _Art 2 _ Manuscript_Heliyon_auto_economic_values.docx"
OUTPUT_PDF = BASE_DIR / "V9.12 _Art 2 _ Manuscript_Heliyon_auto_economic_values.pdf"


TECH_PATTERNS = {
    "sensors_monitoring": (
        "Sensors and monitoring",
        [
            r"\bsensors?\b", r"\bwearable\b", r"\bmonitoring\b",
            r"\benvironmental monitoring\b", r"\bthermal monitoring\b",
            r"\bsmart measurement", r"\bdata logger\b",
        ],
    ),
    "automation_robotics": (
        "Automation and robotics",
        [
            r"\bautomation\b", r"\bautomated\b", r"\bautomatic\b",
            r"\brobot(?:ic|ics|s)?\b", r"\bactuator\b", r"\bcontrolled system\b",
        ],
    ),
    "machine_learning_ai": (
        "Machine learning / AI",
        [
            r"\bmachine learning\b", r"\bartificial intelligence\b", r"\bAI\b",
            r"\bdeep learning\b", r"\bneural network", r"\brandom forest\b",
            r"\bXGBoost\b", r"\bSVM\b", r"\bclassification model\b",
        ],
    ),
    "iot": (
        "Internet of Things (IoT)",
        [
            r"\binternet of things\b", r"\bIoT\b", r"\bconnected sensor",
            r"\bwireless sensor", r"\bcloud\b", r"\bedge computing\b",
        ],
    ),
    "computer_vision": (
        "Computer vision",
        [
            r"\bcomputer vision\b", r"\bimage processing\b", r"\bimage analysis\b",
            r"\bcamera\b", r"\bRGB\b", r"\bdepth camera\b", r"\bYOLO\b",
            r"\bobject detection\b", r"\bvideo\b",
        ],
    ),
    "blockchain_supply_chain": (
        "Blockchain and supply chain",
        [
            r"\bblockchain\b", r"\bsupply chain\b", r"\btraceability\b",
            r"\btransparency\b", r"\bprovenance\b",
        ],
    ),
    "digital_twin": (
        "Digital twin",
        [
            r"\bdigital twin\b", r"\bsimulation\b", r"\bsimulated\b",
            r"\bdynamic model\b", r"\bvirtual model\b",
        ],
    ),
}

ECONOMIC_TERMS = [
    "cost", "costs", "low-cost", "low cost", "affordable", "affordability",
    "economic", "economics", "financial", "feasibility", "viability",
    "investment", "capital", "capex", "opex", "operating expense",
    "operational expense", "maintenance", "ROI", "return on investment",
    "payback", "return", "profit", "profitability", "revenue",
    "cost-benefit", "cost benefit", "cost-effective", "cost effective",
    "savings", "saving", "expense", "expenses", "expenditure",
    "implementation cost", "adoption cost",
]

OPERATIONAL_TERMS = [
    "productivity", "productivity gain", "performance gain", "efficiency",
    "operational efficiency", "labor", "labour", "energy", "water",
    "feed conversion", "FCR", "mortality", "resource", "resources",
    "welfare", "growth", "weight gain", "production indicator",
]

COUNTRY_TERMS = [
    "Brazil", "United States", "USA", "United Kingdom", "China", "India",
    "Indonesia", "Thailand", "Turkey", "Germany", "Netherlands", "Italy",
    "Spain", "France", "Canada", "Australia", "Ethiopia", "South Africa",
    "Pakistan", "Malaysia", "Vietnam", "Philippines", "Bangladesh",
]

VALUE_PATTERNS = [
    ("currency", re.compile(
        r"(?i)(?:US\$|U\.S\.\$|USD|EUR|€|£|GBP|R\$|BRL|\$)\s*"
        r"\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand|bn)|[kK])?"
    )),
    ("percentage", re.compile(r"\b\d{1,3}(?:[.,]\d+)?\s*%")),
    ("payback_period", re.compile(
        r"(?i)\b(?:payback|return period|break-even|breakeven).{0,90}?"
        r"\d+(?:[.,]\d+)?\s*(?:year|years|yr|yrs|month|months|mo|mos)\b"
    )),
    ("roi_return", re.compile(
        r"(?i)\b(?:return on investment|economic return).{0,100}?"
        r"(?:\d{1,3}(?:[.,]\d+)?\s*%|\d+(?:[.,]\d+)?)"
    )),
    ("cost_savings_phrase", re.compile(
        r"(?i)\b(?:cost reduction|cost reductions|reduced cost|lower cost|"
        r"less operational expenses|savings|saving|reduced expenses|"
        r"operational expenses|implementation cost|investment cost|low-cost|low cost)"
        r".{0,120}?(?:\d{1,3}(?:[.,]\d+)?\s*%|"
        r"(?:US\$|U\.S\.\$|USD|EUR|€|£|GBP|R\$|BRL|\$)\s*\d[\d,]*(?:\.\d+)?)"
    )),
    ("productivity_efficiency_phrase", re.compile(
        r"(?i)\b(?:productivity gains?|performance gains?|efficiency gains?|"
        r"reduced mortality|lower mortality|feed conversion|FCR|energy use|water use)"
        r".{0,120}?\d{1,3}(?:[.,]\d+)?\s*%"
    )),
]

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


@dataclass
class CandidateValue:
    source_file: str
    article_title: str
    doi: str
    year: str
    journal: str
    country_context: str
    technology_keys: list[str]
    technology_labels: list[str]
    indicator_type: str
    extracted_value: str
    context: str
    confidence: float
    confidence_label: str
    interpretation: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_extracted_value(value: str) -> str:
    value = clean_text(value)
    value = value.strip(" ;:,.")
    value = re.sub(r"(?i)\bUSD\s+(\d+),$", r"USD \1", value)
    value = re.sub(r"(?i)\bR\$\s*(\d[\d,]*)\s*,?$", r"R$\1", value)
    value = re.sub(r"\$\s*(\d[\d,]*)\s*,?$", r"$\1", value)
    return value


def currency_amount(value: str) -> float | None:
    match = re.search(r"(?:US\$|U\.S\.\$|USD|EUR|€|£|GBP|R\$|BRL|\$)\s*(\d[\d,]*(?:\.\d+)?)", value, re.I)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_pdfs(pdf_dir: Path) -> list[Path]:
    seen = {}
    for path in sorted(pdf_dir.glob("*.pdf")):
        digest = file_md5(path)
        if digest not in seen:
            seen[digest] = path
    return list(seen.values())


def extract_pdf_text(path: Path) -> str:
    parts = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
    except Exception as exc:
        print(f"[WARN] Could not extract {path.name}: {exc}", file=sys.stderr)
    return clean_text(" ".join(parts))


def load_existing_article_metadata() -> dict[str, dict[str, str]]:
    meta = {}
    for name in [
        "publication_ready_full_text_economic_extraction_30_articles.xlsx",
        "cleaned_full_text_economic_evidence_30_articles.xlsx",
    ]:
        path = EXTRACTION_DIR / name
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        for _, row in df.iterrows():
            doi = str(row.get("DOI", "") or "").strip().rstrip(".").lower()
            if doi:
                meta[doi] = {
                    "title": str(row.get("Article title", "") or "").strip(),
                    "year": str(row.get("Year", "") or "").strip(),
                    "journal": str(row.get("Journal / source", "") or "").strip(),
                }
    return meta


def detect_doi(text: str) -> str:
    matches = DOI_RE.findall(text)
    if not matches:
        return ""
    doi = matches[0].strip().rstrip(").,;").lower()
    return doi


def infer_title(text: str, path: Path, doi: str, metadata: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    if doi.lower() in metadata:
        item = metadata[doi.lower()]
        return item.get("title", "") or path.stem, item.get("year", ""), item.get("journal", "")
    year = ""
    year_match = re.search(r"\b(202[2-6])\b", text[:5000])
    if year_match:
        year = year_match.group(1)
    candidate = ""
    first = text[:2200]
    chunks = [c.strip(" .") for c in re.split(r"(?<=[.!?])\s+|\s{3,}", first) if len(c.strip()) > 35]
    if chunks:
        candidate = max(chunks[:8], key=len)[:220]
    return candidate or path.stem, year, ""


def find_countries(text: str) -> str:
    found = []
    for country in COUNTRY_TERMS:
        if re.search(rf"\b{re.escape(country)}\b", text, re.I):
            found.append("United States" if country == "USA" else country)
    return "; ".join(sorted(set(found)))[:220]


def detect_technologies(text: str) -> tuple[list[str], list[str]]:
    keys = []
    labels = []
    for key, (label, patterns) in TECH_PATTERNS.items():
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            keys.append(key)
            labels.append(label)
    return keys, labels


def has_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def context_window(text: str, start: int, end: int, size: int = 380) -> str:
    left = max(0, start - size)
    right = min(len(text), end + size)
    return clean_text(text[left:right])


def reject_numeric_noise(value: str, context: str, value_type: str) -> bool:
    v = value.strip()
    c = context.lower()
    if value_type == "currency":
        amount = currency_amount(v)
        if amount is not None and amount < 10:
            if not re.search(r"(?i)\b(?:per day|per bird|per cycle|price|service|storage|cloud|subscription)\b", context):
                return True
        return False
    if re.fullmatch(r"20[0-2]\d", v) or re.fullmatch(r"202[2-6]", v):
        return True
    if value_type == "roi_return" and re.search(r"\b20[0-2]\d\b", v) and not re.search(r"[%$€£]|USD|EUR|BRL|R\$", v, re.I):
        return True
    if value_type == "roi_return":
        economic_roi_context = re.search(
            r"(?i)\b(?:return on investment|economic return|investment|payback|profit|profitability|financial|economic|cost|savings?)\b",
            context,
        )
        computer_vision_roi_context = re.search(
            r"(?i)\b(?:region of interest|ROI extraction|ROI,|ROI\s*\(|YOLO|image|camera|pixel|"
            r"segmentation|object detection|feature maps?|bounding box|scan detection|chicken claw region|red color)\b",
            context,
        )
        if computer_vision_roi_context and not economic_roi_context:
            return True
        if re.search(r"(?i)\bROI\b", v) and not economic_roi_context:
            return True
    if value_type == "cost_savings_phrase":
        if re.search(r"(?i)\b(?:saving accuracy|data saving|model saving|file saving|saved image|saved video)\b", context):
            if not re.search(r"(?i)\b(?:cost|expense|expenses|expenditure|low-cost|low cost|operational|economic|financial)\b", context):
                return True
    if value_type == "productivity_efficiency_phrase":
        if re.search(r"(?i)\b(?:did not differ significantly|p\s*[=<>]|p-value|statistically significant)\b", context):
            return True
        if not re.search(
            r"(?i)\b(?:productivity gains?|performance gains?|efficiency gains?|improved|improvement|"
            r"reduced|reduction|lower|less|cost reductions?|savings?)\b",
            context,
        ):
            return True
    if value_type in {"percentage", "roi_return"}:
        statistical_noise = re.search(
            r"(?i)\b(?:confidence interval|95\s*%\s*CI|\bCI\b|p\s*[<=>]|"
            r"r\s*=|correlation|coefficient|accuracy|precision|recall|"
            r"specificity|sensitivity|AUC|F1-score|R2|RMSE|MAE)\b",
            context,
        )
        direct_economic_or_operational = re.search(
            r"(?i)\b(?:cost reduction|cost savings?|savings?|less operational expenses|"
            r"lower cost|reduced cost|ROI|return on investment|economic return|payback|"
            r"profitability|productivity gains?|efficiency gains?|reduced mortality|"
            r"energy use|water use|resource use|operating expenses?)\b",
            context,
        )
        if statistical_noise and not direct_economic_or_operational:
            return True
        survey_noise = re.search(
            r"(?i)\b(?:survey|respondents?|responses?|answered|agree|disagree|Likert|questionnaire|completion rate|"
            r"participants?|producers indicated|indicated that)\b",
            context,
        )
        if survey_noise:
            return True
    if "doi" in c or "https://doi.org" in c or "sciencedirect" in c:
        if not (has_any(context, ECONOMIC_TERMS) or has_any(context, OPERATIONAL_TERMS)):
            return True
    if re.search(r"\b(?:vol|volume|issue|page|pages|pp\.|fig|figure|table)\b", c):
        if not (has_any(context, ECONOMIC_TERMS) or has_any(context, OPERATIONAL_TERMS)):
            return True
    return False


def classify_indicator(value_type: str, context: str) -> tuple[str, float, str, str]:
    c = context.lower()
    if value_type == "currency":
        if any(x in c for x in ["investment", "capital", "capex", "equipment", "implementation"]):
            return "Investment/CAPEX value", 0.92, "High", "Monetary value associated with investment or implementation context."
        if any(x in c for x in ["cost", "low-cost", "low cost", "expense", "maintenance"]):
            return "Cost/OPEX value", 0.90, "High", "Monetary value associated with cost or operating-expense context."
        return "Monetary value", 0.82, "High", "Monetary value extracted from a relevant economic context."
    if value_type == "payback_period":
        return "Payback/break-even period", 0.93, "High", "Numeric value directly linked to payback or break-even language."
    if value_type == "roi_return":
        return "ROI/return value", 0.90, "High", "Numeric value directly linked to return, ROI, or profitability language."
    if value_type == "cost_savings_phrase":
        return "Cost saving or expense-reduction value", 0.88, "High", "Numeric value linked to cost saving or expense reduction."
    if value_type == "productivity_efficiency_phrase":
        return "Productivity/efficiency value", 0.84, "High", "Numeric value linked to productivity, resource efficiency, or performance gain."
    if value_type == "percentage" and any(x in c for x in ["roi", "payback", "return on investment", "economic return"]):
        return "ROI/return-related numeric signal", 0.72, "Medium", "Percentage appears near return or payback language but requires verification."
    if value_type != "percentage" and any(x in c for x in ["roi", "payback", "return on investment", "economic return"]):
        return "ROI/return-related numeric signal", 0.80, "High", "Numeric signal appears near explicit return or payback language."
    if any(x in c for x in ["saving", "savings", "cost reduction", "less operational", "lower cost", "reduced cost"]):
        return "Savings/cost-reduction numeric signal", 0.78, "Medium", "Numeric signal appears near savings or cost-reduction language."
    if any(x in c for x in ["productivity", "efficiency", "mortality", "feed conversion", "energy", "water"]):
        return "Operational performance numeric signal", 0.72, "Medium", "Numeric signal appears near operational-performance language."
    if has_any(context, ECONOMIC_TERMS):
        return "General economic numeric signal", 0.62, "Medium", "Numeric signal appears near economic language but requires verification."
    return "Ambiguous numeric signal", 0.35, "Low", "Numeric signal is too ambiguous for direct interpretation."


def extract_candidates_from_text(path: Path, text: str, metadata: dict[str, dict[str, str]]) -> list[CandidateValue]:
    doi = detect_doi(text)
    title, year, journal = infer_title(text, path, doi, metadata)
    country = find_countries(text)
    doc_keys, doc_labels = detect_technologies(text)
    rows: list[CandidateValue] = []
    seen = set()
    for value_type, pattern in VALUE_PATTERNS:
        for match in pattern.finditer(text):
            value = clean_extracted_value(match.group(0))[:120]
            context = context_window(text, match.start(), match.end())
            near = context_window(text, match.start(), match.end(), size=170)
            if not (has_any(near, ECONOMIC_TERMS) or has_any(near, OPERATIONAL_TERMS)):
                continue
            if reject_numeric_noise(value, context, value_type):
                continue
            context_keys, context_labels = detect_technologies(context)
            keys = context_keys or doc_keys
            labels = context_labels or doc_labels
            if not keys:
                continue
            indicator, confidence, label, interpretation = classify_indicator(value_type, near)
            key = (path.name, value.lower(), indicator.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append(CandidateValue(
                source_file=path.name,
                article_title=title,
                doi=doi,
                year=year,
                journal=journal,
                country_context=country,
                technology_keys=keys,
                technology_labels=labels,
                indicator_type=indicator,
                extracted_value=value,
                context=context,
                confidence=confidence,
                confidence_label=label,
                interpretation=interpretation,
            ))
    rows.sort(key=lambda r: (-r.confidence, r.source_file, r.indicator_type))
    return rows


def candidates_to_frame(rows: list[CandidateValue]) -> pd.DataFrame:
    data = []
    for idx, row in enumerate(rows, 1):
        data.append({
            "rank": idx,
            "source_file": row.source_file,
            "article_title": row.article_title,
            "year": row.year,
            "journal": row.journal,
            "doi": row.doi,
            "country_or_empirical_context_detected": row.country_context,
            "technology_keys": "; ".join(row.technology_keys),
            "technology_families": "; ".join(row.technology_labels),
            "indicator_type": row.indicator_type,
            "extracted_candidate_value": row.extracted_value,
            "confidence_score": row.confidence,
            "confidence_label": row.confidence_label,
            "interpretation": row.interpretation,
            "context_snippet_for_manual_check": row.context[:700],
            "use_in_manuscript": "Yes" if row.confidence >= 0.80 else "Supplementary screening only",
            "methodological_note": (
                "Automatically extracted candidate value. Use as economic-evidence screening; "
                "verify manually before reporting as a definitive financial value."
            ),
        })
    return pd.DataFrame(data)


def build_article_summary(candidate_df: pd.DataFrame, pdfs: list[Path], texts: dict[str, str]) -> pd.DataFrame:
    rows = []
    for pdf in pdfs:
        work = candidate_df[candidate_df["source_file"] == pdf.name] if not candidate_df.empty else pd.DataFrame()
        text = texts.get(pdf.name, "")
        tech_keys, tech_labels = detect_technologies(text)
        rows.append({
            "source_file": pdf.name,
            "candidate_values": int(len(work)),
            "high_confidence_values": int((work["confidence_score"] >= 0.80).sum()) if not work.empty else 0,
            "medium_confidence_values": int(((work["confidence_score"] >= 0.60) & (work["confidence_score"] < 0.80)).sum()) if not work.empty else 0,
            "currency_values": int((work["indicator_type"].str.contains("value", case=False, na=False) & work["extracted_candidate_value"].str.contains(r"\$|USD|EUR|€|BRL|R\$", regex=True, na=False)).sum()) if not work.empty else 0,
            "roi_payback_values": int(work["indicator_type"].str.contains("ROI|Payback|return", case=False, regex=True, na=False).sum()) if not work.empty else 0,
            "savings_or_cost_values": int(work["indicator_type"].str.contains("Cost|saving|expense|OPEX|CAPEX", case=False, regex=True, na=False).sum()) if not work.empty else 0,
            "productivity_or_efficiency_values": int(work["indicator_type"].str.contains("Productivity|efficiency|performance", case=False, regex=True, na=False).sum()) if not work.empty else 0,
            "technology_families_detected": "; ".join(tech_labels),
        })
    return pd.DataFrame(rows).sort_values(["high_confidence_values", "candidate_values"], ascending=False).reset_index(drop=True)


def build_technology_summary(candidate_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    if candidate_df.empty:
        return pd.DataFrame()
    for _, row in candidate_df.iterrows():
        keys = [x.strip() for x in str(row["technology_keys"]).split(";") if x.strip()]
        labels = [x.strip() for x in str(row["technology_families"]).split(";") if x.strip()]
        for key, label in zip(keys, labels):
            item = row.to_dict()
            item["technology"] = key
            item["technology_label"] = label
            records.append(item)
    expanded = pd.DataFrame(records)
    grouped = []
    for (key, label), g in expanded.groupby(["technology", "technology_label"], dropna=False):
        high = g[g["confidence_score"] >= 0.80]
        articles = g["source_file"].nunique()
        value_support = (
            high["confidence_score"].sum()
            + 0.50 * g[(g["confidence_score"] >= 0.60) & (g["confidence_score"] < 0.80)]["confidence_score"].sum()
        )
        grouped.append({
            "technology": key,
            "technology_label": label,
            "articles_with_candidate_values": int(articles),
            "candidate_values": int(len(g)),
            "high_confidence_values": int(len(high)),
            "mean_confidence": float(g["confidence_score"].mean()),
            "currency_values": int(g["extracted_candidate_value"].str.contains(r"\$|USD|EUR|€|BRL|R\$", regex=True, na=False).sum()),
            "roi_payback_values": int(g["indicator_type"].str.contains("ROI|Payback|return", case=False, regex=True, na=False).sum()),
            "savings_or_cost_values": int(g["indicator_type"].str.contains("Cost|saving|expense|OPEX|CAPEX", case=False, regex=True, na=False).sum()),
            "productivity_or_efficiency_values": int(g["indicator_type"].str.contains("Productivity|efficiency|performance", case=False, regex=True, na=False).sum()),
            "full_text_value_support_score": round(float(value_support), 4),
            "interpretation": (
                "Higher score means stronger machine-extracted full-text support for economic or operational value signals. "
                "It is not a confirmed monetary performance score."
            ),
        })
    out = pd.DataFrame(grouped)
    return out.sort_values(["full_text_value_support_score", "high_confidence_values"], ascending=False).reset_index(drop=True)


def build_enhanced_mcdm_matrix(tech_summary: pd.DataFrame) -> pd.DataFrame:
    matrix_path = MCDM_TABLE_DIR / "07_no_expert_technology_decision_matrix.xlsx"
    if not matrix_path.exists():
        return tech_summary.copy()
    matrix = pd.read_excel(matrix_path)
    enhanced = matrix.merge(tech_summary, how="left", on=["technology", "technology_label"])
    value_cols = [
        "articles_with_candidate_values", "candidate_values", "high_confidence_values",
        "currency_values", "roi_payback_values", "savings_or_cost_values",
        "productivity_or_efficiency_values", "full_text_value_support_score",
    ]
    for col in value_cols:
        if col in enhanced.columns:
            enhanced[col] = enhanced[col].fillna(0)
    if "full_text_value_support_score" in enhanced.columns:
        max_score = enhanced["full_text_value_support_score"].max()
        enhanced["normalized_full_text_value_support"] = (
            enhanced["full_text_value_support_score"] / max_score if max_score > 0 else 0
        )
    enhanced["v9_12_interpretation"] = (
        "The original decision matrix remains literature-driven from the aligned topic. "
        "The added full-text columns provide automatic support evidence from the 30 PDFs only."
    )
    return enhanced


def save_outputs(candidate_df: pd.DataFrame, article_summary: pd.DataFrame, tech_summary: pd.DataFrame,
                 enhanced_matrix: pd.DataFrame, pdfs: list[Path]) -> None:
    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    main_xlsx = EXTRACTION_DIR / "V9.12_auto_extracted_economic_values_30_pdfs.xlsx"
    with pd.ExcelWriter(main_xlsx) as writer:
        candidate_df.to_excel(writer, sheet_name="Candidate_Economic_Values", index=False)
        candidate_df[candidate_df["confidence_score"] >= 0.80].to_excel(writer, sheet_name="High_Confidence_Values", index=False)
        article_summary.to_excel(writer, sheet_name="Article_Summary", index=False)
        tech_summary.to_excel(writer, sheet_name="Technology_Summary", index=False)
        enhanced_matrix.to_excel(writer, sheet_name="Enhanced_MCDM_Matrix", index=False)
        pd.DataFrame({
            "item": ["PDF files read", "Unique PDF files after MD5 deduplication", "Extraction interpretation"],
            "value": [
                len(list(PDF_DIR.glob("*.pdf"))),
                len(pdfs),
                "Candidate economic values extracted automatically from local full-text context; manual verification is required before reporting exact financial values.",
            ],
        }).to_excel(writer, sheet_name="Methodological_Note", index=False)
    candidate_df.to_csv(EXTRACTION_DIR / "V9.12_auto_extracted_economic_values_30_pdfs.csv", index=False, encoding="utf-8-sig")
    tech_summary.to_excel(EXTRACTION_DIR / "V9.12_value_support_by_technology.xlsx", index=False)
    enhanced_matrix.to_excel(MCDM_TABLE_DIR / "31_V9.12_enhanced_decision_matrix_with_full_text_value_support.xlsx", index=False)
    note = (
        "V9.12 automatic full-text economic-value extraction\n\n"
        f"PDF files in folder: {len(list(PDF_DIR.glob('*.pdf')))}\n"
        f"Unique PDFs processed: {len(pdfs)}\n"
        f"Candidate values extracted: {len(candidate_df)}\n"
        f"High-confidence candidate values: {int((candidate_df['confidence_score'] >= 0.80).sum()) if not candidate_df.empty else 0}\n\n"
        "Interpretation: these are candidate economic or operational-value signals. "
        "They strengthen the multicriteria framework as a reproducible screening layer, "
        "but they are not manually confirmed farm-level feasibility values.\n"
    )
    (EXTRACTION_DIR / "V9.12_methodological_note_auto_extraction.txt").write_text(note, encoding="utf-8")


def replace_text_in_paragraph(paragraph, old: str, new: str) -> bool:
    if old not in paragraph.text:
        return False
    inline = paragraph.runs
    full = paragraph.text.replace(old, new)
    for run in inline:
        run.text = ""
    if inline:
        inline[0].text = full
    else:
        paragraph.add_run(full)
    return True


def add_paragraph_after(paragraph, text: str, bold_lead: str | None = None):
    new_element = OxmlElement("w:p")
    paragraph._p.addnext(new_element)
    new_p = Paragraph(new_element, paragraph._parent)
    if bold_lead and text.startswith(bold_lead):
        run = new_p.add_run(bold_lead)
        run.bold = True
        new_p.add_run(text[len(bold_lead):])
    else:
        new_p.add_run(text)
    new_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return new_p


def insert_after_index(doc: Document, idx: int, texts: list[str]) -> None:
    anchor = doc.paragraphs[idx]
    current = anchor
    for text in texts:
        current = add_paragraph_after(current, text)


def delete_paragraph(paragraph) -> None:
    element = paragraph._element
    parent = element.getparent()
    if parent is not None:
        parent.remove(element)
    paragraph._p = paragraph._element = None


def update_table5(doc: Document, candidate_df: pd.DataFrame, article_summary: pd.DataFrame, tech_summary: pd.DataFrame) -> None:
    # Table 5 is the table with the "Dimension / Evidence item / Articles" header.
    target = None
    for table in doc.tables:
        if table.rows and len(table.rows[0].cells) >= 4:
            header = [c.text.strip() for c in table.rows[0].cells[:4]]
            if header == ["Dimension", "Evidence item", "Articles", "Interpretation for the literature-driven framework"]:
                target = table
                break
    if target is None:
        return
    while len(target.rows) > 1:
        target._tbl.remove(target.rows[-1]._tr)
    high = candidate_df[candidate_df["confidence_score"] >= 0.80] if not candidate_df.empty else pd.DataFrame()
    rows = [
        [
            "Automatic full-text value extraction",
            "High-confidence candidate economic or operational values",
            str(len(high)),
            "Values were retained only when a numeric value appeared close to economic, savings, ROI/payback, cost, investment, productivity, or efficiency language.",
        ],
        [
            "Automatic full-text value extraction",
            "Articles with at least one high-confidence value",
            str(high["source_file"].nunique() if not high.empty else 0),
            "This count indicates how often the 30-PDF corpus reported extractable candidate values rather than generic feasibility language.",
        ],
        [
            "Technology support",
            "Technology families with high-confidence value support",
            str((tech_summary["high_confidence_values"] > 0).sum() if not tech_summary.empty else 0),
            "The value-support layer was used to interpret the multicriteria ranking, not to replace the original topic-derived weights.",
        ],
        [
            "Interpretation boundary",
            "Confirmed farm-level financial feasibility",
            "Not claimed",
            "The values are machine-extracted candidates. Exact ROI, payback, CAPEX, OPEX, country, and scale interpretation requires manual verification and farm-level data.",
        ],
    ]
    for row_values in rows:
        row = target.add_row()
        for cell, value in zip(row.cells, row_values):
            cell.text = value
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10)


def update_manuscript(candidate_df: pd.DataFrame, article_summary: pd.DataFrame, tech_summary: pd.DataFrame) -> None:
    source = SOURCE_DOCX if SOURCE_DOCX.exists() else REFERENCE_DOCX
    shutil.copy2(source, OUTPUT_DOCX)
    doc = Document(OUTPUT_DOCX)
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
    high = candidate_df[candidate_df["confidence_score"] >= 0.80] if not candidate_df.empty else pd.DataFrame()
    n_articles_high = high["source_file"].nunique() if not high.empty else 0
    n_values = len(candidate_df)
    n_high = len(high)
    n_unique = len(article_summary)
    top_tech = "; ".join(tech_summary.head(4)["technology_label"].tolist()) if not tech_summary.empty else "the leading technology families"

    abstract_old = "30 full-text articles were examined for cost, investment, ROI/payback, savings, and implementation signals."
    abstract_new = (
        "30 full-text articles were reprocessed through a conservative automatic extraction layer to identify "
        "candidate cost, investment, ROI/payback, savings, productivity, and implementation values."
    )
    for p in doc.paragraphs:
        replace_text_in_paragraph(p, abstract_old, abstract_new)

    method_extra = (
        "For V9.12, the 30 full-text PDFs were additionally processed through a confidence-scored automatic value-extraction routine. "
        "The routine first deduplicated files by MD5 hash, extracted full text, detected deployable technology families, and retained numeric values only when they appeared within a local context containing economic, cost, investment, ROI, payback, savings, productivity, efficiency, implementation, or operating-expense language. "
        "Each candidate value was classified as monetary, CAPEX/OPEX, ROI/return, payback, savings/cost reduction, or productivity/efficiency evidence, and then assigned a high, medium, or low confidence label. "
        "Only high-confidence values were summarized in the manuscript; medium- and low-confidence values were retained in the supplementary spreadsheet for traceability. "
        "This step improves the multicriteria interpretation by adding a full-text economic-evidence layer, but it does not transform the ranking into a final farm-level financial feasibility study."
    )

    method_anchor = None
    result_anchor = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.startswith("Economic-evidence mining was then applied"):
            method_anchor = i
        if p.text.startswith("The extraction found cost-related signals"):
            result_anchor = i
    if method_anchor is not None:
        insert_after_index(doc, method_anchor, [method_extra])
    if result_anchor is not None:
        doc.paragraphs[result_anchor].text = (
            f"The confidence-scored extraction processed {n_unique} unique full-text PDFs and identified {n_values} candidate numeric economic or operational-value signals. "
            f"Of these, {n_high} were classified as high confidence across {n_articles_high} articles because the numeric value appeared in a local context explicitly linked to cost, investment, savings, productivity, efficiency, implementation, or operating expenses. "
            f"The strongest full-text value support was concentrated in {top_tech}. "
            "The retained high-confidence signals were mainly monetary cost or investment values, cost-reduction evidence, and productivity or efficiency gains. "
            "ROI/payback language was treated conservatively because ROI can also mean region of interest in computer-vision articles. "
            "The table should therefore be read as a screening instrument for the manuscript, while exact monetary values still require manual verification before final reporting."
        )
    for p in list(doc.paragraphs):
        if p.text.strip().startswith("The extraction found cost-related signals in 27 of the 30 articles"):
            delete_paragraph(p)
    update_table5(doc, candidate_df, article_summary, tech_summary)
    for p in doc.paragraphs:
        if p.text.strip():
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            for run in p.runs:
                run.font.name = "Times New Roman"
                if not run.font.size:
                    run.font.size = Pt(12)
    doc.save(OUTPUT_DOCX)


def export_pdf_with_word(docx_path: Path, pdf_path: Path) -> bool:
    ps = f"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open('{str(docx_path).replace("'", "''")}')
$doc.SaveAs([ref] '{str(pdf_path).replace("'", "''")}', [ref] 17)
$doc.Close($false)
$word.Quit()
"""
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True, text=True)
        return True
    except Exception as exc:
        print(f"[WARN] Word PDF export failed: {exc}", file=sys.stderr)
        return False


def main() -> None:
    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = unique_pdfs(PDF_DIR)
    metadata = load_existing_article_metadata()
    all_rows: list[CandidateValue] = []
    texts: dict[str, str] = {}
    for pos, pdf in enumerate(pdfs, 1):
        print(f"[{pos:02d}/{len(pdfs):02d}] Extracting {pdf.name}")
        text = extract_pdf_text(pdf)
        texts[pdf.name] = text
        all_rows.extend(extract_candidates_from_text(pdf, text, metadata))
    candidate_df = candidates_to_frame(all_rows)
    article_summary = build_article_summary(candidate_df, pdfs, texts)
    tech_summary = build_technology_summary(candidate_df)
    enhanced_matrix = build_enhanced_mcdm_matrix(tech_summary)
    save_outputs(candidate_df, article_summary, tech_summary, enhanced_matrix, pdfs)
    update_manuscript(candidate_df, article_summary, tech_summary)
    export_pdf_with_word(OUTPUT_DOCX, OUTPUT_PDF)
    print("=" * 72)
    print("V9.12 automatic economic-value extraction complete")
    print(f"Unique PDFs processed: {len(pdfs)}")
    print(f"Candidate values: {len(candidate_df)}")
    print(f"High-confidence values: {int((candidate_df['confidence_score'] >= 0.80).sum()) if not candidate_df.empty else 0}")
    print(f"Outputs: {EXTRACTION_DIR}")
    print(f"Manuscript: {OUTPUT_DOCX}")
    print("=" * 72)


if __name__ == "__main__":
    main()
