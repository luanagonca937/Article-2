"""
LDA BIBLIOMETRIC ANALYSIS - PYTHON
Adapted for BibTeX files on Technology, Management, and Sustainability.

Corrected version with:
1) Reading and merging multiple .bib files without duplicates
2) Complete database export to CSV and XLSX
3) Detailed article counts by source and duplicate records
4) Word report with article and duplicate statistics
5) K selection avoiding K=1 and, whenever possible, K=2
6) Topic evaluation with Log Perplexity and Topic Coherence
7) Topic prioritization (HIGH, MEDIUM, LOW priority)
8) Topic labeling with a complete report
9) Round word cloud
10) Interactive pyLDAvis visualization

Main fixes:
- Keeps processed abstracts aligned with the original articles.
- Removes duplicate DOI from the field dictionary.
- Improves the BibTeX parser for fields with nested braces.
- Protects CountVectorizer against an empty vocabulary.
- Preserves terms such as Industry 4.0/5.0 as industry4/industry5.
- Avoids K=2 when there are enough documents to test K>=3.
"""

import os
import re
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    BIBTEXPARSER_AVAILABLE = True
except ImportError:
    print("Warning: bibtexparser is not installed. Install it with: pip install bibtexparser")
    BIBTEXPARSER_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn is not installed. Install it with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib is not installed. Install it with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Warning: seaborn is not installed. Install it with: pip install seaborn")
    SEABORN_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    print("Warning: wordcloud is not installed. Install it with: pip install wordcloud")
    WORDCLOUD_AVAILABLE = False

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    print("Warning: python-docx is not installed. Install it with: pip install python-docx")
    DOCX_AVAILABLE = False

try:
    import pyLDAvis
    PYLDAVIS_AVAILABLE = True
except ImportError:
    print("Warning: pyLDAvis is not installed. Install it with: pip install pyLDAvis")
    PYLDAVIS_AVAILABLE = False


warnings.filterwarnings("ignore")


# ==========================
# SETTINGS
# ==========================

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.seed = 1234
        self.min_abstract_chars = 100
        self.min_term_freq = 5
        self.max_term_prop = 0.8
        self.min_docs_per_topic = 10
        self.n_words_per_topic = 10
        self.lda_iterations = 2000
        self.max_topics_cap = 8


config = Config()


# ==========================
# STOP WORDS
# ==========================

ENGLISH_STOP_WORDS = [
    "a", "an", "the", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "will", "would", "shall", "should", "may", "might",
    "must", "can", "could", "of", "in", "to", "for", "with", "on", "at",
    "by", "from", "up", "down", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "but", "done",
    "and", "or", "yet", "if", "else", "when", "where", "which", "while",
    "whom", "who", "whose", "why", "how", "either", "neither", "about",
    "across", "after", "before", "behind", "below", "between", "beyond",
    "during", "except", "into", "like", "near", "onto", "outside", "past",
    "since", "through", "throughout", "toward", "until", "upon", "within",
    "without", "according", "because", "become", "becomes", "becoming",
    "began", "beginning", "begins", "ever", "every", "everything",
    "everywhere", "former", "formerly", "hence", "hereafter", "hereby",
    "herein", "hereupon", "herself", "himself", "however", "itself",
    "least", "less", "made", "many", "maybe", "meantime", "meanwhile",
    "mostly", "much", "myself", "namely", "never", "nevertheless", "next",
    "nobody", "none", "noone", "nothing", "nowhere", "often", "otherwise",
    "perhaps", "rather", "thereafter", "thereby", "therefore", "therein",
    "thereupon", "thru", "thus", "together", "unless", "upward", "used",
    "using", "various", "want", "way", "well", "went", "what", "whatever",
    "whence", "whenever", "whereafter", "whereas", "whereby", "wherein",
    "whereupon", "wherever", "whether", "whichever", "whilst", "whither",
    "whoever", "whole", "whomever", "yes", "zero", "also", "accordingly",
    "consequently", "furthermore", "moreover", "nonetheless",
    "notwithstanding", "regardless", "subsequently", "anyone", "anything",
    "anywhere", "everyone", "someone", "something", "somewhere", "no one",
    "another", "tow"
]

METHOD_STOP_WORDS = [
    "study", "research", "paper", "article", "aim", "objective", "purpose",
    "method", "result", "conclusion", "background", "findings",
    "literature", "review", "discussion", "introduction", "analysis",
    "analyses", "studies", "results", "conclusions", "methods",
    "methodology", "time", "year", "years", "day", "days", "week", "weeks",
    "month", "months", "country", "countries", "world", "global",
    "international", "national", "local", "region", "level", "levels",
    "aspect", "aspects", "type", "types", "number", "numbers", "amount",
    "total", "overall", "different", "several", "average", "size",
    "process", "effect", "effects", "impact", "impacts", "important",
    "major", "main", "key", "critical", "essential", "relationship",
    "show", "shows", "indicate", "indicates", "suggest", "suggests",
    "reveal", "reveals", "elsevier", "springer", "wiley", "taylor", "come",
    "put", "right", "due", "current", "use", "see", "fit", "people",
    "provide", "contribute", "control", "collected", "highlight", "factor",
    "option", "component", "activity", "determine", "effective", "evaluated",
    "functions", "poor", "status", "based", "compared", "regarding",
    "concerning", "considering", "including", "excluding", "following",
    "previous", "subsequent", "prior", "post", "multiple", "numerous",
    "significant", "significantly", "particularly", "especially",
    "specifically", "generally", "usually", "typically", "commonly",
    "frequently", "rarely", "seldom", "always", "sometimes", "normally"
]

STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS + METHOD_STOP_WORDS))
print(f"Total stop words loaded: {len(STOP_WORDS)}")

KEEP_WORDS = [
    "digital", "transformation", "industry", "industry4", "industry5",
    "sustainability", "sustainable", "innovation", "technology",
    "management", "artificial", "intelligence", "ai", "blockchain", "iot",
    "internet", "things", "big", "data", "analytics", "cloud",
    "manufacturing", "smart_technology", "factory", "farming", "agriculture",
    "agroforestry", "lean", "six", "sigma", "circular", "economy", "green",
    "renewable", "energy", "governance", "policy", "framework", "model",
    "strategy", "strategic", "risk", "resilience", "supply", "chain",
    "cooperative", "cooperation", "collaboration", "leadership", "hoshin",
    "kanri", "quality", "performance", "adoption", "implementation",
    "challenge", "barrier", "opportunity", "cybersecurity", "security",
    "privacy", "ethics", "ethical", "value", "ecosystem", "platform",
    "network", "infrastructure", "urban", "rural", "development", "climate",
    "change", "adaptation", "mitigation", "stakeholder", "community",
    "social", "economic", "environmental"
]


# ==========================
# BIBTEX READING
# ==========================

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    value = str(value)
    value = re.sub(r"\s+", " ", value.replace("\r", " ").replace("\n", " "))
    return value.strip()


def split_bibtex_entries(content: str) -> List[str]:
    starts = [m.start() for m in re.finditer(r"@\w+\s*{", content)]
    if not starts:
        return []
    entries = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(content)
        entry = content[start:end].strip()
        if entry:
            entries.append(entry)
    return entries


def parse_bibtex_fields(entry: str) -> Dict[str, str]:
    match = re.match(r"@\w+\s*{\s*([^,]+),", entry, flags=re.DOTALL)
    if not match:
        return {}

    fields = {"ID": normalize_text(match.group(1))}
    i = match.end()
    n = len(entry)

    while i < n:
        while i < n and entry[i] in " \t\r\n,":
            i += 1
        if i >= n or entry[i] == "}":
            break

        key_start = i
        while i < n and re.match(r"[A-Za-z0-9_\-]", entry[i]):
            i += 1
        key = entry[key_start:i].strip().upper()

        while i < n and entry[i].isspace():
            i += 1
        if i >= n or entry[i] != "=":
            i += 1
            continue
        i += 1

        while i < n and entry[i].isspace():
            i += 1
        if i >= n:
            break

        if entry[i] == "{":
            i += 1
            value_start = i
            depth = 1
            while i < n and depth > 0:
                if entry[i] == "{":
                    depth += 1
                elif entry[i] == "}":
                    depth -= 1
                i += 1
            value = entry[value_start:i - 1]
        elif entry[i] == '"':
            i += 1
            value_start = i
            escaped = False
            while i < n:
                char = entry[i]
                if char == '"' and not escaped:
                    break
                escaped = char == "\\" and not escaped
                if char != "\\":
                    escaped = False
                i += 1
            value = entry[value_start:i]
            i += 1
        else:
            value_start = i
            while i < n and entry[i] not in ",}":
                i += 1
            value = entry[value_start:i]

        if key:
            fields[key] = normalize_text(value)

    return fields


def record_from_bibtexparser_entry(entry: Dict[str, Any], wanted_fields: List[str]) -> Dict[str, str]:
    normalized = {str(key).upper().replace("_", "-"): normalize_text(value) for key, value in entry.items()}
    record = {field.replace("-", "_"): normalized.get(field, "") for field in wanted_fields}
    record["ID"] = normalized.get("ID", "")
    record["ENTRYTYPE"] = normalized.get("ENTRYTYPE", "")
    if not record.get("JOURNAL"):
        for alt in ["CONTAINER_TITLE", "BOOKTITLE", "JOURNALTITLE"]:
            if normalized.get(alt.replace("_", "-")):
                record["JOURNAL"] = normalized.get(alt.replace("_", "-"))
                break
    if not record.get("KEYWORDS") and normalized.get("KEYWORD"):
        record["KEYWORDS"] = normalized.get("KEYWORD")
    return record


def records_from_manual_bibtex_parser(content: str, wanted_fields: List[str]) -> List[Dict[str, str]]:
    records = []
    for entry in split_bibtex_entries(content):
        parsed = parse_bibtex_fields(entry)
        if not parsed.get("TITLE"):
            continue

        record = {field.replace("-", "_"): parsed.get(field, "") for field in wanted_fields}
        record["ID"] = parsed.get("ID", "")
        record["ENTRYTYPE"] = parsed.get("ENTRYTYPE", "")
        if not record.get("JOURNAL") and record.get("CONTAINER_TITLE"):
            record["JOURNAL"] = record["CONTAINER_TITLE"]
        record.pop("CONTAINER_TITLE", None)
        records.append(record)
    return records


def record_identity_key(record: Dict[str, str]) -> str:
    title = normalize_text(record.get("TITLE", "")).lower()
    title = re.sub(r"[^\w\s]", " ", title, flags=re.UNICODE)
    title = re.sub(r"\s+", " ", title).strip()
    if title:
        return f"title::{title}"
    doi = normalize_text(record.get("DOI", "")).lower()
    if doi:
        return f"doi::{doi}"
    return f"id::{normalize_text(record.get('ID', '')).lower()}"


def load_bib_file_complete(filepath: str) -> pd.DataFrame:
    wanted_fields = [
        "TITLE", "ABSTRACT", "AUTHOR", "YEAR", "JOURNAL", "CONTAINER-TITLE",
        "BOOKTITLE", "JOURNALTITLE", "DOI", "KEYWORDS", "KEYWORD", "PUBLISHER",
        "VOLUME", "NUMBER", "PAGES", "URL", "TYPE", "LANGUAGE"
    ]

    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            content = f.read()
    except Exception as e:
        print(f"   Error loading {filepath}: {e}")
        return pd.DataFrame()

    records = []
    parser_used = "manual fallback"
    manual_records = None
    if BIBTEXPARSER_AVAILABLE:
        try:
            parser = BibTexParser(common_strings=True)
            parser.ignore_nonstandard_types = False
            parser.homogenize_fields = False
            bib_database = bibtexparser.loads(content, parser=parser)
            records = [
                record_from_bibtexparser_entry(entry, wanted_fields)
                for entry in bib_database.entries
            ]
            records = [record for record in records if record.get("TITLE")]
            parser_used = "bibtexparser"
        except Exception as exc:
            print(f"   bibtexparser failed for {os.path.basename(filepath)}; using manual fallback. Reason: {str(exc)[:120]}")
            records = []

    if not records:
        manual_records = records_from_manual_bibtex_parser(content, wanted_fields)
        records = manual_records
        parser_used = "manual fallback"
    else:
        manual_records = records_from_manual_bibtex_parser(content, wanted_fields)
        if len(manual_records) > len(records):
            existing_keys = {record_identity_key(record) for record in records}
            supplemental = [
                record for record in manual_records
                if record_identity_key(record) not in existing_keys
            ]
            if supplemental:
                records.extend(supplemental)
                parser_used = f"bibtexparser + manual supplement ({len(supplemental)})"

    for record in records:
        record.pop("CONTAINER_TITLE", None)
        record.pop("BOOKTITLE", None)
        record.pop("JOURNALTITLE", None)
        record.pop("KEYWORD", None)

    print(f"   Loaded {len(records)} records from {os.path.basename(filepath)} ({parser_used})")
    return pd.DataFrame(records)


def normalized_title(title: str) -> str:
    title = normalize_text(title).lower()
    title = re.sub(r"[^\w\s]", " ", title, flags=re.UNICODE)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def analyze_duplicates(df_list: List[pd.DataFrame], file_names: List[str]):
    print("\n" + "=" * 70)
    print("ARTICLE ANALYSIS BY SOURCE AND DUPLICATES")
    print("=" * 70)

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

    M = pd.concat(df_list, ignore_index=True)
    if "TITLE" not in M.columns:
        return M, total_before, len(M), 0, 0, []

    M["_TITLE_NORM"] = M["TITLE"].map(normalized_title)
    title_counts = M["_TITLE_NORM"].value_counts()
    duplicated_titles = title_counts[title_counts > 1]
    total_duplicates = int(len(duplicated_titles))
    total_duplicate_records = int(duplicated_titles.sum() - total_duplicates)

    print("\nDUPLICATES FOUND:")
    print("-" * 60)
    print(f"   Duplicate titles: {total_duplicates}")
    print(f"   Duplicate records removed: {total_duplicate_records}")

    M_unique = M.drop_duplicates(subset=["_TITLE_NORM"], keep="first").copy()
    dup_stats_by_base = []

    print("\nDUPLICATE ANALYSIS BY SOURCE:")
    print("-" * 60)
    unique_titles_global = set(M_unique["_TITLE_NORM"].dropna())

    for i, (df, name) in enumerate(zip(df_list, file_names)):
        original_count = len(df)
        titles_in_base = set(df["TITLE"].dropna().map(normalized_title))
        remaining = len(titles_in_base.intersection(unique_titles_global))

        other_titles = set()
        for j, df2 in enumerate(df_list):
            if i != j:
                other_titles.update(set(df2["TITLE"].dropna().map(normalized_title)))

        shared = len(titles_in_base.intersection(other_titles))
        removed = max(original_count - remaining, 0)

        dup_stats_by_base.append({
            "Source": name,
            "Original_Articles": original_count,
            "Kept_Articles": remaining,
            "Removed_Duplicates": removed,
            "Shared_Titles": shared
        })
        print(f"   {name}:")
        print(f"      Original: {original_count} | Kept: {remaining} | Removed: {removed} | Shared: {shared}")

    print("-" * 60)
    print(f"   {'GRAND TOTAL':<30} {len(M_unique):>6} articles")
    print("=" * 70)

    if total_duplicates > 0:
        examples = (
            M[M["_TITLE_NORM"].isin(duplicated_titles.index)]
            .groupby("_TITLE_NORM")["TITLE"]
            .first()
            .reset_index(drop=True)
        )
        dup_report = pd.DataFrame({
            "Title": examples,
            "Occurrences": duplicated_titles.values[:len(examples)]
        })
        dup_report.to_excel(config.results_dir / "00_duplicates_report.xlsx", index=False)
        print("Duplicate report saved as: 00_duplicates_report.xlsx")

    M_unique = M_unique.drop(columns=["_TITLE_NORM"], errors="ignore")
    return M_unique.reset_index(drop=True), total_before, len(M_unique), total_duplicates, total_duplicate_records, dup_stats_by_base


def create_word_report(df_summary, total_before, total_after, duplicates, removed_records, dup_stats_by_base, file_names):
    if not DOCX_AVAILABLE:
        print("Warning: python-docx is not installed. The Word report will not be generated.")
        return

    doc = Document()
    title = doc.add_heading("Bibliometric Analysis Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para = doc.add_paragraph(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    doc.add_heading("1. Loaded Articles by Source", level=1)
    table1 = doc.add_table(rows=len(df_summary) + 2, cols=2)
    table1.style = "Light Grid Accent 1"
    table1.cell(0, 0).text = "Source"
    table1.cell(0, 1).text = "Number of Articles"
    for col in range(2):
        table1.cell(0, col).paragraphs[0].runs[0].font.bold = True

    for i, row in df_summary.iterrows():
        table1.cell(i + 1, 0).text = str(row["Source"])
        table1.cell(i + 1, 1).text = str(row["Loaded Articles"])

    total_row = len(df_summary) + 1
    table1.cell(total_row, 0).text = "GRAND TOTAL"
    table1.cell(total_row, 1).text = str(total_before)
    for col in range(2):
        table1.cell(total_row, col).paragraphs[0].runs[0].font.bold = True

    doc.add_heading("2. Duplicate Analysis", level=1)
    if dup_stats_by_base:
        table2 = doc.add_table(rows=len(dup_stats_by_base) + 2, cols=5)
        table2.style = "Light Grid Accent 1"
        headers = ["Source", "Original Articles", "Kept Articles", "Removed Duplicates", "Shared Titles"]
        for col, header in enumerate(headers):
            table2.cell(0, col).text = header
            table2.cell(0, col).paragraphs[0].runs[0].font.bold = True

        for i, stats in enumerate(dup_stats_by_base):
            table2.cell(i + 1, 0).text = str(stats["Source"])
            table2.cell(i + 1, 1).text = str(stats["Original_Articles"])
            table2.cell(i + 1, 2).text = str(stats["Kept_Articles"])
            table2.cell(i + 1, 3).text = str(stats["Removed_Duplicates"])
            table2.cell(i + 1, 4).text = str(stats["Shared_Titles"])

        last = len(dup_stats_by_base) + 1
        table2.cell(last, 0).text = "TOTAL"
        table2.cell(last, 1).text = str(sum(s["Original_Articles"] for s in dup_stats_by_base))
        table2.cell(last, 2).text = str(total_after)
        table2.cell(last, 3).text = str(removed_records)
        table2.cell(last, 4).text = str(sum(s["Shared_Titles"] for s in dup_stats_by_base))
        for col in range(5):
            table2.cell(last, col).paragraphs[0].runs[0].font.bold = True

    doc.add_heading("3. General Summary", level=1)
    table3 = doc.add_table(rows=5, cols=2)
    table3.style = "Light Grid Accent 1"
    rows = [
        ("Metric", "Value"),
        ("Total before deduplication", total_before),
        ("Duplicate titles found", duplicates),
        ("Duplicate records removed", removed_records),
        ("Total unique articles after deduplication", total_after),
    ]
    for i, (metric, value) in enumerate(rows):
        table3.cell(i, 0).text = str(metric)
        table3.cell(i, 1).text = str(value)
    for col in range(2):
        table3.cell(0, col).paragraphs[0].runs[0].font.bold = True

    doc.add_heading("4. Analyzed Files", level=1)
    for fname in file_names:
        doc.add_paragraph(str(fname), style="List Bullet")

    output_path = config.results_dir / "00_duplicate_articles_report.docx"
    doc.save(output_path)
    print(f"Word report saved as: {output_path}")


# ==========================
# PREPROCESSING
# ==========================

def simple_tokenize(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"\bindustry\s*4\.0\b", " industry4 ", text)
    text = re.sub(r"\bindustry\s*5\.0\b", " industry5 ", text)
    text = re.sub(r"\b4\.0\b", " industry4 ", text)
    text = re.sub(r"\b5\.0\b", " industry5 ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stop_set = set(STOP_WORDS)
    keep_set = set(KEEP_WORDS)
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


def preprocess_abstracts(abstracts: List[str]) -> Tuple[List[List[str]], List[int]]:
    processed = []
    valid_indices = []
    rejected_count = 0

    for idx, abstract in enumerate(abstracts):
        if not abstract or len(str(abstract)) < config.min_abstract_chars:
            rejected_count += 1
            continue

        tokens = simple_tokenize(abstract)
        if tokens:
            processed.append(tokens)
            valid_indices.append(idx)
        else:
            rejected_count += 1

    print(f"Documents after processing: {len(processed)}")
    print(f"Documents rejected: {rejected_count}")
    return processed, valid_indices


# ==========================
# TOPIC EVALUATION
# ==========================

def calculate_log_perplexity(lda_model, dtm) -> float:
    try:
        return float(np.log(lda_model.perplexity(dtm)))
    except Exception as e:
        print(f"Error calculating log perplexity: {e}")
        return np.nan


def calculate_topic_coherence(lda_model, feature_names: List[str], dtm, n_words: int = 10) -> float:
    try:
        dtm_bool = (dtm > 0).astype(int)
        feature_index = {word: i for i, word in enumerate(feature_names)}
        topic_coherences = []

        for topic in lda_model.components_:
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            scores = []

            for i in range(len(top_words)):
                for j in range(i + 1, len(top_words)):
                    wi = feature_index[top_words[i]]
                    wj = feature_index[top_words[j]]
                    col_i = dtm_bool[:, wi]
                    col_j = dtm_bool[:, wj]
                    co_occurrence = col_i.multiply(col_j).sum()
                    freq_i = col_i.sum()
                    if freq_i > 0:
                        scores.append(float(co_occurrence / freq_i))

            topic_coherences.append(float(np.mean(scores)) if scores else 0.0)

        return float(np.mean(topic_coherences))
    except Exception as e:
        print(f"Error calculating topic coherence: {e}")
        return np.nan


def calculate_topic_coherence_for_topic(feature_names, dtm, top_words) -> float:
    try:
        dtm_bool = (dtm > 0).astype(int)
        feature_index = {word: i for i, word in enumerate(feature_names)}
        scores = []

        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                if top_words[i] not in feature_index or top_words[j] not in feature_index:
                    continue
                wi = feature_index[top_words[i]]
                wj = feature_index[top_words[j]]
                col_i = dtm_bool[:, wi]
                col_j = dtm_bool[:, wj]
                co_occurrence = col_i.multiply(col_j).sum()
                freq_i = col_i.sum()
                if freq_i > 0:
                    scores.append(float(co_occurrence / freq_i))

        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def select_optimal_k_with_metrics(dtm, feature_names, max_topics: int) -> Tuple[int, pd.DataFrame, str]:
    results = []
    start_k = 3 if max_topics >= 3 else 2

    print("\n=== TOPIC EVALUATION ===")
    print(f"Testing K from {start_k} to {max_topics}...")
    print("-" * 70)
    print(f"{'K':<6} {'Log Perplexity':<18} {'Topic Coherence':<18} {'Composite':<12}")
    print("-" * 70)

    for k in range(start_k, max_topics + 1):
        print(f"  Training K={k}...", end=" ", flush=True)
        try:
            lda_test = LatentDirichletAllocation(
                n_components=k,
                random_state=config.seed,
                max_iter=500,
                n_jobs=-1,
                learning_method="batch"
            )
            lda_test.fit(dtm)
            log_perp = calculate_log_perplexity(lda_test, dtm)
            coherence = calculate_topic_coherence(lda_test, feature_names, dtm, config.n_words_per_topic)
            composite = -log_perp + (coherence * 10) if not np.isnan(log_perp) and not np.isnan(coherence) else 0

            results.append({
                "K": k,
                "Log_Perplexity": log_perp,
                "Topic_Coherence": coherence,
                "Composite_Score": composite
            })
            print(f"\r  K={k}: LogPerp={log_perp:.4f}, Coherence={coherence:.4f}, Score={composite:.4f}")
        except Exception as e:
            print(f"\r  K={k}: Error - {str(e)[:80]}")

    print("-" * 70)

    if not results:
        fallback_k = 3 if max_topics >= 3 else 2
        return fallback_k, pd.DataFrame(), f"No metrics available, using K={fallback_k} as fallback"

    df_results = pd.DataFrame(results)
    best_k = int(df_results.sort_values("Composite_Score", ascending=False).iloc[0]["K"])
    premise_applied = "Yes - K=1 avoided; K=2 avoided when possible" if start_k == 3 else "Partial - insufficient documents to avoid K=2"
    print(f"\nBest K selected: {best_k}")
    return best_k, df_results, premise_applied


# ==========================
# DTM E LDA
# ==========================

def create_dtm_simple(tokens_list: List[List[str]]):
    documents = [" ".join(tokens) for tokens in tokens_list]
    min_df = min(config.min_term_freq, max(1, len(documents)))

    vectorizer = CountVectorizer(min_df=min_df, max_df=config.max_term_prop)
    try:
        dtm = vectorizer.fit_transform(documents)
    except ValueError:
        print("Warning: empty vocabulary with the current filters. Reducing min_df to 1.")
        vectorizer = CountVectorizer(min_df=1, max_df=1.0)
        dtm = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) == 0:
        raise ValueError("Could not create a vocabulary for the LDA model.")

    print(f"DTM dimensions: {dtm.shape}")
    return dtm, vectorizer, feature_names


def run_lda_simple(dtm, n_topics: int):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=config.seed,
        max_iter=config.lda_iterations,
        verbose=1,
        n_jobs=-1,
        learning_method="batch"
    )
    doc_topic = lda.fit_transform(dtm)
    return lda, doc_topic


def get_top_words(lda_model, feature_names: List[str], n_words: int = 10) -> Dict[str, List[str]]:
    topic_words = {}
    for idx, topic in enumerate(lda_model.components_):
        top_idx = topic.argsort()[-n_words:][::-1]
        topic_words[f"Topic_{idx + 1}"] = [feature_names[i] for i in top_idx]
    return topic_words


def calculate_topic_priority(doc_topic: np.ndarray, topic_words: dict, dtm=None, feature_names=None) -> pd.DataFrame:
    dominant_topic = np.argmax(doc_topic, axis=1) + 1
    topic_counts = pd.Series(dominant_topic).value_counts().sort_index()
    topic_percentages = (topic_counts / len(dominant_topic) * 100).round(2)
    avg_prob = doc_topic.mean(axis=0)

    topic_coherence_scores = {}
    if dtm is not None and feature_names is not None:
        for i in range(doc_topic.shape[1]):
            topic_num = i + 1
            words = topic_words.get(f"Topic_{topic_num}", [])
            topic_coherence_scores[topic_num] = calculate_topic_coherence_for_topic(feature_names, dtm, words[:5]) if words else 0.0

    max_avg_prob = max(avg_prob) if max(avg_prob) > 0 else 1
    max_freq = max(topic_counts.values) if len(topic_counts) > 0 else 1
    results = []

    for i in range(doc_topic.shape[1]):
        topic_num = i + 1
        freq_score = (topic_counts.get(topic_num, 0) / max_freq) * 40
        prob_score = (avg_prob[i] / max_avg_prob) * 35
        coherence_val = topic_coherence_scores.get(topic_num, 0.0)
        coherence_score = min(coherence_val * 25, 25)
        composite_score = freq_score + prob_score + coherence_score

        if composite_score >= 66:
            priority = "HIGH PRIORITY"
            priority_icon = "[HIGH]"
            priority_color = "#D73027"
        elif composite_score >= 33:
            priority = "MEDIUM PRIORITY"
            priority_icon = "[MEDIUM]"
            priority_color = "#FDAE61"
        else:
            priority = "LOW PRIORITY"
            priority_icon = "[LOW]"
            priority_color = "#4575B4"

        results.append({
            "Topic": topic_num,
            "Documents": int(topic_counts.get(topic_num, 0)),
            "Percentage": float(topic_percentages.get(topic_num, 0)),
            "Avg_Probability": round(float(avg_prob[i]), 4),
            "Topic_Coherence": round(float(coherence_val), 4),
            "Frequency_Score": round(float(freq_score), 2),
            "Probability_Score": round(float(prob_score), 2),
            "Coherence_Score": round(float(coherence_score), 2),
            "Composite_Score": round(float(composite_score), 2),
            "Priority_Level": priority,
            "Priority_Icon": priority_icon,
            "Priority_Color": priority_color,
            "Top_Words": ", ".join(topic_words.get(f"Topic_{topic_num}", [])[:5])
        })

    return pd.DataFrame(results).sort_values("Composite_Score", ascending=False)


def generate_topic_labels(topic_words: dict, df_priority: pd.DataFrame) -> dict:
    label_keywords = {
        "digital": "Digital Transformation",
        "transformation": "Digital Transformation",
        "industry": "Industry 4.0/5.0",
        "industry4": "Industry 4.0",
        "industry5": "Industry 5.0",
        "sustainability": "Sustainability",
        "sustainable": "Sustainability",
        "green": "Sustainability",
        "circular": "Circular Economy",
        "innovation": "Innovation",
        "technology": "Technological Innovation",
        "artificial": "AI and Analytics",
        "intelligence": "AI and Analytics",
        "ai": "AI and Analytics",
        "data": "AI and Analytics",
        "analytics": "AI and Analytics",
        "blockchain": "Blockchain and IoT",
        "iot": "Blockchain and IoT",
        "smart_technology": "Smart Technology",
        "farming": "Smart Cities/Farming",
        "agriculture": "Smart Cities/Farming",
        "lean": "Lean and Management",
        "management": "Lean and Management",
        "strategy": "Lean and Management",
        "quality": "Lean and Management",
        "supply": "Supply Chain",
        "chain": "Supply Chain",
        "governance": "Governance and Policies",
        "policy": "Governance and Policies",
        "stakeholder": "Stakeholders and Collaboration",
        "collaboration": "Stakeholders and Collaboration",
        "cooperative": "Stakeholders and Collaboration",
        "risk": "Risk and Resilience",
        "resilience": "Risk and Resilience",
        "security": "Security and Ethics",
        "cybersecurity": "Security and Ethics",
        "privacy": "Security and Ethics"
    }

    topic_labels = {}
    for topic_name, words in topic_words.items():
        topic_num = int(topic_name.split("_")[1])
        label_found = None
        for word in words[:5]:
            for key, label in label_keywords.items():
                if key in word.lower():
                    label_found = label
                    break
            if label_found:
                break

        if not label_found:
            label_found = f"Topic {topic_num}"

        priority_row = df_priority[df_priority["Topic"] == topic_num]
        priority_icon = priority_row["Priority_Icon"].values[0] if not priority_row.empty else "[LOW]"
        topic_labels[topic_num] = f"{priority_icon} {label_found}"

    return topic_labels


# ==========================
# VISUALIZACOES
# ==========================

def create_pyldavis(lda_model, dtm, vectorizer, output_path: str):
    if not PYLDAVIS_AVAILABLE:
        print("Warning: pyLDAvis is not available.")
        return False

    try:
        topic_term_dists = lda_model.components_
        topic_term_dists = topic_term_dists / topic_term_dists.sum(axis=1)[:, np.newaxis]
        doc_topic_dists = lda_model.transform(dtm)
        doc_lengths = np.array(dtm.sum(axis=1)).flatten()
        vocab = vectorizer.get_feature_names_out().tolist()
        term_freq = np.array(dtm.sum(axis=0)).flatten()

        vis = pyLDAvis.prepare(
            topic_term_dists=topic_term_dists,
            doc_topic_dists=doc_topic_dists,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_freq
        )
        pyLDAvis.save_html(vis, str(output_path))
        print(f"pyLDAvis saved: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating pyLDAvis: {e}")
        return False


def create_round_wordcloud(word_freq: pd.DataFrame, output_path: str):
    if not WORDCLOUD_AVAILABLE or word_freq.empty or not MATPLOTLIB_AVAILABLE:
        print("Wordcloud or matplotlib not available, or no data.")
        return

    stop_set = set(STOP_WORDS)
    filtered_words = {
        word: int(freq)
        for word, freq in zip(word_freq["word"].head(300), word_freq["frequency"].head(300))
        if word not in stop_set and len(str(word)) > 2 and not str(word).isdigit()
    }
    if not filtered_words:
        print("No valid words available for the word cloud.")
        return

    try:
        x, y = np.ogrid[:400, :400]
        center = (200, 200)
        radius = 180
        mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2).astype(np.uint8) * 255

        wc = WordCloud(
            mask=mask,
            background_color="white",
            max_words=200,
            repeat=False,
            min_font_size=5,
            colormap="Set1",
            random_state=config.seed,
            width=800,
            height=800
        ).generate_from_frequencies(filtered_words)

        plt.figure(figsize=(10, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud - Technology, Management, and Sustainability", fontsize=14, fontweight="bold")
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Round word cloud saved: {output_path}")
    except Exception as e:
        print(f"Error creating round word cloud: {e}")


def create_topic_evaluation_plot(k_metrics: pd.DataFrame, best_k: int, output_path: str):
    if not MATPLOTLIB_AVAILABLE or k_metrics.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(k_metrics["K"], k_metrics["Log_Perplexity"], "o-", color="blue", linewidth=2, markersize=8)
    axes[0].axvline(x=best_k, color="red", linestyle="--", linewidth=2, label=f"Best K = {best_k}")
    axes[0].set_xlabel("Number of Topics (K)")
    axes[0].set_ylabel("Log Perplexity (lower is better)")
    axes[0].set_title("Log Perplexity by Number of Topics", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_metrics["K"], k_metrics["Topic_Coherence"], "o-", color="green", linewidth=2, markersize=8)
    axes[1].axvline(x=best_k, color="red", linestyle="--", linewidth=2, label=f"Best K = {best_k}")
    axes[1].set_xlabel("Number of Topics (K)")
    axes[1].set_ylabel("Topic Coherence (higher is better)")
    axes[1].set_title("Topic Coherence by Number of Topics", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Topic Evaluation Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Topic evaluation plot saved: {output_path}")


def create_topic_priority_chart(df_priority: pd.DataFrame, output_path: str):
    if not MATPLOTLIB_AVAILABLE:
        return

    colors = {"HIGH PRIORITY": "#D73027", "MEDIUM PRIORITY": "#FDAE61", "LOW PRIORITY": "#4575B4"}
    df_sorted = df_priority.sort_values("Composite_Score", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = range(len(df_sorted))
    bar_colors = [colors[p] for p in df_sorted["Priority_Level"]]
    bars = ax.barh(y_pos, df_sorted["Composite_Score"], color=bar_colors, edgecolor="black", linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Topic {row['Topic']}\n{row['Priority_Icon']} {row['Priority_Level']}" for _, row in df_sorted.iterrows()], fontsize=10)
    ax.set_xlabel("Composite Score (0-100)", fontweight="bold")
    ax.set_xlim(0, 105)
    ax.axvline(x=66, color="#D73027", linestyle="--", linewidth=2, alpha=0.8, label="HIGH PRIORITY (>=66)")
    ax.axvline(x=33, color="#FDAE61", linestyle="--", linewidth=2, alpha=0.8, label="MEDIUM PRIORITY (33-66)")

    for bar, score in zip(bars, df_sorted["Composite_Score"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{score:.1f}", va="center", fontsize=9, fontweight="bold")

    ax.axvspan(66, 100, alpha=0.1, color="#D73027")
    ax.axvspan(33, 66, alpha=0.1, color="#FDAE61")
    ax.axvspan(0, 33, alpha=0.1, color="#4575B4")
    ax.set_title("Topic Prioritization by Composite Score", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Priority chart saved: {output_path}")


def create_topic_distribution_pie(df_priority: pd.DataFrame, output_path: str):
    if not MATPLOTLIB_AVAILABLE:
        return

    priority_groups = df_priority.groupby("Priority_Level").agg({"Percentage": "sum", "Topic": lambda x: list(x)}).reset_index()
    colors = {"HIGH PRIORITY": "#D73027", "MEDIUM PRIORITY": "#FDAE61", "LOW PRIORITY": "#4575B4"}
    pie_colors = [colors[p] for p in priority_groups["Priority_Level"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    labels = [f"{row['Priority_Level']}\n({row['Percentage']:.1f}%)" for _, row in priority_groups.iterrows()]
    sizes = priority_groups["Percentage"].values

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%", startangle=90)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax1.set_title("Distribution by Priority Level", fontweight="bold")

    df_sorted = df_priority.sort_values("Topic")
    bar_colors = [colors[p] for p in df_sorted["Priority_Level"]]
    bars = ax2.bar(df_sorted["Topic"], df_sorted["Percentage"], color=bar_colors, edgecolor="black")
    ax2.set_xlabel("Topic")
    ax2.set_ylabel("Document Percentage (%)")
    ax2.set_title("Document Distribution by Topic", fontweight="bold")
    for bar, pct in zip(bars, df_sorted["Percentage"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution pie chart saved: {output_path}")


def create_topic_correlation_heatmap(doc_topic: np.ndarray, output_path: str):
    if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
        return

    corr_matrix = np.corrcoef(doc_topic.T)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        xticklabels=[f"Topic {i + 1}" for i in range(corr_matrix.shape[0])],
        yticklabels=[f"Topic {i + 1}" for i in range(corr_matrix.shape[0])]
    )
    plt.title("Topic Correlation Matrix - LDA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Correlation heatmap saved: {output_path}")


def create_priority_summary_chart(df_priority: pd.DataFrame, output_path: str):
    if not MATPLOTLIB_AVAILABLE:
        return

    priority_counts = df_priority["Priority_Level"].value_counts()
    colors = {"HIGH PRIORITY": "#D73027", "MEDIUM PRIORITY": "#FDAE61", "LOW PRIORITY": "#4575B4"}
    order = ["HIGH PRIORITY", "MEDIUM PRIORITY", "LOW PRIORITY"]
    counts = [priority_counts.get(p, 0) for p in order]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(order, counts, color=[colors[p] for p in order], edgecolor="black", linewidth=2)
    plt.xlabel("Priority Level", fontweight="bold")
    plt.ylabel("Number of Topics", fontweight="bold")
    plt.title("Topic Prioritization Summary", fontsize=14, fontweight="bold")

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{count} topic(s)", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.ylim(0, max(counts) + 1 if counts else 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Priority summary chart saved: {output_path}")


# ==========================
# REPORT
# ==========================

def generate_report(df_priority: pd.DataFrame, topic_labels: dict, best_k: int, premise_applied: str, word_freq: pd.DataFrame, processed_tokens: List, k_metrics: pd.DataFrame):
    print("\n=== GENERATING COMPLETE REPORT ===")

    report_topics = []
    for _, row in df_priority.iterrows():
        topic_num = row["Topic"]
        report_topics.append({
            "Topic": topic_num,
            "Topic_Label": topic_labels.get(topic_num, f"Topic {topic_num}"),
            "Documents": row["Documents"],
            "Percentage": row["Percentage"],
            "Avg_Probability": row["Avg_Probability"],
            "Topic_Coherence": row.get("Topic_Coherence", 0),
            "Priority_Level": row["Priority_Level"],
            "Composite_Score": row["Composite_Score"],
            "Top_Words": row["Top_Words"]
        })

    df_report = pd.DataFrame(report_topics)
    model_info = pd.DataFrame([
        {"Parameter": "Optimal K (Topics)", "Value": best_k},
        {"Parameter": "Premise Applied (K != 1, avoid K=2 when possible)", "Value": premise_applied},
        {"Parameter": "Documents Analyzed", "Value": len(processed_tokens)},
        {"Parameter": "Unique Words", "Value": len(word_freq)},
        {"Parameter": "Stop Words Used", "Value": len(STOP_WORDS)},
        {"Parameter": "Keep Words", "Value": len(KEEP_WORDS)},
        {"Parameter": "Analysis Date", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ])

    if not k_metrics.empty:
        best_k_row = k_metrics[k_metrics["K"] == best_k]
        if not best_k_row.empty:
            model_info = pd.concat([
                model_info,
                pd.DataFrame([
                    {"Parameter": "Log Perplexity (best K)", "Value": best_k_row.iloc[0]["Log_Perplexity"]},
                    {"Parameter": "Topic Coherence (best K)", "Value": best_k_row.iloc[0]["Topic_Coherence"]}
                ])
            ], ignore_index=True)

    high_priority = df_report[df_report["Priority_Level"] == "HIGH PRIORITY"]
    medium_priority = df_report[df_report["Priority_Level"] == "MEDIUM PRIORITY"]
    low_priority = df_report[df_report["Priority_Level"] == "LOW PRIORITY"]

    priority_summary = pd.DataFrame([
        {"Priority_Level": "HIGH PRIORITY (Score >= 66)", "Count": len(high_priority), "Topics": ", ".join([f"Topic {t}" for t in high_priority["Topic"]])},
        {"Priority_Level": "MEDIUM PRIORITY (Score 33-66)", "Count": len(medium_priority), "Topics": ", ".join([f"Topic {t}" for t in medium_priority["Topic"]])},
        {"Priority_Level": "LOW PRIORITY (Score < 33)", "Count": len(low_priority), "Topics": ", ".join([f"Topic {t}" for t in low_priority["Topic"]])}
    ])

    with pd.ExcelWriter(config.results_dir / "COMPLETE_LDA_REPORT.xlsx") as writer:
        df_report.to_excel(writer, sheet_name="Topics_Priority", index=False)
        model_info.to_excel(writer, sheet_name="Model_Information", index=False)
        priority_summary.to_excel(writer, sheet_name="Priority_Summary", index=False)
        if not k_metrics.empty:
            k_metrics.to_excel(writer, sheet_name="Topic_Evaluation", index=False)
        word_freq.head(500).to_excel(writer, sheet_name="Word_Frequency", index=False)

    print("Complete report saved: COMPLETE_LDA_REPORT.xlsx")
    print("\n" + "=" * 70)
    print("COMPLETE LDA ANALYSIS REPORT")
    print("=" * 70)
    print(f"Optimal K: {best_k}")
    print(f"Documents analyzed: {len(processed_tokens)}")
    print(f"Unique words: {len(word_freq)}")
    print(f"HIGH PRIORITY: {len(high_priority)} topic(s)")
    print(f"MEDIUM PRIORITY: {len(medium_priority)} topic(s)")
    print(f"LOW PRIORITY: {len(low_priority)} topic(s)")
    print("=" * 70)
    return df_report


# ==========================
# BIBLIOGRAPHIC ANALYSIS
# ==========================

def split_author_names(author_text: Any) -> List[str]:
    text = normalize_text(author_text)
    if not text:
        return []
    text = re.sub(r"\s+and\s+", ";", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\|\s*", ";", text)
    authors = [normalize_text(author) for author in text.split(";")]
    return [author for author in authors if author]


def split_keywords(keyword_text: Any) -> List[str]:
    text = normalize_text(keyword_text)
    if not text:
        return []
    parts = re.split(r";|\||,", text)
    keywords = [normalize_text(part).lower() for part in parts]
    return [keyword for keyword in keywords if len(keyword) > 2]


def run_bibliographic_analysis(df_lda: pd.DataFrame):
    """Generate descriptive bibliographic analyses after LDA and before Hoshin Kanri."""
    print("\n=== BIBLIOGRAPHIC ANALYSIS ===")

    if df_lda.empty:
        print("Warning: bibliographic analysis skipped because the LDA result table is empty.")
        return None

    biblio_dir = config.results_dir / "Bibliographic Analysis"
    img_dir = biblio_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    df = df_lda.copy()
    if "year" in df.columns:
        df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year_numeric"] = np.nan

    total_documents = len(df)
    year_min = int(df["year_numeric"].min()) if df["year_numeric"].notna().any() else ""
    year_max = int(df["year_numeric"].max()) if df["year_numeric"].notna().any() else ""
    source_count = int(df["journal"].replace("", np.nan).nunique()) if "journal" in df.columns else 0
    doi_count = int(df["doi"].replace("", np.nan).nunique()) if "doi" in df.columns else 0

    author_counter = Counter()
    if "author" in df.columns:
        for author_text in df["author"].fillna(""):
            author_counter.update(split_author_names(author_text))

    keyword_counter = Counter()
    if "keywords" in df.columns:
        for keyword_text in df["keywords"].fillna(""):
            keyword_counter.update(split_keywords(keyword_text))

    summary = pd.DataFrame([
        {"Metric": "Total documents analyzed", "Value": total_documents},
        {"Metric": "First publication year", "Value": year_min},
        {"Metric": "Last publication year", "Value": year_max},
        {"Metric": "Unique sources/journals", "Value": source_count},
        {"Metric": "Unique DOIs", "Value": doi_count},
        {"Metric": "Unique authors", "Value": len(author_counter)},
        {"Metric": "Unique keywords", "Value": len(keyword_counter)},
        {"Metric": "Analysis date", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    ])

    if df["year_numeric"].notna().any():
        publications_by_year = (
            df.dropna(subset=["year_numeric"])
            .assign(Year=lambda x: x["year_numeric"].astype(int))
            .groupby("Year")
            .size()
            .reset_index(name="Documents")
            .sort_values("Year")
        )
    else:
        publications_by_year = pd.DataFrame(columns=["Year", "Documents"])

    if "journal" in df.columns:
        top_journals = (
            df["journal"]
            .fillna("")
            .map(normalize_text)
            .replace("", np.nan)
            .dropna()
            .value_counts()
            .head(25)
            .rename_axis("Source")
            .reset_index(name="Documents")
        )
    else:
        top_journals = pd.DataFrame(columns=["Source", "Documents"])

    top_authors = pd.DataFrame(author_counter.most_common(50), columns=["Author", "Documents"])
    top_keywords = pd.DataFrame(keyword_counter.most_common(50), columns=["Keyword", "Occurrences"])

    if {"main_topic", "topic_label"}.issubset(df.columns):
        topic_distribution = (
            df.groupby(["main_topic", "topic_label"])
            .size()
            .reset_index(name="Documents")
            .sort_values("Documents", ascending=False)
        )
    else:
        topic_distribution = pd.DataFrame(columns=["main_topic", "topic_label", "Documents"])

    if df["year_numeric"].notna().any() and "main_topic" in df.columns:
        topic_by_year = (
            df.dropna(subset=["year_numeric"])
            .assign(Year=lambda x: x["year_numeric"].astype(int))
            .pivot_table(index="Year", columns="main_topic", values="title", aggfunc="count", fill_value=0)
            .reset_index()
        )
    else:
        topic_by_year = pd.DataFrame()

    if "topic_label" in df.columns and "journal" in df.columns:
        sources_by_topic = (
            df.assign(journal_clean=df["journal"].fillna("").map(normalize_text))
            .query("journal_clean != ''")
            .groupby(["main_topic", "topic_label", "journal_clean"])
            .size()
            .reset_index(name="Documents")
            .sort_values(["main_topic", "Documents"], ascending=[True, False])
        )
    else:
        sources_by_topic = pd.DataFrame(columns=["main_topic", "topic_label", "journal_clean", "Documents"])

    with pd.ExcelWriter(biblio_dir / "BIBLIOGRAPHIC_ANALYSIS.xlsx") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        publications_by_year.to_excel(writer, sheet_name="Publications_by_Year", index=False)
        top_journals.to_excel(writer, sheet_name="Top_Sources", index=False)
        top_authors.to_excel(writer, sheet_name="Top_Authors", index=False)
        top_keywords.to_excel(writer, sheet_name="Top_Keywords", index=False)
        topic_distribution.to_excel(writer, sheet_name="Topic_Distribution", index=False)
        if not topic_by_year.empty:
            topic_by_year.to_excel(writer, sheet_name="Topics_by_Year", index=False)
        sources_by_topic.to_excel(writer, sheet_name="Sources_by_Topic", index=False)

    if MATPLOTLIB_AVAILABLE:
        if not publications_by_year.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(publications_by_year["Year"], publications_by_year["Documents"], marker="o", linewidth=2, color="#2E86AB")
            plt.fill_between(publications_by_year["Year"], publications_by_year["Documents"], alpha=0.15, color="#2E86AB")
            plt.xlabel("Year")
            plt.ylabel("Documents")
            plt.title("Publication Trend by Year", fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(img_dir / "publication_trend_by_year.png", dpi=150, bbox_inches="tight")
            plt.close()

        if not top_journals.empty:
            top_plot = top_journals.head(15).sort_values("Documents", ascending=True)
            plt.figure(figsize=(12, 8))
            plt.barh(top_plot["Source"], top_plot["Documents"], color="#4C78A8", edgecolor="black")
            plt.xlabel("Documents")
            plt.title("Top Sources/Journals", fontweight="bold")
            plt.tight_layout()
            plt.savefig(img_dir / "top_sources.png", dpi=150, bbox_inches="tight")
            plt.close()

        if not top_authors.empty:
            top_plot = top_authors.head(15).sort_values("Documents", ascending=True)
            plt.figure(figsize=(12, 8))
            plt.barh(top_plot["Author"], top_plot["Documents"], color="#59A14F", edgecolor="black")
            plt.xlabel("Documents")
            plt.title("Top Authors", fontweight="bold")
            plt.tight_layout()
            plt.savefig(img_dir / "top_authors.png", dpi=150, bbox_inches="tight")
            plt.close()

        if not top_keywords.empty:
            top_plot = top_keywords.head(20).sort_values("Occurrences", ascending=True)
            plt.figure(figsize=(12, 9))
            plt.barh(top_plot["Keyword"], top_plot["Occurrences"], color="#F28E2B", edgecolor="black")
            plt.xlabel("Occurrences")
            plt.title("Top Keywords", fontweight="bold")
            plt.tight_layout()
            plt.savefig(img_dir / "top_keywords.png", dpi=150, bbox_inches="tight")
            plt.close()

        if not topic_distribution.empty:
            topic_plot = topic_distribution.sort_values("Documents", ascending=True)
            labels = [f"Topic {row['main_topic']}: {row['topic_label']}" for _, row in topic_plot.iterrows()]
            plt.figure(figsize=(12, max(6, len(topic_plot) * 0.55)))
            plt.barh(labels, topic_plot["Documents"], color="#B07AA1", edgecolor="black")
            plt.xlabel("Documents")
            plt.title("Document Distribution by LDA Topic", fontweight="bold")
            plt.tight_layout()
            plt.savefig(img_dir / "topic_distribution.png", dpi=150, bbox_inches="tight")
            plt.close()

    if MATPLOTLIB_AVAILABLE and SEABORN_AVAILABLE and not topic_by_year.empty and topic_by_year.shape[1] > 1:
        heatmap_data = topic_by_year.set_index("Year")
        plt.figure(figsize=(12, max(6, len(heatmap_data) * 0.4)))
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.3, linecolor="white")
        plt.xlabel("LDA Topic")
        plt.ylabel("Year")
        plt.title("LDA Topics by Year", fontweight="bold")
        plt.tight_layout()
        plt.savefig(img_dir / "topics_by_year_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("Bibliographic analysis completed.")
    print(f"Files saved in: {biblio_dir}")
    return {
        "summary": summary,
        "publications_by_year": publications_by_year,
        "top_journals": top_journals,
        "top_authors": top_authors,
        "top_keywords": top_keywords,
        "topic_distribution": topic_distribution,
        "topics_by_year": topic_by_year,
        "directory": biblio_dir,
    }


# ==========================
# HOSHIN KANRI ANALYSIS
# ==========================

def run_hoshin_kanri_analysis(df_lda: pd.DataFrame, topic_number: int):
    """Analyze articles from an LDA topic against Hoshin Kanri guidelines."""
    required_cols = {"title", "year", "abstract", "main_topic"}
    missing = sorted(required_cols.difference(df_lda.columns))
    if missing:
        print(f"Warning: Hoshin Kanri analysis skipped. Missing columns: {', '.join(missing)}")
        return None

    if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
        print("Warning: matplotlib/seaborn is unavailable. Hoshin Kanri analysis will not be generated.")
        return None

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except ImportError:
        print("Warning: scikit-learn clustering modules are unavailable.")
        return None

    print("\n=== HOSHIN KANRI ANALYSIS ===")
    hoshin_dir = config.results_dir / "Hoshin Kanri"
    img_dir = hoshin_dir / "presentation_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    df_articles = df_lda[df_lda["main_topic"] == topic_number].copy()
    if df_articles.empty:
        print(f"Warning: no articles found in Topic {topic_number}. Hoshin Kanri analysis skipped.")
        return None

    df_articles = df_articles.reset_index(drop=True)
    df_articles.index = df_articles.index + 1
    print(f"Articles from Topic {topic_number} loaded: {len(df_articles)}")

    keywords_map = {
        "D01": ["continuous improvement", "organizational learning", "kaizen", "learning culture", "melhoria continua"],
        "D02": ["breakthrough", "challenging goals", "high impact", "performance orientation", "metas desafiadoras"],
        "D03": ["leadership", "top management", "commitment", "active involvement", "lideranca"],
        "D04": ["nemawashi", "consensus", "conflict resolution", "alignment", "conflito", "consenso"],
        "D05": ["systemic view", "holistic", "cross functional", "visao sistemica"],
        "D06": ["problem solving", "teamwork", "troubleshooting", "resolucao de problemas"],
        "D07": ["cross functional teams", "project management", "systemic issues"],
        "D08": ["long term vision", "strategic vision", "future oriented"],
        "D09": ["medium term goals", "measurable", "innovative"],
        "D10": ["vital few", "annual policies", "strategic priorities", "poucas politicas"],
        "D11": ["annual policies", "balanced", "sufficient"],
        "D12": ["annual planning", "participation", "bottom up"],
        "D13": ["catchball", "policy deployment", "mutual adjustment"],
        "D14": ["cause and effect", "root cause analysis", "causa e efeito"],
        "D15": ["management routines", "daily management", "rotinas de gestao"],
        "D16": ["responsibility", "ownership", "accountability", "responsabilidade"],
        "D17": ["self discipline", "daily routine", "autodisciplina"],
        "D18": ["periodic reviews", "performance meetings", "feedback"],
        "D19": ["visual management", "visual boards", "transparency", "gestao visual"],
        "D20": ["top management audit", "periodic diagnosis", "gemba walk", "diagnostico"],
    }

    guideline_descriptions = {
        "D01": "Establish a continuous improvement culture combined with a learning organization culture",
        "D02": "Develop a high-impact results orientation with challenging goals to drive performance improvement",
        "D03": "Ensure active leadership participation at all organizational levels to promote mobilization, alignment, and commitment",
        "D04": "Use the Nemawashi principle to deal with conflicts in the decision-making process",
        "D05": "Develop an orientation toward a systemic view",
        "D06": "Develop problem-solving and teamwork orientation among employees",
        "D07": "Develop the ability to address systemic issues through cross-functional teams",
        "D08": "Establish a long-term vision that is meaningful at all organizational levels",
        "D09": "Translate the long-term vision into measurable medium-term goals",
        "D10": "Translate medium-term goals into a few essential annual policies",
        "D11": "Deploy annual policies at each organizational level",
        "D12": "Involve everyone in the organization in annual planning",
        "D13": "Deploy each policy using the Catchball method",
        "D14": "Define goals and means based on cause-and-effect analysis",
        "D15": "Integrate goals into management routines",
        "D16": "Assign responsibilities through cascade deployment",
        "D17": "Develop self-discipline within teams",
        "D18": "Prepare periodic performance reports",
        "D19": "Keep information visible and updated",
        "D20": "Conduct a periodic diagnosis by top management",
    }

    categories_map = {
        **{f"D{i:02d}": "Organizational Culture" for i in range(1, 6)},
        **{f"D{i:02d}": "Capabilities" for i in range(6, 8)},
        **{f"D{i:02d}": "Focus" for i in range(8, 11)},
        **{f"D{i:02d}": "Alignment" for i in range(11, 15)},
        **{f"D{i:02d}": "Integration" for i in range(15, 20)},
        "D20": "Review",
    }

    guideline_cols = [f"D{i:02d}" for i in range(1, 21)]
    alignment_results = []
    for idx, row in df_articles.iterrows():
        title = normalize_text(row.get("title", ""))
        abstract = normalize_text(row.get("abstract", ""))
        text = re.sub(r"[-/]", " ", f"{title} {abstract}".lower())
        scores = {d: int(any(kw in text for kw in kws)) for d, kws in keywords_map.items()}
        row_data = {
            "Rank": idx,
            "Year": row.get("year", ""),
            "Title": title[:120],
            "Topic_Probability": row.get("topic_probability", ""),
        }
        row_data.update(scores)
        row_data["Total"] = sum(scores.values())
        alignment_results.append(row_data)

    df_matrix = pd.DataFrame(alignment_results)
    if df_matrix.empty:
        print("Warning: empty Hoshin matrix.")
        return None

    df_consolidation = pd.DataFrame([
        {
            "Guideline": d,
            "Category": categories_map[d],
            "Description": guideline_descriptions[d],
            "Articles": int(df_matrix[d].sum()),
            "Percentage": round(float(df_matrix[d].sum() / len(df_matrix) * 100), 2),
        }
        for d in guideline_cols
    ]).sort_values("Percentage", ascending=False)

    n_clusters = min(4, len(df_matrix))
    X = df_matrix[guideline_cols].values
    if n_clusters >= 2 and X.sum() > 0:
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.seed, n_init=10)
        df_matrix["Cluster"] = kmeans.fit_predict(X_scaled)
    else:
        X_scaled = X
        df_matrix["Cluster"] = 0

    cluster_nomes = {
        0: "Leadership and Organizational Culture",
        1: "Strategy and Alignment",
        2: "Integration and Operational Management",
        3: "General and Multidisciplinary Approach",
    }
    df_matrix["Cluster_Nome"] = df_matrix["Cluster"].map(cluster_nomes).fillna("Cluster Hoshin")

    with pd.ExcelWriter(hoshin_dir / "ALIGNMENT_MATRIX_FILLED.xlsx") as writer:
        df_matrix.to_excel(writer, sheet_name="Article_Alignment_Matrix", index=False)
        df_consolidation.to_excel(writer, sheet_name="Guideline_Consolidation", index=False)

    n_articles = min(40, len(df_matrix))
    top_articles = df_matrix.nlargest(n_articles, "Total").sort_values(["Cluster", "Total"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(22, max(8, n_articles * 0.35)))
    labels = [
        f"C{row['Cluster']} | R{int(row['Rank'])} | {row['Year']} | {str(row['Title'])[:35]}"
        for _, row in top_articles.iterrows()
    ]
    sns.heatmap(
        top_articles[guideline_cols].values,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        cbar_kws={"label": "Alignment (1=Yes, 0=No)"},
        xticklabels=guideline_cols,
        yticklabels=labels,
        linewidths=0.3,
        linecolor="gray",
        ax=ax,
    )
    for break_pos in [5, 7, 10, 14, 19]:
        ax.axvline(x=break_pos, color="black", linewidth=1.5)

    category_positions = {
        "Organizational Culture": (2.5, -0.8),
        "Capabilities": (6.0, -0.8),
        "Focus": (8.5, -0.8),
        "Alignment": (12.0, -0.8),
        "Integration": (16.5, -0.8),
        "Review": (19.5, -0.8),
    }
    for category, (pos_x, pos_y) in category_positions.items():
        ax.text(
            pos_x,
            pos_y,
            category,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            clip_on=False,
        )

    plt.xlabel("Hoshin Kanri Guidelines")
    plt.ylabel("Articles")
    plt.tight_layout()
    plt.savefig(img_dir / "heatmap_articles_with_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()

    if len(df_matrix) >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(12, 8))
        colors = ["#1B4332", "#2D6A4F", "#40916C", "#52B788"]
        for cluster_id in sorted(df_matrix["Cluster"].unique()):
            mask = df_matrix["Cluster"] == cluster_id
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=colors[int(cluster_id) % len(colors)],
                label=cluster_nomes.get(int(cluster_id), f"Cluster {cluster_id}"),
                s=90,
                alpha=0.75,
                edgecolors="black",
            )
        for i, (x, y) in enumerate(X_pca):
            plt.annotate(str(df_matrix.iloc[i]["Rank"]), (x, y), fontsize=8, ha="center", va="center")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        plt.legend(loc="best", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(img_dir / "pca_clusters.png", dpi=150, bbox_inches="tight")
        plt.close()

    top10 = df_consolidation.head(10)
    plt.figure(figsize=(14, 10))
    bars = plt.barh(range(len(top10)), top10["Percentage"], color=plt.cm.RdYlGn(top10["Percentage"] / 100), edgecolor="black")
    plt.yticks(range(len(top10)), [f"{row['Guideline']} - {row['Category']}" for _, row in top10.iterrows()])
    plt.xlabel("Articles mentioning the guideline (%)")
    plt.axvline(x=50, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    for bar, val in zip(bars, top10["Percentage"]):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center")
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(img_dir / "top10_guidelines.png", dpi=150, bbox_inches="tight")
    plt.close()

    article_table = df_matrix[["Rank", "Year", "Title", "Total", "Cluster", "Cluster_Nome"]].sort_values(["Cluster", "Total"], ascending=[True, False])
    article_table.to_excel(hoshin_dir / "table_articles_per_cluster.xlsx", index=False)

    cluster_table = []
    for cluster_id in sorted(df_matrix["Cluster"].unique()):
        cluster_data = df_matrix[df_matrix["Cluster"] == cluster_id]
        cluster_table.append({
            "Cluster": int(cluster_id),
            "Name": cluster_nomes.get(int(cluster_id), f"Cluster {cluster_id}"),
            "Article_Count": len(cluster_data),
            "Average_Guidelines": round(float(cluster_data["Total"].mean()), 1),
            "Min_Guidelines": int(cluster_data["Total"].min()),
            "Max_Guidelines": int(cluster_data["Total"].max()),
        })
    pd.DataFrame(cluster_table).to_excel(hoshin_dir / "table_cluster_summary.xlsx", index=False)
    df_consolidation.to_excel(hoshin_dir / "table_guidelines_complete.xlsx", index=False)

    print("Hoshin Kanri analysis completed.")
    print(f"Files saved in: {hoshin_dir}")
    return {
        "matrix": df_matrix,
        "consolidation": df_consolidation,
        "directory": hoshin_dir,
    }


# ==========================
# MAIN FUNCTION
# ==========================

def main():
    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn is required to run LDA.")
        return

    print("=" * 60)
    print("LDA ANALYSIS SCRIPT - CORRECTED PYTHON VERSION")
    print("WITH TOPIC PRIORITIZATION AND PYLDAVIS")
    print("=" * 60)
    print(f"\nWorking directory: {config.base_dir}")
    print(f"Results directory: {config.results_dir}")

    pd.DataFrame({"stop_word": sorted(STOP_WORDS)}).to_excel(config.results_dir / "00_stop_words_list.xlsx", index=False)
    pd.DataFrame({"keep_word": sorted(KEEP_WORDS)}).to_excel(config.results_dir / "00_keep_words_list.xlsx", index=False)

    print("\n=== LOADING BIBTEX FILES ===")
    bib_files = sorted(config.base_dir.glob("*.bib"))
    print(f"Found {len(bib_files)} .bib files:")
    for f in bib_files:
        print(f"  - {f.name}")

    if not bib_files:
        print("\nError: No .bib files found.")
        return

    databases = []
    file_names = []
    for bib_file in bib_files:
        print(f"\nLoading: {bib_file.name}")
        df = load_bib_file_complete(str(bib_file))
        if not df.empty:
            databases.append(df)
            file_names.append(bib_file.name)
            print(f"   Loaded {len(df)} records")

    if not databases:
        print("\nError: No data loaded.")
        return

    M, total_before, total_after, duplicates, removed_records, dup_stats_by_base = analyze_duplicates(databases, file_names)

    df_summary = pd.DataFrame([
        {"Source": name, "Loaded Articles": len(df)}
        for name, df in zip(file_names, databases)
    ])
    create_word_report(df_summary, total_before, total_after, duplicates, removed_records, dup_stats_by_base, file_names)

    print("\n=== SAVING COMPLETE DATABASE ===")
    output_excel = config.results_dir / "base_bibliometrix_COMPLETE.xlsx"
    output_csv = config.results_dir / "base_bibliometrix_COMPLETE.csv"
    M.to_excel(output_excel, index=False)
    M.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Complete database saved as:\n   - {output_excel}\n   - {output_csv}")

    summary_stats = pd.DataFrame({
        "Metric": ["Total before deduplication", "Total after deduplication", "Duplicate titles", "Removed records"],
        "Value": [total_before, total_after, duplicates, removed_records]
    })
    summary_stats.to_excel(config.results_dir / "00_deduplication_summary.xlsx", index=False)

    if "ABSTRACT" not in M.columns:
        print("Error: ABSTRACT column was not found in the BibTeX files.")
        return

    abstracts_df = M[["TITLE", "AUTHOR", "YEAR", "ABSTRACT"]].copy()
    abstracts_df.to_excel(config.results_dir / "complete_abstracts.xlsx", index=False)
    print("Abstracts saved")

    print("\n=== TEXT PRE-PROCESSING ===")
    abstracts = M["ABSTRACT"].fillna("").tolist()
    processed_tokens, valid_indices = preprocess_abstracts(abstracts)
    M_processed = M.iloc[valid_indices].reset_index(drop=True)

    if len(processed_tokens) < 2:
        print("Error: Not enough documents after preprocessing.")
        return

    print("\n=== WORD FREQUENCY ===")
    all_tokens = [token for doc in processed_tokens for token in doc]
    word_freq = pd.DataFrame(Counter(all_tokens).items(), columns=["word", "frequency"])
    word_freq = word_freq.sort_values("frequency", ascending=False)
    print(f"Total unique words: {len(word_freq)}")
    word_freq.to_excel(config.results_dir / "00_word_frequency_list.xlsx", index=False)

    create_round_wordcloud(word_freq, config.results_dir / "06_round_word_cloud.png")

    print("\n=== CREATING DTM ===")
    dtm, vectorizer, feature_names = create_dtm_simple(processed_tokens)

    print("\n=== SELECTING OPTIMAL K ===")
    max_topics = min(config.max_topics_cap, len(processed_tokens) // config.min_docs_per_topic)
    if max_topics < 3:
        if len(processed_tokens) >= config.min_docs_per_topic * 3:
            max_topics = 3
        else:
            max_topics = 2
            print("Warning: too few documents to safely avoid K=2. Using K=2 as fallback.")

    best_k, k_metrics, premise_applied = select_optimal_k_with_metrics(dtm, feature_names, max_topics)

    if not k_metrics.empty:
        k_metrics.to_excel(config.results_dir / "02_k_metrics.xlsx", index=False)
        create_topic_evaluation_plot(k_metrics, best_k, config.results_dir / "04_topic_evaluation.png")

    print(f"\n=== RUNNING LDA WITH K={best_k} ===")
    lda_model, doc_topic = run_lda_simple(dtm, best_k)

    topic_words = get_top_words(lda_model, feature_names, config.n_words_per_topic)
    print("\nTop words per topic:")
    for topic, words in topic_words.items():
        print(f"  {topic}: {', '.join(words)}")

    print("\n=== CALCULATING TOPIC PRIORITY ===")
    df_priority = calculate_topic_priority(doc_topic, topic_words, dtm, feature_names)
    topic_labels = generate_topic_labels(topic_words, df_priority)

    print("\nTopic Labels Generated:")
    for topic_num, label in topic_labels.items():
        print(f"   Topic {topic_num}: {label}")

    dominant_topic = np.argmax(doc_topic, axis=1) + 1
    topic_prob = np.max(doc_topic, axis=1)

    results_df = pd.DataFrame({
        "title": M_processed["TITLE"].values,
        "author": M_processed["AUTHOR"].values if "AUTHOR" in M_processed.columns else "",
        "year": M_processed["YEAR"].values,
        "journal": M_processed["JOURNAL"].values if "JOURNAL" in M_processed.columns else "",
        "doi": M_processed["DOI"].values if "DOI" in M_processed.columns else "",
        "keywords": M_processed["KEYWORDS"].values if "KEYWORDS" in M_processed.columns else "",
        "publisher": M_processed["PUBLISHER"].values if "PUBLISHER" in M_processed.columns else "",
        "abstract": M_processed["ABSTRACT"].values,
        "main_topic": dominant_topic,
        "topic_label": [topic_labels.get(t, f"Topic {t}") for t in dominant_topic],
        "topic_probability": topic_prob
    })

    print("\n=== SAVING RESULTS ===")
    with pd.ExcelWriter(config.results_dir / "03_lda_results.xlsx") as writer:
        results_df.to_excel(writer, sheet_name="Documents with Topics", index=False)
        pd.DataFrame([(k, ", ".join(v)) for k, v in topic_words.items()], columns=["Topic", "Top_Words"]).to_excel(writer, sheet_name="LDA Words", index=False)
        df_priority.to_excel(writer, sheet_name="Topic_Priority", index=False)
        topic_dist = pd.Series(dominant_topic).value_counts().sort_index()
        pd.DataFrame({"Topic": topic_dist.index, "Count": topic_dist.values}).to_excel(writer, sheet_name="LDA Distribution", index=False)
        word_freq.head(1000).to_excel(writer, sheet_name="Word Frequency", index=False)

    print("\n=== CREATING VISUALIZATIONS ===")
    create_topic_priority_chart(df_priority, config.results_dir / "07_topic_priority_chart.png")
    create_topic_distribution_pie(df_priority, config.results_dir / "08_topic_distribution_pie.png")
    create_priority_summary_chart(df_priority, config.results_dir / "10_priority_summary.png")
    create_topic_correlation_heatmap(doc_topic, config.results_dir / "09_topic_correlation.png")

    print("\n=== CREATING PYLDAVIS INTERACTIVE VISUALIZATION ===")
    create_pyldavis(lda_model, dtm, vectorizer, config.results_dir / "11_pyldavis_visualization.html")

    generate_report(df_priority, topic_labels, best_k, premise_applied, word_freq, processed_tokens, k_metrics)
    run_bibliographic_analysis(results_df)

    most_important_topic = int(df_priority.sort_values("Composite_Score", ascending=False).iloc[0]["Topic"])
    most_important_label = topic_labels.get(most_important_topic, f"Topic {most_important_topic}")
    print(f"\nMost important topic selected for Hoshin Kanri: Topic {most_important_topic} - {most_important_label}")
    run_hoshin_kanri_analysis(results_df, topic_number=most_important_topic)

    high_priority = df_priority[df_priority["Priority_Level"] == "HIGH PRIORITY"]
    medium_priority = df_priority[df_priority["Priority_Level"] == "MEDIUM PRIORITY"]
    low_priority = df_priority[df_priority["Priority_Level"] == "LOW PRIORITY"]

    print("\n" + "=" * 60)
    print("LDA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Optimal K: {best_k}")
    print(f"Premise: {premise_applied}")
    print(f"Documents: {len(processed_tokens)}")
    print(f"Unique words: {len(word_freq)}")
    print(f"Stop words: {len(STOP_WORDS)}")
    print(f"HIGH PRIORITY: {len(high_priority)} topic(s)")
    print(f"MEDIUM PRIORITY: {len(medium_priority)} topic(s)")
    print(f"LOW PRIORITY: {len(low_priority)} topic(s)")
    print(f"Results saved in: {config.results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
