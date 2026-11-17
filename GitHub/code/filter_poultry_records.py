"""Poultry-only subset of the new collection. Raw BibTeX entries are kept verbatim; only the filter decides inclusion."""
import csv, glob, os, re, sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf8")
INP, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
INCLUDE = re.compile(r"poultry|broiler|laying hen|layer hen|layer chicken|chicken|egg production|hatcher", re.I)
EXCLUDE = re.compile(r"chicken[- ]and[- ]egg|chicken[- ]or[- ]the[- ]egg", re.I)
FIELDS = ("title", "abstract", "keywords", "author_keywords", "keywords-plus")


def field(entry, name):
    m = re.search(rf"(?im)^\s*{name}\s*=\s*[{{\"](.*?)[}}\"]\s*,?\s*$", entry, re.S)
    return m.group(1) if m else ""


log, kept_total, seen_total = [], 0, 0
by_src = Counter()
for f in sorted(glob.glob(os.path.join(INP, "*.bib"))):
    raw = open(f, encoding="utf8", errors="ignore").read()
    starts = [m.start() for m in re.finditer(r"(?m)^\s*@\w+\s*\{", raw)]
    if len(starts) < 2:  # IEEE export is not line-broken
        starts = [m.start() for m in re.finditer(r"@\w+\s*\{", raw)]
    parts = [raw[a:b] for a, b in zip(starts, starts[1:] + [len(raw)])]
    keep = []
    for e in parts:
        text = " ".join(field(e, n) for n in FIELDS)
        if not text.strip():
            text = e
        cleaned = EXCLUDE.sub(" ", text)
        hit = INCLUDE.search(cleaned)
        seen_total += 1
        if hit:
            keep.append(e)
            by_src[re.sub(r"[_\d].*", "", os.path.basename(f))] += 1
            log.append({"file": os.path.basename(f), "match": hit.group(0).lower(), "title": re.sub(r"\s+", " ", field(e, "title"))[:160]})
    if keep:
        open(os.path.join(OUT, os.path.basename(f)), "w", encoding="utf8").write("\n".join(keep))
    kept_total += len(keep)
with open(os.path.join(os.path.dirname(OUT.rstrip("/\\")), "poultry_records_included.csv"), "w", newline="", encoding="utf-8-sig") as h:
    w = csv.DictWriter(h, fieldnames=["file", "match", "title"])
    w.writeheader()
    w.writerows(log)
print("seen", seen_total, "kept", kept_total, dict(by_src))
print("by term", Counter(x["match"] for x in log).most_common())
for x in log[:12]:
    print(" -", x["match"], "|", x["title"][:105])
print("files with entries", len(glob.glob(os.path.join(OUT, "*.bib"))))
