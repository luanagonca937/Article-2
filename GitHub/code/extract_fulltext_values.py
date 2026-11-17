"""Candidate economic values from full texts that the user obtained legally.

Usage: python extract_fulltext_values.py <fulltext_dir> <output.xlsx>
<fulltext_dir> holds one PDF (or PubMed Central XML) per article. Full texts are not redistributed.
Values with confidence >= 0.80 were checked manually against the source before use in the article.
"""
import importlib.util, re, sys
from pathlib import Path

import fitz
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v912", HERE / "economic_value_extraction_rules.py")
m = importlib.util.module_from_spec(spec)
sys.modules["v912"] = m
spec.loader.exec_module(m)

src, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for path in sorted(src.iterdir()):
    if path.suffix.lower() == ".pdf":
        text = m.clean_text(" ".join(page.get_text() for page in fitz.open(path)))
    elif path.suffix.lower() == ".xml":
        raw = re.split(r"<ref-list", path.read_text(encoding="utf8", errors="ignore"))[0]
        text = m.clean_text(re.sub(r"<[^>]+>", " ", raw))
    else:
        continue
    text = re.split(r"\bReferences\b(?!.*\bReferences\b)", text)[0]
    for c in m.extract_candidates_from_text(path, text, {}):
        row = dict(c.__dict__)
        row["source_file"] = path.name
        rows.append(row)
df = pd.DataFrame(rows)
df.to_excel(out, index=False)
print(f"{len(df)} candidate values from {df['source_file'].nunique() if len(df) else 0} files; "
      f"{int((df['confidence'] >= 0.80).sum()) if len(df) else 0} with confidence >= 0.80")
