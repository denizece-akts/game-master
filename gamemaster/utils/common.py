import re
import ast
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from importlib import metadata

import numpy as np
import pandas as pd
import dateparser
from unidecode import unidecode


MOJIBAKE_FIXES = [
    ("Â", ""), ("â€¢", "•"), ("â€“", "–"), ("â€”", "—"),
    ("â€˜", "‘"), ("â€™", "’"), ("â€œ", "“"), ("â€\x9d", "”"),
    ("â€¦", "…"), ("â€", "'"), ("Ã©", "é"), ("Ã", "Å"),
]


def fix_mojibake(text: str):
    if not isinstance(text, str):
        return text
    out = text
    for src, tgt in MOJIBAKE_FIXES:
        out = out.replace(src, tgt)
    return unidecode(out)


def normalize_whitespace(s: str):
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", " ", s.strip())


def to_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    s = re.sub(r"[^\d\.\-eE]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan


DATE_FORMATS = [
    "%d %b, %Y", "%d %B, %Y", "%b %d, %Y", "%B %d, %Y",
    "%d %B", "%B %d", "%d %b", "%b %d",
]


def parse_date_any(x):
    if pd.isna(x):
        return pd.NaT
    s = normalize_whitespace(str(x))
    try:
        dt = dateparser.parse(s, dayfirst=False, yearfirst=False, fuzzy=True)
        return pd.to_datetime(dt)
    except Exception:
        pass
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if "%Y" not in fmt:
                from datetime import datetime as _dt
                dt = dt.replace(year=_dt.utcnow().year)
            return pd.to_datetime(dt)
        except Exception:
            continue
    return pd.NaT


def parse_listlike(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            out = ast.literal_eval(s)
            if isinstance(out, list):
                return [normalize_whitespace(fix_mojibake(str(x))) for x in out]
        except Exception:
            pass
    s2 = s.strip("[]")
    parts = re.split(r"'\s*,\s*'|,\s*", s2)
    parts = [normalize_whitespace(fix_mojibake(p.strip(" '\""))) for p in parts if p.strip(" '\"")]
    return parts


def make_game_key(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = s.replace("™", "").replace("®", "")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = re.sub(r"[^\w\s:\-\&]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def clamp_sentences(text: str, max_sent: int = 2) -> str:
    return " ".join(_split_sentences(text)[:max_sent])


def clamp_chars(text: str, max_chars: int | None):
    if not isinstance(text, str) or max_chars is None or len(text) <= max_chars:
        return text if isinstance(text, str) else ""
    cut = text[:max_chars]
    if " " in cut:
        return cut.rsplit(" ", 1)[0] + " …"
    return cut + " …"


def _json_safe(o):
    from pathlib import Path as _Path
    import numpy as _np
    if isinstance(o, _Path):
        return str(o)
    if isinstance(o, (_np.floating, _np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_json_safe(x) for x in o]
    return o


def sha256_file(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dist_checksum(dist_name: str, buf_size: int = 1024 * 1024) -> str:
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"
    files = dist.files or []
    h = hashlib.sha256()
    for f in sorted(files, key=str):
        full_path = Path(dist.locate_file(f))
        if not full_path.is_file():
            continue
        h.update(str(full_path).encode("utf-8"))
        with full_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(buf_size), b""):
                h.update(chunk)
    return h.hexdigest()


def print_versions_and_checksums():
    print("=== RUNTIME VERSIONS ===")
    print(f"python: {sys.version.split()[0]}")
    dists = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("sentence-transformers", "sentence-transformers"),
        ("faiss-cpu", "faiss-cpu"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("huggingface_hub", "huggingface-hub"),
        ("dateparser", "dateparser"),
        ("Unidecode", "Unidecode"),
    ]
    for display, dist_name in dists:
        try:
            ver = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        print(f"{display}: {ver}")
    print("========================")

    print("=== PACKAGE CHECKSUMS (SHA256) ===")
    python_checksum = sha256_file(Path(sys.executable))
    print(f"python: {python_checksum}")
    for display, dist_name in dists:
        cs = dist_checksum(dist_name)
        print(f"{display}: {cs}")
