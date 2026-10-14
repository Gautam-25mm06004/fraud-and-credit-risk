"""Download both datasets into data/ (about 70 MB).

* ULB / Worldline credit-card fraud data (creditcard.csv.zip), via the GitHub
  mirror stat432/credit-analysis. Original: Kaggle "mlg-ulb/creditcardfraud".
* South German Credit (corrected 2019 version, SouthGermanCredit.asc), via the
  GitHub mirror mayurborkar/Internship-South-German-Bank-Credit-Risk.
  Original: UCI Machine Learning Repository, dataset 573.

    python scripts/download_data.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "creditcard.csv.zip": "https://raw.githubusercontent.com/stat432/credit-analysis/main/data-raw/creditcard.csv.zip",
    "SouthGermanCredit.asc": "https://raw.githubusercontent.com/mayurborkar/Internship-South-German-Bank-Credit-Risk/main/SouthGermanCredit/SouthGermanCredit.asc",
}


def main():
    dest = ROOT / "data"; dest.mkdir(exist_ok=True)
    ok = True
    for name, url in FILES.items():
        out = dest / name
        if out.exists() and out.stat().st_size > 10_000:
            print(f"{name} present"); continue
        print(f"downloading {name} ...")
        try:
            tmp = out.with_suffix(out.suffix + ".part")
            urllib.request.urlretrieve(url, tmp); tmp.replace(out)
        except Exception as e:
            ok = False; print(f"FAILED {name}: {e}\n  get it manually from {url}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
