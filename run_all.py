"""Run the whole analysis end to end:

    python run_all.py              # uses files already in data/raw/
    python run_all.py --download   # fetch the Edinburgh files from Inside Airbnb first
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "https://data.insideairbnb.com/united-kingdom/scotland/edinburgh/2026-09-19"
FILES = ["data/listings.csv.gz", "data/calendar.csv.gz", "visualisations/reviews.csv",
         "visualisations/neighbourhoods.geojson", "visualisations/neighbourhoods.csv"]


def download() -> None:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = raw / f.split("/")[-1]
        if dest.exists():
            print(f"  = {dest.name} (already there)")
            continue
        print(f"  + {dest.name}")
        urllib.request.urlretrieve(f"{BASE}/{f}", dest)


def step(title: str, script: str) -> None:
    print(f"\n== {title}")
    subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    if ap.parse_args().download:
        print("== 0. Download Inside Airbnb files")
        download()
    step("1. Clean + feature engineering", "prepare.py")
    step("2. Seasonality", "seasonality.py")
    step("3. Models (baselines, mixed model, LightGBM + SHAP)", "model.py")
    step("4. Maps + BI exports", "make_map.py")
    step("5. Findings report", "report.py")
    print("\nDone. Open outputs/edinburgh_price_map.html and docs/findings.md")


if __name__ == "__main__":
    main()
