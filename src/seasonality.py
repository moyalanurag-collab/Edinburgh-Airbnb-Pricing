"""Step 2 - seasonality.

The current Inside Airbnb calendar file no longer contains nightly prices, so seasonality is measured through DEMAND signals:

  1. Review volume by calendar month (reviews are written after a stay, so they trace when guests actually stayed,
     about 1-3 weeks earlier). Years 2022-2025 are used because they are complete post-pandemic years.
  2. Forward calendar availability: share of listing-nights marked unavailable (booked OR blocked by the host)
     by month and by weekday over the next 12 months. 'Unavailable' is therefore an upper bound on bookings.

Outputs: data/processed/seasonality_reviews_by_month.csv, seasonality_calendar_by_month.csv,
         seasonality_calendar_by_weekday.csv, outputs/fig_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "outputs"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
BLUE, ORANGE, GREY = "#2F5D8A", "#E07B39", "#7A7A7A"

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6",
                     "axes.axisbelow": True, "figure.dpi": 130, "font.size": 9})


def reviews_by_month(listing_ids) -> pd.DataFrame:
    rv = pd.read_csv(RAW / "reviews.csv", parse_dates=["date"])
    rv = rv[rv["listing_id"].isin(listing_ids)]
    rv["year"], rv["month"] = rv["date"].dt.year, rv["date"].dt.month
    full = rv[rv["year"].between(2022, 2025)]
    by = full.groupby(["year", "month"]).size().unstack("year")
    # normalise each year by its own monthly mean first, so year-on-year growth in review volume does not distort the shape
    normalised = by / by.mean(axis=0)
    by["avg_per_year"] = by.mean(axis=1)
    by["seasonal_index"] = normalised.mean(axis=1)      # 1.00 = an average month
    by.index.name = "month"
    by["month_name"] = [MONTHS[m - 1] for m in by.index]
    return by.reset_index()


def calendar_profile(listing_ids):
    cal = pd.read_csv(RAW / "calendar.csv.gz", usecols=["listing_id", "date", "available"], parse_dates=["date"])
    cal = cal[cal["listing_id"].isin(listing_ids)]
    cal["unavailable"] = (cal["available"] == "f").astype(int)
    cal["year_month"] = cal["date"].dt.to_period("M").astype(str)
    cal["weekday"] = cal["date"].dt.day_name()
    by_month = cal.groupby("year_month").agg(unavailable_share=("unavailable", "mean"), nights=("unavailable", "size")).reset_index()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_wd = cal.groupby("weekday")["unavailable"].mean().reindex(order).rename("unavailable_share").reset_index()
    return by_month, by_wd


def main() -> None:
    FIG.mkdir(exist_ok=True)
    ids = set(pd.read_csv(OUT / "listings_clean.csv", usecols=["id"])["id"])

    rv = reviews_by_month(ids)
    rv.to_csv(OUT / "seasonality_reviews_by_month.csv", index=False)
    cm, wd = calendar_profile(ids)
    # the first / last calendar months are partial; flag them instead of silently dropping
    cm["partial_month"] = cm["nights"] < cm["nights"].max() * 0.9
    cm.to_csv(OUT / "seasonality_calendar_by_month.csv", index=False)
    wd.to_csv(OUT / "seasonality_calendar_by_weekday.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    colours = [ORANGE if m == "Aug" else BLUE for m in rv["month_name"]]
    ax[0].bar(rv["month_name"], rv["seasonal_index"], color=colours)
    ax[0].axhline(1, color=GREY, lw=1, ls="--")
    ax[0].set_title("Guest demand by month (review-volume index, 2022-25; 1.0 = avg month)", loc="left", fontsize=9)
    for i, v in enumerate(rv["seasonal_index"]):
        ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    full = cm[~cm["partial_month"]]
    ax[1].plot(full["year_month"], 100 * full["unavailable_share"], marker="o", color=BLUE)
    ax[1].set_title("Listing-nights unavailable, next 12 months (%)", loc="left", fontsize=9)
    ax[1].tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(FIG / "fig_seasonality.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(wd["weekday"].str[:3], 100 * wd["unavailable_share"], color=BLUE)
    lo = 100 * wd["unavailable_share"].min()
    ax.set_ylim(max(lo - 4, 0), 100 * wd["unavailable_share"].max() + 2)
    ax.set_title("Unavailable share by weekday (%) - axis starts above zero", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_weekday.png"); plt.close(fig)

    peak = rv.loc[rv["seasonal_index"].idxmax()]
    low = rv.loc[rv["seasonal_index"].idxmin()]
    print(f"demand peak {peak.month_name} (index {peak.seasonal_index:.2f}), trough {low.month_name} ({low.seasonal_index:.2f}); "
          f"weekday unavailability range {100*wd.unavailable_share.min():.1f}-{100*wd.unavailable_share.max():.1f}%")


if __name__ == "__main__":
    main()
