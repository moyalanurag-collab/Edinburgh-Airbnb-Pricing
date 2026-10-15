"""Step 1 - clean the Inside Airbnb Edinburgh listings and engineer features.

Input : data/raw/listings.csv.gz   (Inside Airbnb, scraped 2026-09-19, CC BY 4.0)
Output: data/processed/listings_clean.csv       (one row per listing, modelling-ready)
        data/processed/cleaning_log.json        (how many rows each rule removed)

Cleaning rules (each one is logged):
  * price is missing / not a number              -> cannot be modelled
  * price outside GBP 20 - 1,500 per night       -> implausible or non-comparable (placeholder prices, data-entry errors)
  * minimum stay > 28 nights                     -> long lets are not nightly short-let pricing
  * room type 'Hotel room' / 'Shared room'       -> too few listings (~40) to model reliably
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "listings.csv.gz"
OUT_DIR = ROOT / "data" / "processed"

# Edinburgh Royal Mile / Old Town (approximate), used for distance-to-centre
CENTRE = (55.9500, -3.1900)

AMENITY_FLAGS = {  # flag name -> regex searched in the amenities list (case-insensitive)
    "has_wifi": r"\bwifi\b",
    "has_kitchen": r"\bkitchen\b",
    "has_washer": r"\bwasher\b",
    "has_dryer": r"\bdryer\b",
    "has_dishwasher": r"dishwasher",
    "has_parking": r"parking",
    "has_hot_tub": r"hot tub",
    "has_pool": r"\bpool\b",
    "has_air_con": r"air conditioning",
    "has_workspace": r"workspace",
    "has_ev_charger": r"ev charger",
    "has_gym": r"\bgym\b",
    "has_lift": r"\belevator\b|\blift\b",
    "has_fireplace": r"fireplace",
    "has_bbq": r"\bbbq\b|barbecue|grill",
    "has_crib": r"\bcrib\b",
    "has_self_checkin": r"self check-in|lockbox|smart lock|keypad",
}


def haversine_km(lat, lon, lat0, lon0):
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def parse_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def parse_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    txt = df["bathrooms_text"].fillna("").str.lower()
    num = pd.to_numeric(txt.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    num = num.where(~txt.str.contains("half"), 0.5)
    bath = df["bathrooms"].where(df["bathrooms"].notna(), num)
    df["bathrooms_n"] = bath
    df["shared_bathroom"] = txt.str.contains("shared").astype(int)
    return df


def build() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW, low_memory=False)
    log = {"rows_raw": len(df)}

    df["price_gbp"] = parse_price(df["price"])
    log["dropped_missing_price"] = int(df["price_gbp"].isna().sum())
    df = df[df["price_gbp"].notna()]
    log["dropped_price_outside_20_1500"] = int(((df["price_gbp"] < 20) | (df["price_gbp"] > 1500)).sum())
    df = df[df["price_gbp"].between(20, 1500)]
    log["dropped_long_stay_over_28_nights"] = int((df["minimum_nights"] > 28).sum())
    df = df[df["minimum_nights"] <= 28]
    log["dropped_hotel_or_shared_room"] = int(df["room_type"].isin(["Hotel room", "Shared room"]).sum())
    df = df[df["room_type"].isin(["Entire home/apt", "Private room"])].copy()
    log["rows_model_ready"] = len(df)

    # --- target
    df["log_price"] = np.log(df["price_gbp"])

    # --- size / capacity
    df = parse_bathrooms(df)
    df["bedrooms_n"] = df["bedrooms"].fillna(df.groupby("accommodates")["bedrooms"].transform("median")).fillna(1)
    df.loc[df["room_type"] == "Private room", "bedrooms_n"] = df.loc[df["room_type"] == "Private room", "bedrooms_n"].clip(upper=1)
    df["beds_n"] = df["beds"].fillna(df["accommodates"] / 2).clip(lower=1)
    df["bathrooms_n"] = df["bathrooms_n"].fillna(1)

    # --- building type, separate from room type (Airbnb's property_type mixes the two, e.g. 'Private room in condo')
    pt = df["property_type"].fillna("").str.lower()
    df["building_type"] = np.select(
        [pt.str.contains("condo"), pt.str.contains("rental unit|loft|serviced apartment"), pt.str.contains("townhouse"),
         pt.str.contains("guesthouse|guest suite|cottage|cabin|bungalow|tiny"), pt.str.contains("home|house|villa|castle|bed and breakfast")],
        ["Condo", "Flat / rental unit", "Townhouse", "Guesthouse / cottage", "House / other home"], default="Other")
    df["property_group"] = df["building_type"]

    # --- amenities
    am = df["amenities"].fillna("[]").str.lower()
    for flag, pat in AMENITY_FLAGS.items():
        df[flag] = am.str.contains(pat, flags=re.I, regex=True).astype(int)
    df["n_amenities"] = am.apply(lambda s: s.count('",') + 1 if s not in ("[]", "") else 0)

    # --- host
    # host_since is blank in this release; tenure is published as whole years + extra months instead
    df["host_years"] = df["hosts_time_as_host_years"] + df["hosts_time_as_host_months"] / 12
    df["host_years"] = df["host_years"].fillna(df["host_years"].median())
    df["is_superhost"] = (df["host_is_superhost"] == "t").astype(int)
    df["host_listings"] = df["calculated_host_listings_count"].fillna(1)
    df["multi_listing_host"] = (df["host_listings"] > 1).astype(int)
    df["has_licence_field"] = df["license"].notna().astype(int)

    # --- stay rules and demand-side signals (not derived from price)
    df["min_nights"] = df["minimum_nights"]
    df["availability_365_n"] = df["availability_365"].fillna(df["availability_365"].median())
    df["n_reviews"] = df["number_of_reviews"].fillna(0)
    df["reviews_per_month_n"] = df["reviews_per_month"].fillna(0)
    for c in ["review_scores_rating", "review_scores_location", "review_scores_value", "review_scores_cleanliness"]:
        df[c + "_n"] = df[c].fillna(df[c].median())
    df["has_reviews"] = (df["n_reviews"] > 0).astype(int)

    # --- geography
    df["dist_centre_km"] = haversine_km(df["latitude"], df["longitude"], *CENTRE)
    df["neighbourhood"] = df["neighbourhood_cleansed"]

    cols = [
        "id", "name", "neighbourhood", "latitude", "longitude", "room_type", "property_type", "property_group", "building_type",
        "price_gbp", "log_price", "accommodates", "bedrooms_n", "beds_n", "bathrooms_n", "shared_bathroom",
        *AMENITY_FLAGS, "n_amenities", "host_years", "is_superhost", "host_listings", "multi_listing_host",
        "has_licence_field", "min_nights", "availability_365_n", "n_reviews", "reviews_per_month_n",
        "review_scores_rating_n", "review_scores_location_n", "review_scores_value_n", "review_scores_cleanliness_n",
        "has_reviews", "dist_centre_km",
    ]
    return df[cols].reset_index(drop=True), log


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, log = build()
    df.to_csv(OUT_DIR / "listings_clean.csv", index=False)
    (OUT_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    print(f"median price GBP {df.price_gbp.median():.0f}; {df.neighbourhood.nunique()} neighbourhoods; saved {len(df):,} rows")


if __name__ == "__main__":
    main()
