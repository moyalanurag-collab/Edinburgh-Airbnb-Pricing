import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare  # noqa: E402


def test_parse_price_handles_symbols_commas_and_blanks():
    s = pd.Series(["$197.50", "£1,200.00", "", None, "abc"])
    out = prepare.parse_price(s)
    assert out.iloc[0] == 197.5 and out.iloc[1] == 1200.0
    assert out.iloc[2:].isna().all()


def test_haversine_zero_and_known_distance():
    assert prepare.haversine_km(55.95, -3.19, 55.95, -3.19) == 0
    # Edinburgh Castle to Arthur's Seat is roughly 1.5 km
    d = prepare.haversine_km(55.9486, -3.2008, 55.9441, -3.1618)
    assert 2.0 < d < 3.0


def test_bathroom_parsing_text_half_and_shared():
    df = pd.DataFrame({"bathrooms": [np.nan, np.nan, 2.0], "bathrooms_text": ["1.5 baths", "Half-bath", "Shared half-bath"]})
    out = prepare.parse_bathrooms(df)
    assert out["bathrooms_n"].tolist() == [1.5, 0.5, 2.0]
    assert out["shared_bathroom"].tolist() == [0, 0, 1]


def test_amenity_patterns_match_expected_text():
    import re
    am = '["wifi", "dishwasher", "free parking on premises", "ev charger - level 2", "self check-in"]'
    hits = {k for k, pat in prepare.AMENITY_FLAGS.items() if re.search(pat, am, flags=re.I)}
    assert {"has_wifi", "has_dishwasher", "has_parking", "has_ev_charger", "has_self_checkin"} <= hits
    assert "has_hot_tub" not in hits


def test_cleaned_file_invariants_if_pipeline_has_run():
    f = prepare.OUT_DIR / "listings_clean.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    assert d["price_gbp"].between(20, 1500).all()
    assert d["min_nights"].max() <= 28
    assert d["id"].is_unique
    assert set(d["room_type"]) <= {"Entire home/apt", "Private room"}
    assert not d.drop(columns=["name"]).isna().any().any()
    assert np.allclose(d["log_price"], np.log(d["price_gbp"]))
