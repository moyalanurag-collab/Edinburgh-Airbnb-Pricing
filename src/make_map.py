"""Step 4 - geographic view.

Writes
  outputs/edinburgh_price_map.html        interactive map (open in any browser; works on GitHub Pages)
  outputs/fig_map_median_price.png        static choropleth for the README
  outputs/fig_map_premium.png             static choropleth of the model-based neighbourhood premium
  data/processed/neighbourhoods_enriched.geojson   polygons + metrics (drop straight into Power BI / Tableau / QGIS)
  data/processed/listings_for_bi.csv      one row per listing with model price and gap, for BI tools

Map layers: (1) median nightly price per neighbourhood, (2) model-based neighbourhood premium after controlling for size,
quality and distance, (3) every listing coloured by how far its price sits above/below the model's fair price.
Neighbourhoods with fewer than 10 listings are shown in grey: their numbers are too noisy to interpret.
"""
import json
from pathlib import Path

import branca.colormap as cm
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
MIN_LISTINGS = 10


def rings(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [poly[0] for poly in geom["coordinates"]]


def static_map(gj, values: dict, title: str, label: str, path: Path, cmap="YlOrRd", centre0=False) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7))
    vals = [v for v in values.values() if v is not None and not np.isnan(v)]
    vmin, vmax = (-max(abs(min(vals)), abs(max(vals))), max(abs(min(vals)), abs(max(vals)))) if centre0 else (min(vals), max(vals))
    cmap_ = plt.get_cmap("RdBu_r" if centre0 else cmap)
    patches, colours = [], []
    for f in gj["features"]:
        v = values.get(f["properties"]["neighbourhood"])
        for r in rings(f["geometry"]):
            patches.append(Polygon(np.array(r), closed=True))
            colours.append(cmap_((v - vmin) / (vmax - vmin)) if v is not None and not np.isnan(v) else (0.88, 0.88, 0.88, 1))
    ax.add_collection(PatchCollection(patches, facecolors=colours, edgecolors="white", linewidths=0.4))
    ax.autoscale_view(); ax.set_aspect(1 / np.cos(np.radians(55.95)))
    ax.set_axis_off(); ax.set_title(title, loc="left", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap_, norm=plt.Normalize(vmin, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01); cb.set_label(label)
    ax.text(0.01, 0.01, f"Grey = fewer than {MIN_LISTINGS} listings. Data: Inside Airbnb (CC BY 4.0)", transform=ax.transAxes, fontsize=7, color="#666")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    gj = json.loads((ROOT / "data" / "raw" / "neighbourhoods.geojson").read_text(encoding="utf-8"))
    nb = pd.read_csv(DATA / "neighbourhood_effects.csv")
    pred = pd.read_csv(DATA / "listing_predictions.csv")
    clean = pd.read_csv(DATA / "listings_clean.csv")

    ok = nb[nb["reliable"]].set_index("neighbourhood")
    med = {k: float(ok.loc[k, "median_price"]) if k in ok.index else None for k in (f["properties"]["neighbourhood"] for f in gj["features"])}
    prem = {k: float(ok.loc[k, "premium_pct_vs_city"]) if k in ok.index else None for k in med}

    # enriched polygons for BI tools
    info = nb.set_index("neighbourhood")
    for f in gj["features"]:
        k = f["properties"]["neighbourhood"]
        f["properties"].update({"listings": int(info.loc[k, "listings"]) if k in info.index else 0,
                                "median_price": med[k], "premium_pct_vs_city": prem[k], "reliable": int(k in ok.index)})
    (DATA / "neighbourhoods_enriched.geojson").write_text(json.dumps(gj), encoding="utf-8")

    bi = pred.merge(clean[["id", "property_type", "building_type", "bedrooms_n", "bathrooms_n", "dist_centre_km", "n_reviews",
                           "review_scores_rating_n", "is_superhost", "multi_listing_host", "min_nights"]], on="id")
    bi.to_csv(DATA / "listings_for_bi.csv", index=False)

    static_map(gj, med, "Median nightly price by neighbourhood (GBP)", "GBP per night", OUT / "fig_map_median_price.png")
    static_map(gj, prem, "Neighbourhood price premium vs city average, after controlling for size, quality and distance",
               "% premium (mixed model)", OUT / "fig_map_premium.png", centre0=True)

    # ---------------- interactive map ----------------------------------------------------
    m = folium.Map(location=[55.9533, -3.1883], zoom_start=12, tiles="OpenStreetMap", prefer_canvas=True)
    c1 = cm.LinearColormap(["#ffffcc", "#fd8d3c", "#800026"], vmin=min(v for v in med.values() if v), vmax=max(v for v in med.values() if v),
                           caption="Median nightly price (GBP)")
    lim = max(abs(min(v for v in prem.values() if v is not None)), abs(max(v for v in prem.values() if v is not None)))
    c2 = cm.LinearColormap(["#2166ac", "#f7f7f7", "#b2182b"], vmin=-lim, vmax=lim, caption="Premium vs city average (%)")

    def layer(name, values, cmap_, fmt, show):
        fg = folium.FeatureGroup(name=name, show=show)
        folium.GeoJson(
            gj,
            style_function=lambda f, values=values, cmap_=cmap_: {
                "fillColor": cmap_(values[f["properties"]["neighbourhood"]]) if values[f["properties"]["neighbourhood"]] is not None else "#dddddd",
                "color": "white", "weight": 0.6, "fillOpacity": 0.78},
            tooltip=folium.GeoJsonTooltip(fields=["neighbourhood", "listings", "median_price", "premium_pct_vs_city"],
                                          aliases=["Neighbourhood", "Listings", "Median price (GBP)", "Premium vs city (%)"], localize=True),
        ).add_to(fg)
        fg.add_to(m)

    layer("1. Median nightly price", med, c1, "{:.0f}", True)
    layer("2. Neighbourhood premium (model)", prem, c2, "{:+.0f}%", False)

    pts = folium.FeatureGroup(name="3. Listings: price vs model fair price", show=False)
    colour = cm.LinearColormap(["#2166ac", "#cccccc", "#b2182b"], vmin=-50, vmax=50, caption="Listing price vs model fair price (%)")
    for r in pred.itertuples():
        g = max(min(r.gap_pct, 50), -50)
        folium.CircleMarker([r.latitude, r.longitude], radius=3, color=None, weight=0, fill=True, fill_color=colour(g), fill_opacity=0.75,
                            tooltip=f"{r.name[:60]} | £{r.price_gbp:.0f} vs model £{r.model_price_gbp:.0f} ({r.gap_pct:+.0f}%) | {r.neighbourhood}").add_to(pts)
    pts.add_to(m)
    c1.add_to(m); c2.add_to(m); colour.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(OUT / "edinburgh_price_map.html"))
    print(f"map saved: {(OUT / 'edinburgh_price_map.html').stat().st_size/1e6:.1f} MB; {sum(v is not None for v in med.values())} neighbourhoods coloured")


if __name__ == "__main__":
    main()
