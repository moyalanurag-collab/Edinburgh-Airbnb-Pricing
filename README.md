# Edinburgh Airbnb Pricing: what drives nightly price?

Which features, locations and seasons drive Airbnb prices in Edinburgh, and how well can price be predicted? The project cleans the public Inside Airbnb data, fits an interpretable mixed-effects regression and a gradient-boosting model (explained with SHAP), measures seasonality, and maps price and neighbourhood effects.

| | |
|---|---|
| **Data** | [Inside Airbnb](https://insideairbnb.com/get-the-data/), Edinburgh, scrape of 19 Sep 2026 - 6,342 listings, 5,138 modelled. Licence **CC BY 4.0** |
| **Methods** | Python: pandas, statsmodels (OLS + mixed model), LightGBM, SHAP, folium |
| **Headline** | Median £215/night. Boosting predicts a held-out listing's price to within a median **£32 (16.6%)**, vs £52 for a neighbourhood x room-type median |
| **Tests** | 5 pytest tests on the cleaning and feature code |

![Median price by neighbourhood](outputs/fig_map_median_price.png)

## Key findings (all computed in `src/`, detail in [docs/findings.md](docs/findings.md))
* **Size and type dominate.** Room type (private room is about 29% cheaper), guests accommodated and bedrooms are the top three drivers, then **distance to the Old Town** (each doubling of 1 + distance in km goes with about 16% lower price, elasticity -0.25).
* **Location matters beyond distance, but less than you'd think.** After controlling for size, quality and distance, neighbourhood explains only about **9%** of the remaining price variation. A mixed model with shrinkage gives a premium for 88 neighbourhoods with 10+ listings (largest premiums: Dalmeny/Kirkliston +32%, New Town West +21%; lowest: Colinton Mains/Firrhill -15%, Restalrig/Lochend -13%, Gracemount area -12%).
* **Seasonality is strong.** Guest demand (review volume, normalised per year) peaks in **August** (index 1.43, the Festival) and bottoms out in **January** (0.46) - a 3x swing; May to September all run above average.
* **Prediction.** LightGBM R² 0.74 on log price (median error £32/night); it still reaches R² 0.72 on *neighbourhoods it has never seen*, so it is learning transferable drivers. Dropping demand-side signals (reviews, availability) leaves R² 0.69.
* **A pricing lens.** Each listing gets an out-of-fold "fair price"; the map's third layer shows which listings sit far above or below it.

![Seasonality](outputs/fig_seasonality.png)

![SHAP summary](outputs/fig_shap_summary.png)

## Interactive map
Open [`outputs/edinburgh_price_map.html`](outputs/edinburgh_price_map.html) in a browser (needs internet for the base map). Layers: (1) median price by neighbourhood, (2) model-based neighbourhood premium, (3) every listing coloured by price vs model fair price. Neighbourhoods with fewer than 10 listings are greyed out.

For Power BI or Tableau, the same data is ready in `data/processed/`:
* `neighbourhoods_enriched.geojson` - polygons with listings, median price and premium (Power BI Shape Map / Tableau spatial file / QGIS)
* `listings_for_bi.csv` - one row per listing with coordinates, price, model fair price and gap
* `neighbourhood_effects.csv`, `shap_importance.csv`, `seasonality_*.csv`

## Reproduce
```bash
pip install -r requirements.txt
python run_all.py --download     # downloads the 5 Inside Airbnb files, then runs every step (~2 min)
python -m pytest -q
```

Steps: `prepare.py` (cleaning + features) -> `seasonality.py` -> `model.py` (baselines, mixed model, LightGBM, SHAP) -> `make_map.py` -> `report.py`. To use Glasgow instead, change the URL in `run_all.py` to the Glasgow folder on Inside Airbnb and the centre point in `prepare.py`.

## Method notes
* **Target:** log(nightly price). Cleaning drops listings with no price (17%), prices outside £20-£1,500, minimum stay over 28 nights, and hotel/shared rooms (too few); each rule is logged in `data/processed/cleaning_log.json`.
* **Features:** capacity, bathrooms, 17 amenity flags (text-matched), host profile, minimum nights, review scores, distance to the Old Town, coordinates, building type vs room type.
* **Evaluation:** a 20% test set untouched by fitting and tuning; boosting tuned by randomised 5-fold search on the training set; plus a spatial check (GroupKFold by neighbourhood).
* **Interpretation:** the linear/mixed model gives % effects with confidence intervals; SHAP shows direction and size in the boosting model. Both are associations, not causal effects.

## Limitations
* Single snapshot of *asking* prices; the current calendar file carries no nightly prices, so **seasonality is demand-based** (reviews and forward availability), not price-based. "Unavailable" includes host-blocked nights.
* Reviews, ratings and availability partly *respond to* price, so they improve prediction but are not levers a host could pull. The attributes-only model separates this.
* `host_since` and `instant_bookable` are empty in this release and are not used. Prices carry a `$` symbol in the source file and are treated as GBP.
* Amenities are matched from free text and may miss synonyms.

## Attribution
Contains data from [Inside Airbnb](https://insideairbnb.com), licensed under CC BY 4.0. Raw files are not committed; `run_all.py --download` fetches them. Code is MIT licensed.
