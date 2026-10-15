"""Step 3 - price drivers: baselines, interpretable regression, gradient boosting, neighbourhood effects.

Target: log(nightly price in GBP). Everything is evaluated on a held-out 20% test set that no model saw while fitting or tuning,
plus a harder spatial check (GroupKFold by neighbourhood: predicting areas the model has never seen).

Models
  * Baselines     : global median; median by neighbourhood x room type (train data only)
  * Mixed model   : linear regression on listing features + a random intercept per neighbourhood (shrinks small areas
                    towards the city average, so tiny neighbourhoods are not over-interpreted). Gives % price effects with CIs.
  * LightGBM      : gradient boosting, tuned with randomised search; explained with SHAP.

Outputs (data/processed/ and outputs/): model_metrics.json, ols_coefficients.csv, neighbourhood_effects.csv,
  shap_importance.csv, listing_predictions.csv, fig_*.png
"""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, train_test_split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG = ROOT / "outputs"
SEED = 42
BLUE, ORANGE, GREY, RED = "#2F5D8A", "#E07B39", "#7A7A7A", "#B5484C"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6",
                     "axes.axisbelow": True, "figure.dpi": 130, "font.size": 9})

AMEN = ["has_wifi", "has_kitchen", "has_washer", "has_dryer", "has_dishwasher", "has_parking", "has_hot_tub", "has_pool",
        "has_air_con", "has_workspace", "has_ev_charger", "has_gym", "has_lift", "has_fireplace", "has_bbq", "has_crib", "has_self_checkin"]
NUM = ["accommodates", "bedrooms_n", "beds_n", "bathrooms_n", "shared_bathroom", "n_amenities", "host_years", "is_superhost",
       "host_listings", "multi_listing_host", "has_licence_field", "min_nights", "availability_365_n",
       "n_reviews", "reviews_per_month_n", "review_scores_rating_n", "review_scores_location_n", "review_scores_value_n",
       "review_scores_cleanliness_n", "has_reviews", "dist_centre_km", "latitude", "longitude", *AMEN]
LABELS = {
    "accommodates": "Guests accommodated", "bedrooms_n": "Bedrooms", "beds_n": "Beds", "bathrooms_n": "Bathrooms",
    "shared_bathroom": "Shared bathroom", "n_amenities": "Number of amenities", "host_years": "Host experience (years)",
    "is_superhost": "Superhost", "host_listings": "Host's listings", "multi_listing_host": "Multi-listing host",
    "has_licence_field": "Licence number shown", "min_nights": "Minimum nights",
    "availability_365_n": "Days available (next 365)", "n_reviews": "Number of reviews", "reviews_per_month_n": "Reviews per month",
    "review_scores_rating_n": "Rating (overall)", "review_scores_location_n": "Rating (location)", "review_scores_value_n": "Rating (value)",
    "review_scores_cleanliness_n": "Rating (cleanliness)", "has_reviews": "Has reviews", "dist_centre_km": "Distance to Old Town (km)",
    "latitude": "Latitude", "longitude": "Longitude", "room_type": "Room type", "building_type": "Building type",
    "neighbourhood": "Neighbourhood", **{a: a[4:].replace("_", " ").capitalize() for a in AMEN},
}


def metrics(y_log, pred_log) -> dict:
    y, p = np.exp(y_log), np.exp(pred_log)
    return {"r2_log_price": round(float(r2_score(y_log, pred_log)), 3),
            "mae_gbp": round(float(mean_absolute_error(y, p)), 1),
            "median_abs_error_gbp": round(float(median_absolute_error(y, p)), 1),
            "median_abs_pct_error": round(float(np.median(np.abs(p - y) / y) * 100), 1)}


def mixed_formula() -> str:
    terms = ["C(room_type)", "C(building_type)", "accommodates", "bedrooms_n", "bathrooms_n", "shared_bathroom",
             "np.log1p(dist_centre_km)", "n_amenities", "is_superhost", "multi_listing_host", "has_licence_field",
             "np.log(min_nights)", "has_reviews", "np.log1p(n_reviews)", "review_scores_rating_n", "review_scores_location_n",
             "review_scores_value_n", "review_scores_cleanliness_n", "host_years", "np.log1p(availability_365_n)",
             "has_parking", "has_hot_tub", "has_air_con", "has_workspace", "has_lift", "has_dishwasher", "has_ev_charger", "has_gym", "has_fireplace"]
    return "log_price ~ " + " + ".join(terms)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "listings_clean.csv")
    for c in ["room_type", "building_type", "neighbourhood"]:
        df[c] = df[c].astype("category")
    train, test = train_test_split(df, test_size=0.2, random_state=SEED)
    out: dict = {"n_listings": len(df), "n_train": len(train), "n_test": len(test)}

    # ---------------- baselines -------------------------------------------------------
    glob = train["log_price"].median()
    out["baseline_global_median"] = metrics(test["log_price"], np.full(len(test), glob))
    key = train.groupby(["neighbourhood", "room_type"], observed=True)["log_price"].median()
    rt = train.groupby("room_type", observed=True)["log_price"].median()
    pred_nb = np.array([key.get((n, r), rt[r]) for n, r in zip(test["neighbourhood"], test["room_type"])])
    out["baseline_neighbourhood_roomtype_median"] = metrics(test["log_price"], pred_nb)

    # ---------------- mixed model (interpretable) -----------------------------------
    f = mixed_formula()
    ols = smf.ols(f, data=train).fit(cov_type="HC3")
    out["ols_fixed_effects_only"] = metrics(test["log_price"], ols.predict(test))
    mm = smf.mixedlm(f, data=train, groups=train["neighbourhood"].astype(str)).fit(method=["powell", "lbfgs"], reml=True)
    re = {k: float(v.iloc[0]) for k, v in mm.random_effects.items()}
    fixed_pred = mm.predict(test)
    pred_mm = fixed_pred + test["neighbourhood"].astype(str).map(re).fillna(0).to_numpy()
    out["mixed_model_with_neighbourhood_effects"] = metrics(test["log_price"], pred_mm)
    share_nb = float(mm.cov_re.iloc[0, 0] / (mm.cov_re.iloc[0, 0] + mm.scale))
    out["share_of_unexplained_variance_between_neighbourhoods"] = round(share_nb, 3)

    # coefficient table from the robust OLS (same fixed effects), % effect on price
    ci = ols.conf_int()
    coef = pd.DataFrame({"term": ols.params.index, "coef_log": ols.params.values, "pct_effect": (np.exp(ols.params.values) - 1) * 100,
                         "ci_low_pct": (np.exp(ci[0].values) - 1) * 100, "ci_high_pct": (np.exp(ci[1].values) - 1) * 100,
                         "p_value": ols.pvalues.values})
    coef = coef[coef["term"] != "Intercept"].copy()
    # log-transformed terms are elasticities (a 1% rise in X changes price by about coef_log %), not per-unit effects
    coef["interpretation"] = np.where(coef["term"].str.startswith("np.log"), "elasticity: +1% in variable -> ~coef_log % price", "% price change per unit / vs reference level")
    coef = coef.round(4)
    coef.to_csv(DATA / "ols_coefficients.csv", index=False)

    # neighbourhood effects (shrunken random intercepts) -> % premium vs city average
    nb = df.groupby("neighbourhood", observed=True).agg(listings=("id", "size"), median_price=("price_gbp", "median"),
                                                         mean_log_price=("log_price", "mean")).reset_index()
    nb["premium_pct_vs_city"] = nb["neighbourhood"].astype(str).map(lambda k: (np.exp(re.get(k, 0.0)) - 1) * 100).round(1)
    nb["reliable"] = nb["listings"] >= 10
    nb.sort_values("premium_pct_vs_city", ascending=False).to_csv(DATA / "neighbourhood_effects.csv", index=False)

    # ---------------- gradient boosting ----------------------------------------------
    feats = [*NUM, "room_type", "building_type", "neighbourhood"]
    X, y = df[feats], df["log_price"]
    Xtr, Xte, ytr, yte = X.loc[train.index], X.loc[test.index], y.loc[train.index], y.loc[test.index]
    base = lgb.LGBMRegressor(objective="regression", n_estimators=600, random_state=SEED, verbose=-1, n_jobs=4,
                             cat_smooth=40, min_data_per_group=20, cat_l2=20)
    search = RandomizedSearchCV(base, {"learning_rate": [0.02, 0.03, 0.05], "num_leaves": [8, 16, 24, 32], "min_child_samples": [10, 20, 40],
                                       "subsample": [0.7, 0.85, 1.0], "subsample_freq": [1], "colsample_bytree": [0.5, 0.7, 0.9],
                                       "reg_lambda": [0.0, 1.0, 5.0, 20.0]},
                                n_iter=20, cv=KFold(5, shuffle=True, random_state=SEED), scoring="neg_mean_absolute_error",
                                random_state=SEED, n_jobs=1)
    search.fit(Xtr, ytr)
    gbm = search.best_estimator_
    out["gbm_best_params"] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in search.best_params_.items()}
    out["gbm_test"] = metrics(yte, gbm.predict(Xte))

    # spatial generalisation: predict neighbourhoods the model has never seen (neighbourhood feature removed)
    feats_sp = [f_ for f_ in feats if f_ != "neighbourhood"]
    oof_sp = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(df, y, df["neighbourhood"].astype(str)):
        m = lgb.LGBMRegressor(**{**gbm.get_params(), "n_estimators": gbm.n_estimators}).fit(df.iloc[tr][feats_sp], y.iloc[tr])
        oof_sp[te] = m.predict(df.iloc[te][feats_sp])
    out["gbm_unseen_neighbourhoods_groupcv"] = metrics(y, oof_sp)

    # attributes-only model: drop demand-side signals (reviews, availability) that partly REACT to price rather than cause it
    demand = ["n_reviews", "reviews_per_month_n", "availability_365_n", "has_reviews"]
    feats_attr = [f_ for f_ in feats if f_ not in demand]
    m_attr = lgb.LGBMRegressor(**gbm.get_params()).fit(Xtr[feats_attr], ytr)
    out["gbm_listing_attributes_only_test"] = metrics(yte, m_attr.predict(Xte[feats_attr]))

    # out-of-fold fair-price estimate for every listing (used for the 'priced above/below model' map layer)
    oof = np.zeros(len(df))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(df):
        m = lgb.LGBMRegressor(**gbm.get_params()).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    out["gbm_5fold_oof"] = metrics(y, oof)
    pred = df[["id", "name", "neighbourhood", "latitude", "longitude", "room_type", "accommodates", "price_gbp"]].copy()
    pred["model_price_gbp"] = np.exp(oof).round(1)
    pred["gap_pct"] = ((pred["price_gbp"] / pred["model_price_gbp"] - 1) * 100).round(1)
    pred.to_csv(DATA / "listing_predictions.csv", index=False)

    # ---------------- SHAP --------------------------------------------------------------
    expl = shap.TreeExplainer(gbm)
    sv = expl.shap_values(Xte)
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp["label"] = imp["feature"].map(LABELS)
    imp["approx_pct_swing"] = ((np.exp(imp["mean_abs_shap"]) - 1) * 100).round(1)
    imp.to_csv(DATA / "shap_importance.csv", index=False)

    # ---------------- figures --------------------------------------------------------------
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(top["label"], top["mean_abs_shap"], color=BLUE)
    ax.set_title("What moves price most? Mean |SHAP| on log-price (LightGBM, test set)", loc="left", fontsize=9)
    ax.set_xlabel("Average absolute effect on log(price)")
    fig.tight_layout(); fig.savefig(FIG / "fig_shap_importance.png"); plt.close(fig)

    names = [LABELS.get(c, c) for c in feats]
    plt.figure(figsize=(8, 5.6))
    shap.summary_plot(sv, Xte.rename(columns=dict(zip(feats, names))), max_display=15, show=False, plot_size=None)
    plt.title("SHAP summary: direction and size of each driver", loc="left", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "fig_shap_summary.png"); plt.close()

    for var, fname, xl in [("dist_centre_km", "fig_shap_distance.png", "Distance to Old Town (km)"), ("accommodates", "fig_shap_guests.png", "Guests accommodated")]:
        i = feats.index(var)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.scatter(Xte[var], sv[:, i], s=8, alpha=0.5, color=BLUE)
        ax.set_xlabel(xl); ax.set_ylabel("Effect on log(price)"); ax.axhline(0, color=GREY, lw=1)
        ax.set_title(f"Partial effect of {xl.lower()}", loc="left", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)

    c = coef.copy()
    c["label"] = c["term"].str.replace(r"C\((\w+)\)\[T\.(.*)\]", lambda m: f"{LABELS.get(m.group(1), m.group(1))}: {m.group(2)}", regex=True)
    c["label"] = c["label"].map(lambda t: LABELS.get(t, t))
    c = c[(c["p_value"] < 0.05) & ~c["term"].str.startswith("np.log")]
    c = c.loc[c["pct_effect"].abs().sort_values(ascending=False).index].head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(c["pct_effect"], range(len(c)), xerr=[c["pct_effect"] - c["ci_low_pct"], c["ci_high_pct"] - c["pct_effect"]],
                fmt="o", color=BLUE, ecolor=GREY, capsize=2)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c["label"]); ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("% change in nightly price, holding everything else fixed (95% CI)")
    ax.set_title("Largest significant price effects (linear model; per unit or vs reference level)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_ols_effects.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(np.exp(yte), np.exp(gbm.predict(Xte)), s=8, alpha=0.5, color=BLUE)
    lim = [20, 800]; ax.plot(lim, lim, color=ORANGE, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual nightly price (GBP)"); ax.set_ylabel("Predicted (GBP)")
    ax.set_title("Held-out test listings: predicted vs actual", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_pred_vs_actual.png"); plt.close(fig)

    rows = [("Global median", out["baseline_global_median"]), ("Neighbourhood x room-type median", out["baseline_neighbourhood_roomtype_median"]),
            ("Linear (fixed effects only)", out["ols_fixed_effects_only"]), ("Mixed model (+ neighbourhood effects)", out["mixed_model_with_neighbourhood_effects"]),
            ("LightGBM", out["gbm_test"])]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh([r[0] for r in rows][::-1], [r[1]["median_abs_error_gbp"] for r in rows][::-1], color=[BLUE if "LightGBM" not in r[0] else ORANGE for r in rows][::-1])
    for i, r in enumerate(rows[::-1]):
        ax.text(r[1]["median_abs_error_gbp"], i, f"  £{r[1]['median_abs_error_gbp']:.0f}  (R² {r[1]['r2_log_price']:.2f})", va="center", fontsize=8)
    ax.set_xlim(0, max(r[1]["median_abs_error_gbp"] for r in rows) * 1.35)
    ax.set_title("Median absolute error per night on unseen test listings (lower is better)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_model_comparison.png"); plt.close(fig)

    (DATA / "model_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "gbm_best_params"}, indent=1))


if __name__ == "__main__":
    main()
