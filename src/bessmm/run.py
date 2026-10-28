"""Run the study: 1 MW / 2 MWh battery, January 2025 to September 2026.

    python -m bessmm.run                 # about 10 minutes (seven strategies, ~640 days, plus sensitivities)

Writes results/ (daily revenue by strategy, annualised summary, sensitivity to the reserve duration tau,
monthly aFRR capacity prices) and figures/.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

from . import data
from .battery import Battery
from .strategies import run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

START, END = "2025-01-01", "2026-09-30"
ORDER = ["da_naive", "da_naive_id", "da_pf", "da_id_pf", "afrr_only", "stack_seq", "stack_pf"]


def annualise(daily: pd.DataFrame, power: float) -> pd.DataFrame:
    """Mean daily revenue x 365 per MW, by strategy (thousand EUR per MW and year), with the market split."""
    cols = ["da", "id", "afrr", "degradation", "total"]
    g = daily.groupby("strategy")[cols].mean() * 365 / power / 1000
    g["cycles_per_day"] = daily.groupby("strategy")["cycles"].mean()
    g["days"] = daily.groupby("strategy").size()
    return g.reindex([s for s in ORDER if s in g.index])


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args(argv)
    res, figs = data.ROOT / "results", data.ROOT / "figures"
    res.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    inp = data.load_all(args.start, args.end)
    days = pd.date_range(args.start, args.end, freq="D")
    bat = Battery()

    daily = run(bat, inp, days)
    daily.to_csv(res / "daily_revenue.csv", index=False, float_format="%.2f")
    daily["year"] = daily["day"].dt.year
    summary = {"battery": bat.__dict__, "period": [args.start, args.end], "days": int(daily["day"].nunique())}
    tables = {}
    for label, sel in [("all", daily), *[(str(y), daily[daily["year"] == y]) for y in sorted(daily["year"].unique())]]:
        t = annualise(sel, bat.power)
        t.to_csv(res / f"annual_{label}.csv", float_format="%.2f")
        tables[label] = t.round(2).to_dict(orient="index")
    summary["kEUR_per_MW_year"] = tables
    pf = daily[daily["strategy"] == "stack_pf"].set_index("day")
    seq = daily[daily["strategy"] == "stack_seq"].set_index("day")
    summary["stack_seq_share_of_stack_pf"] = float(seq["total"].sum() / pf["total"].sum())
    summary["stack_seq_mean_reserve_MW"] = {"pos": float(seq["r_pos_mean"].mean()), "neg": float(seq["r_neg_mean"].mean())}

    sens = []
    for tau in (0.5, 1.0, 2.0):
        b = Battery(tau=tau)
        sub = days[::7]                                                  # every 7th day keeps it quick
        d = run(b, inp, sub)
        t = annualise(d[d["strategy"].isin(["stack_pf", "stack_seq"])], b.power)
        for s, row in t.iterrows():
            sens.append({"tau_h": tau, "strategy": s, "total": row["total"], "afrr": row["afrr"]})
    sens = pd.DataFrame(sens)
    sens.to_csv(res / "sensitivity_tau.csv", index=False, float_format="%.2f")
    summary["sensitivity_tau_every_7th_day"] = sens.round(2).to_dict(orient="records")

    cap = pd.DataFrame({"pos": inp["cap_pos"].loc[args.start:args.end].mean(axis=1),
                        "neg": inp["cap_neg"].loc[args.start:args.end].mean(axis=1)})
    capm = cap.resample("MS").mean()
    capm.to_csv(res / "afrr_capacity_monthly.csv", float_format="%.2f")
    summary["afrr_capacity_price_monthly_mean"] = {"first": capm.iloc[0].round(2).to_dict(), "last": capm.iloc[-1].round(2).to_dict()}
    spread = (inp["da"].loc[args.start:args.end].max(axis=1) - inp["da"].loc[args.start:args.end].min(axis=1)).resample("MS").mean()
    (res / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # figures
    m = daily.assign(month=daily["day"].dt.to_period("M").dt.to_timestamp()).groupby(["month", "strategy"])["total"].mean().unstack() * 30.4 / bat.power / 1000
    fig, ax = plt.subplots(figsize=(9, 4))
    for s in ["da_pf", "da_id_pf", "afrr_only", "stack_seq", "stack_pf"]:
        ax.plot(m.index, m[s], label=s)
    ax.set(ylabel="thousand EUR per MW and month", title="Monthly revenue of a 1 MW / 2 MWh battery by strategy")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "monthly_revenue.png", dpi=150)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(capm.index, capm["pos"], label="aFRR capacity price, up")
    ax1.plot(capm.index, capm["neg"], label="aFRR capacity price, down")
    ax1.set_ylabel("EUR per MW and hour")
    ax2 = ax1.twinx()
    ax2.plot(spread.index, spread, color="grey", ls="--", label="day-ahead daily max - min")
    ax2.set_ylabel("EUR/MWh")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.set_title("aFRR capacity prices and day-ahead spreads")
    fig.tight_layout()
    fig.savefig(figs / "capacity_prices_and_spreads.png", dpi=150)
    t = annualise(daily, bat.power)
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(t))
    for col in ["da", "id", "afrr", "degradation"]:
        ax.bar(t.index, t[col], bottom=np.where(t[col] >= 0, bottom, 0), label=col)
        bottom += np.where(t[col] >= 0, t[col], 0)
    ax.set(ylabel="thousand EUR per MW and year", title="Revenue by market and strategy, Jan 2025 - Sep 2026")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "revenue_by_market.png", dpi=150)
    return summary


if __name__ == "__main__":
    main()
