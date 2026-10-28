"""Bidding strategies evaluated day by day at realised prices.

Perfect foresight (upper bounds of each market or market combination):
  da_pf         day-ahead arbitrage only
  da_id_pf      day-ahead and intraday jointly
  afrr_only     sell all power as aFRR capacity in every block, no trading
  stack_pf      aFRR capacity, day-ahead and intraday jointly (upper bound of revenue stacking)

Sequential with forecasts (the order of the real markets: aFRR capacity auction, then day-ahead
auction, then intraday):
  da_naive      day-ahead schedule on a naive price forecast (d-1, or d-7 for Mon/Sat/Sun)
  da_naive_id   the same day-ahead schedule, re-optimised intraday at realised ID-AEP prices
  stack_seq     aFRR capacity on yesterday's capacity prices and the naive price forecast, then the
                day-ahead schedule on the naive forecast with the reserve fixed, then intraday
                re-optimisation at realised ID-AEP prices with reserve and day-ahead position fixed

Intraday re-optimisation at realised ID-AEP is optimistic: ID-AEP is an average over the session, not
a price available when the trade is decided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery import Battery, revenue, solve_day

ALL = frozenset({"da", "id", "afrr"})


def naive(m: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
    lag = 7 if day.day_of_week in (0, 5, 6) else 1
    return m.loc[day - pd.Timedelta(days=lag)].to_numpy(float)


def day_strategies(bat: Battery, inp: dict, day: pd.Timestamp) -> dict:
    da, idp = inp["da"].loc[day].to_numpy(float), inp["id"].loc[day].to_numpy(float)
    cp, cn = inp["cap_pos"].loc[day].to_numpy(float), inp["cap_neg"].loc[day].to_numpy(float)
    act = (da, idp, cp, cn)
    out = {}
    out["da_pf"] = revenue(bat, solve_day(bat, *act, markets=frozenset({"da"})), *act)
    out["da_id_pf"] = revenue(bat, solve_day(bat, *act, markets=frozenset({"da", "id"})), *act)
    out["afrr_only"] = revenue(bat, solve_day(bat, *act, markets=frozenset({"afrr"})), *act)
    out["stack_pf"] = revenue(bat, solve_day(bat, *act, markets=ALL), *act)

    da_f = naive(inp["da"], day)
    prev = day - pd.Timedelta(days=1)
    cp_f, cn_f = inp["cap_pos"].loc[prev].to_numpy(float), inp["cap_neg"].loc[prev].to_numpy(float)
    s = solve_day(bat, da_f, da_f, cp, cn, markets=frozenset({"da"}))           # cap prices unused here
    out["da_naive"] = revenue(bat, s, *act)
    s_id = solve_day(bat, da, idp, cp, cn, fix_q=s["q"], markets=frozenset({"da", "id"}))
    out["da_naive_id"] = revenue(bat, s_id, *act)
    s1 = solve_day(bat, da_f, da_f, cp_f, cn_f, markets=ALL)                     # stage 1: reserve
    r = (s1["r_pos"], s1["r_neg"])
    s2 = solve_day(bat, da_f, da_f, cp_f, cn_f, fix_r=r, markets=frozenset({"da", "afrr"}))   # stage 2: DA
    s3 = solve_day(bat, da, idp, cp, cn, fix_r=r, fix_q=s2["q"], markets=ALL)  # stage 3: intraday
    out["stack_seq"] = revenue(bat, s3, *act)
    out["stack_seq"]["r_pos_mean"] = float(r[0].mean())
    out["stack_seq"]["r_neg_mean"] = float(r[1].mean())
    return out


def run(bat: Battery, inp: dict, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Long table: one row per day and strategy with revenue by market (EUR per day for the battery)."""
    rows = []
    for d in days:
        if any(np.isnan(inp[k].loc[d].to_numpy(float)).any() for k in ("da", "id", "cap_pos", "cap_neg")):
            continue
        for name, rev in day_strategies(bat, inp, d).items():
            rows.append({"day": d, "strategy": name, **rev})
    return pd.DataFrame(rows)
