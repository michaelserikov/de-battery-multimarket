import numpy as np
import pandas as pd
import pytest

from bessmm import data
from bessmm.battery import Battery, revenue, solve_day
from bessmm.strategies import ALL, day_strategies

BAT = Battery()
FLAT = np.full(24, 50.0)
ZERO6 = np.zeros(6)


def spread():
    p = np.full(24, 50.0)
    p[2:4], p[18:20] = 0.0, 200.0
    return p


def test_da_arbitrage_charges_low_and_discharges_high():
    s = solve_day(BAT, spread(), spread(), ZERO6, ZERO6, markets=frozenset({"da"}))
    assert s["c"][2:4].sum() > 1.5 and s["d"][18:20].sum() > 1.5
    assert np.allclose(s["q"], s["d"] - s["c"])                      # no intraday leg
    assert s["soc"][-1] == pytest.approx(1.0)                         # ends half full
    assert (np.minimum(s["c"], s["d"]) < 1e-9).all()


def test_flat_prices_no_trade_and_afrr_only_sells_full_power():
    s = solve_day(BAT, FLAT, FLAT, ZERO6, ZERO6, markets=frozenset({"da", "id"}))
    assert s["d"].sum() < 1e-9
    a = solve_day(BAT, FLAT, FLAT, np.full(6, 5.0), np.full(6, 3.0), markets=frozenset({"afrr"}))
    assert np.allclose(a["r_pos"], 1) and np.allclose(a["r_neg"], 1)
    rev = revenue(BAT, a, FLAT, FLAT, np.full(6, 5.0), np.full(6, 3.0))
    assert rev["afrr"] == pytest.approx(4 * 6 * (5 + 3))


def test_reserve_limits_trading_headroom():
    s = solve_day(BAT, spread(), spread(), np.full(6, 1000.0), np.full(6, 1000.0), markets=ALL)
    assert np.allclose(s["r_pos"], 1) and s["d"].sum() < 1e-6        # high capacity prices: keep power for aFRR


def test_fixed_decisions_are_respected():
    q = solve_day(BAT, spread(), spread(), ZERO6, ZERO6, markets=frozenset({"da"}))["q"]   # a deliverable plan
    s = solve_day(BAT, spread(), FLAT, ZERO6, ZERO6, fix_q=q, markets=ALL)
    assert np.allclose(s["q"], q)
    rev = revenue(BAT, s, spread(), FLAT, ZERO6, ZERO6)
    assert rev["total"] == pytest.approx(rev["da"] + rev["id"] + rev["afrr"] + rev["degradation"])


def test_stacking_is_an_upper_bound(inp_days):
    inp, day = inp_days
    out = day_strategies(BAT, inp, day)
    tot = {k: v["total"] for k, v in out.items()}
    assert tot["stack_pf"] >= max(tot["da_pf"], tot["da_id_pf"], tot["afrr_only"], tot["stack_seq"]) - 1e-6
    assert tot["da_id_pf"] >= tot["da_pf"] - 1e-6
    assert tot["da_naive_id"] >= tot["da_naive"] - 1e-6               # intraday re-optimisation never loses


@pytest.fixture
def inp_days():
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-03-01", periods=10)
    base = 60 + 40 * np.sin((np.arange(24) - 7) / 24 * 2 * np.pi)
    da = pd.DataFrame(base + rng.normal(0, 15, (10, 24)), index=days)
    idp = da + rng.normal(0, 10, (10, 24))
    cp = pd.DataFrame(rng.uniform(2, 12, (10, 6)), index=days)
    cn = pd.DataFrame(rng.uniform(1, 8, (10, 6)), index=days)
    return {"da": da, "id": idp, "cap_pos": cp, "cap_neg": cn}, days[8]


def test_day_ahead_plan_must_be_deliverable():
    da = np.full(24, 100.0)
    idp = np.zeros(24)                # selling day-ahead and buying back intraday would pay 100 every hour
    s = solve_day(BAT, da, idp, ZERO6, ZERO6, markets=frozenset({"da", "id"}))
    assert s["q"].sum() <= BAT.energy * BAT.eta + 1e-6   # the plan cannot sell more than the stored energy


def test_parse_afrr_and_idaep():
    raw = pd.DataFrame({"PRODUCT": ["POS_00_04", "NEG_20_24"], "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]": [5.85, 3.1]})
    p = data.parse_afrr(raw)
    assert p["pos"][0] == 5.85 and p["neg"][5] == 3.1 and np.isnan(p["pos"][1])
    text = "Datum von;(Uhrzeit) von;Zeitzone;(Uhrzeit) bis;Zeitzone;ID AEP in EUR/MWh\n2025-06-01;00:00;UTC;00:15;UTC;61,18\n"
    s = data.parse_idaep(text)
    assert s.iloc[0] == pytest.approx(61.18) and s.index[0] == pd.Timestamp("2025-06-01", tz="UTC")


def test_daily_matrix_local_days():
    idx = pd.date_range("2025-03-29 23:00", periods=48, freq="h", tz="UTC")   # spans the spring clock change
    m = data.daily_matrix(pd.Series(np.arange(48.0), index=idx))
    assert list(m.columns) == list(range(24)) and not m.isna().any().any()
