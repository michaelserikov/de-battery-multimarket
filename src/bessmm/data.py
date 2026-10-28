"""Market data, cached on disk: day-ahead prices (SMARD), intraday index ID-AEP (netztransparenz.de)
and aFRR capacity prices (regelleistung.net). All series are turned into local (Europe/Berlin) days.

- Day-ahead: SMARD filter 4169, bidding zone DE-LU, hourly, EUR/MWh (CC BY 4.0, Bundesnetzagentur | SMARD.de).
- ID-AEP: volume-weighted price of intraday trades per quarter hour, from the TSOs' netztransparenz.de
  WebAPI (free OAuth client; environment variables NTP_CLIENT_ID and NTP_CLIENT_SECRET); averaged to hours.
- aFRR capacity: daily results of the German-Austrian aFRR capacity auction (regelleistung.net, D-1),
  six four-hour products per direction (POS_00_04 ... NEG_20_24), German average and marginal
  capacity prices in EUR per MW and hour.

The SMARD and netztransparenz clients are adapted from github.com/michaelserikov/de-power-market-ppa
and github.com/michaelserikov/de-weather-to-power (same author).
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(os.environ.get("BESSMM_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data" / "raw"
TZ = "Europe/Berlin"
HOURS = list(range(24))
UA = {"User-Agent": "de-battery-multimarket (research; github.com/michaelserikov/de-battery-multimarket)"}
_session = requests.Session()
_session.headers.update(UA)


def _get(url: str, retries: int = 4, **kw) -> requests.Response:
    for attempt in range(retries):
        r = _session.get(url, timeout=60, **kw)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed: {url}")


# --- local daily matrices ----------------------------------------------------------------------------

def daily_matrix(s: pd.Series) -> pd.DataFrame:
    """Hourly UTC series -> local days x 24 local hours (clock-change days interpolated or averaged)."""
    local = s.index.tz_convert(TZ)
    g = pd.DataFrame({"day": local.tz_localize(None).normalize(), "hour": local.hour, "v": s.to_numpy()})
    m = g.groupby(["day", "hour"])["v"].mean().unstack("hour").reindex(columns=HOURS)
    return m.interpolate(axis=1, limit_direction="both")


# --- SMARD day-ahead ---------------------------------------------------------------------------------

SMARD = "https://www.smard.de/app/chart_data/4169/DE-LU"


def day_ahead(start: str, end: str, cache: Path = RAW) -> pd.DataFrame:
    """Day-ahead prices as local days x 24 hours."""
    folder = cache / "4169_DE-LU"
    folder.mkdir(parents=True, exist_ok=True)
    t0, t1 = pd.Timestamp(start, tz=TZ).tz_convert("UTC"), (pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)).tz_convert("UTC")
    try:
        starts = _get(f"{SMARD}/index_hour.json").json()["timestamps"]
    except requests.RequestException:  # SMARD unreachable: use cached weekly chunks
        starts = sorted(int(p.stem) for p in folder.glob("*.json"))
    rows = []
    week = pd.Timedelta(days=7)
    for ts in starts:
        c0 = pd.Timestamp(ts, unit="ms", tz="UTC")
        if c0 + week <= t0 or c0 >= t1:
            continue
        path = folder / f"{ts}.json"
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
        else:
            doc = _get(f"{SMARD}/4169_DE-LU_hour_{ts}.json").json()
            if c0 + week < pd.Timestamp.now(tz="UTC"):
                path.write_text(json.dumps(doc), encoding="utf-8")
            time.sleep(0.2)
        rows += [(t, v) for t, v in doc.get("series", []) if v is not None]
    s = pd.Series({pd.Timestamp(t, unit="ms", tz="UTC"): v for t, v in rows}).sort_index()
    s = s[(s.index >= t0) & (s.index < t1)]
    return daily_matrix(s).loc[start:end]


# --- netztransparenz ID-AEP --------------------------------------------------------------------------

NTP_TOKEN = "https://identity.netztransparenz.de/users/connect/token"
NTP_BASE = "https://ds.netztransparenz.de/api/v1/data/IdAep"
_token: dict = {}


def _bearer() -> str:
    if _token.get("expires", 0) < time.time() + 60:
        cid, secret = os.environ.get("NTP_CLIENT_ID"), os.environ.get("NTP_CLIENT_SECRET")
        if not cid or not secret:
            raise RuntimeError("set NTP_CLIENT_ID and NTP_CLIENT_SECRET (free WebAPI client from api-portal.netztransparenz.de)")
        r = _session.post(NTP_TOKEN, data={"grant_type": "client_credentials", "client_id": cid, "client_secret": secret}, timeout=30)
        r.raise_for_status()
        doc = r.json()
        _token.update(value=doc["access_token"], expires=time.time() + doc.get("expires_in", 3600))
    return _token["value"]


def parse_idaep(text: str) -> pd.Series:
    """CSV 'Datum von;(Uhrzeit) von;Zeitzone;(Uhrzeit) bis;Zeitzone;ID AEP in EUR/MWh' -> series by quarter start (UTC)."""
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    if df.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    cols = list(df.columns)
    idx = pd.to_datetime(df[cols[0]] + " " + df[cols[1]], format="%Y-%m-%d %H:%M", utc=True)
    val = pd.to_numeric(df[cols[-1]].str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    s = pd.Series(val.to_numpy(), index=idx)
    return s[~s.index.duplicated()].sort_index()


def intraday(start: str, end: str, cache: Path = RAW) -> pd.DataFrame:
    """ID-AEP averaged to hours, as local days x 24 hours (one cached CSV per past month)."""
    folder = cache / "netztransparenz" / "idaep"
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for m0 in pd.date_range((pd.Timestamp(start) - pd.Timedelta(days=1)).replace(day=1), end, freq="MS"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        path = folder / f"{m0:%Y-%m}.csv"
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            r = _get(f"{NTP_BASE}/{m0:%Y-%m-%dT%H:%M:%S}/{m1:%Y-%m-%dT%H:%M:%S}", headers={"Authorization": f"Bearer {_bearer()}"})
            text = r.content.decode("utf-8-sig")
            if m1 < pd.Timestamp.now():
                path.write_text(text, encoding="utf-8")
            time.sleep(1)
        parts.append(parse_idaep(text))
    q = pd.concat(parts)
    q = q[~q.index.duplicated()]
    return daily_matrix(q.resample("h").mean()).loc[start:end]


# --- regelleistung.net aFRR capacity -----------------------------------------------------------------

RL_URL = ("https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/resultsoverview"
          "?date={day}&exportFormat=xlsx&market=CAPACITY&productTypes=aFRR")
BLOCKS = list(range(6))  # 00-04, 04-08, ..., 20-24 local time


def parse_afrr(raw: pd.DataFrame, price: str = "AVERAGE") -> dict:
    """One day's results -> {'pos': [6 prices], 'neg': [6 prices]} in EUR/MW/h (German price column)."""
    col = next(c for c in raw.columns if c.startswith(f"GERMANY_{price}_CAPACITY_PRICE"))
    out = {"pos": [np.nan] * 6, "neg": [np.nan] * 6}
    for prod, v in zip(raw["PRODUCT"].astype(str), pd.to_numeric(raw[col], errors="coerce")):
        direction, h0 = prod.split("_")[0].lower(), int(prod.split("_")[1])
        if direction in out and h0 % 4 == 0:
            out[direction][h0 // 4] = float(v)
    return out


def afrr_capacity(start: str, end: str, cache: Path = RAW, price: str = "AVERAGE") -> dict:
    """aFRR capacity prices: {'pos': days x 6 blocks, 'neg': days x 6 blocks} in EUR/MW/h."""
    folder = cache / "regelleistung" / "afrr_capacity"
    folder.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    pos, neg = [], []
    for d in days:
        path = folder / f"{d:%Y-%m-%d}.xlsx"
        if not path.exists():
            r = _get(RL_URL.format(day=f"{d:%Y-%m-%d}"))
            path.write_bytes(r.content)
            time.sleep(0.3)
        p = parse_afrr(pd.read_excel(path), price)
        pos.append(p["pos"])
        neg.append(p["neg"])
    return {"pos": pd.DataFrame(pos, index=days, columns=BLOCKS), "neg": pd.DataFrame(neg, index=days, columns=BLOCKS)}


def load_all(start: str, end: str, cache: Path = RAW) -> dict:
    """All market inputs for local days start..end, with one week of history for naive forecasts."""
    hist = (pd.Timestamp(start) - pd.Timedelta(days=7)).date().isoformat()
    return {"da": day_ahead(hist, end, cache), "id": intraday(hist, end, cache), **{f"cap_{k}": v for k, v in afrr_capacity(hist, end, cache).items()}}
