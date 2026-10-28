"""Daily mixed-integer model of a battery that can sell aFRR capacity and trade day-ahead and intraday.

Decisions for one local day (24 hours, six four-hour aFRR blocks):
  r_pos[b], r_neg[b]   aFRR capacity reserved upward / downward in block b (MW), paid the capacity price
  q[t]                 day-ahead net position in hour t (MW, sale positive), settled at the day-ahead price
  c0, d0, soc0, u0     the day-ahead plan: q = d0 - c0 must itself be a feasible schedule of the battery
                       (same power, energy and reserve limits), so no position is sold that the battery
                       could not deliver
  c[t], d[t]           physical charging and discharging (MW); the difference d - c - q is traded intraday
                       at the intraday price (ID-AEP)
  soc[t]               state of charge at the end of hour t (MWh); u[t] binary forbids charging and
                       discharging in the same hour

Constraints: power headroom d + r_pos <= P and c + r_neg <= P in every hour of the block; energy headroom
soc >= tau * r_pos and soc <= E - tau * r_neg, so a reserved MW could be delivered for tau hours; the day
starts and ends half full; |q| <= P. Objective: day-ahead revenue + intraday revenue + capacity payments
- degradation cost per MWh discharged.

Simplifications (stated in the README): price-taker; capacity paid at the German average accepted price;
aFRR activation energy and its effect on the state of charge are not modelled; hourly resolution
(the quarter-hour products since October 2025 are averaged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

T, B = 24, 6
BLOCK = np.repeat(np.arange(B), 4)          # block of each hour


@dataclass(frozen=True)
class Battery:
    power: float = 1.0          # MW
    energy: float = 2.0         # MWh
    rte: float = 0.88           # round-trip efficiency
    deg_cost: float = 10.0      # EUR per MWh discharged
    tau: float = 1.0            # hours a reserved MW must be deliverable from the state of charge
    soc0_frac: float = 0.5

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.rte))


# variable layout: physical schedule, day-ahead position, reserves, day-ahead plan
IC, ID_, IS, IQ, IU = (np.arange(T) + k * T for k in range(5))
IRP, IRN = 5 * T + np.arange(B), 5 * T + B + np.arange(B)
IC0, ID0, IS0, IU0 = (5 * T + 2 * B + np.arange(T) + k * T for k in range(4))
NV = 9 * T + 2 * B


def solve_day(bat: Battery, da: np.ndarray, idp: np.ndarray, cap_pos: np.ndarray, cap_neg: np.ndarray,
              fix_r: tuple[np.ndarray, np.ndarray] | None = None, fix_q: np.ndarray | None = None,
              markets: frozenset = frozenset({"da", "id", "afrr"})) -> dict:
    """Optimal decisions for one day under the given prices; markets not listed are switched off."""
    P, E, eta = bat.power, bat.energy, bat.eta
    s0 = bat.soc0_frac * E
    lb, ub = np.zeros(NV), np.full(NV, np.inf)
    ub[IC] = ub[ID_] = P
    ub[IS] = E
    lb[IQ], ub[IQ] = -P, P
    ub[IU] = 1
    ub[IRP] = ub[IRN] = P
    ub[IC0] = ub[ID0] = P
    ub[IS0] = E
    ub[IU0] = 1
    if "afrr" not in markets:
        ub[IRP] = ub[IRN] = 0
    if fix_r is not None:
        lb[IRP] = ub[IRP] = fix_r[0]
        lb[IRN] = ub[IRN] = fix_r[1]
    if fix_q is not None:
        lb[IQ] = ub[IQ] = fix_q
    elif "da" not in markets:
        lb[IQ] = ub[IQ] = 0
    if not ({"da", "id"} & markets):
        ub[IC] = ub[ID_] = 0

    rows, lo, hi = [], [], []

    def add(coef: dict, l: float, h: float) -> None:
        row = np.zeros(NV)
        for i, v in coef.items():
            row[i] += v
        rows.append(row), lo.append(l), hi.append(h)

    for t in range(T):
        b = BLOCK[t]
        for ic, idd, isc, iu in ((IC, ID_, IS, IU), (IC0, ID0, IS0, IU0)):   # physical schedule and day-ahead plan
            add({ic[t]: 1, iu[t]: -P}, -np.inf, 0)                    # c <= P u
            add({idd[t]: 1, iu[t]: P}, -np.inf, P)                    # d <= P (1 - u)
            add({idd[t]: 1, IRP[b]: 1}, -np.inf, P)                   # power headroom up
            add({ic[t]: 1, IRN[b]: 1}, -np.inf, P)                    # power headroom down
            prev = {isc[t - 1]: -1} if t else {}
            add({isc[t]: 1, **prev, ic[t]: -eta, idd[t]: 1 / eta}, s0 if t == 0 else 0, s0 if t == 0 else 0)
            add({isc[t]: 1, IRP[b]: -bat.tau}, 0, np.inf)             # energy headroom up
            add({isc[t]: 1, IRN[b]: bat.tau}, -np.inf, E)             # energy headroom down
        add({IQ[t]: 1, ID0[t]: -1, IC0[t]: 1}, 0, 0)                  # q is the day-ahead plan
        if "id" not in markets:
            add({ID_[t]: 1, IC[t]: -1, IQ[t]: -1}, 0, 0)          # no intraday trade: physical = day-ahead
    for b in range(B):                                            # block start: the initial or previous state
        if b == 0:
            add({IRP[0]: bat.tau}, -np.inf, s0)
            add({IRN[0]: bat.tau}, -np.inf, E - s0)
    add({IS[T - 1]: 1}, s0, s0)                                   # end half full
    add({IS0[T - 1]: 1}, s0, s0)

    # maximise revenue -> minimise its negative
    c = np.zeros(NV)
    c[IQ] = -(da - idp)              # q at DA, minus q at ID (the ID leg trades d - c - q)
    c[ID_] = -(idp - bat.deg_cost)
    c[IC] = idp
    c[IRP] = -4 * cap_pos
    c[IRN] = -4 * cap_neg
    integrality = np.zeros(NV)
    integrality[IU] = 1
    integrality[IU0] = 1
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality,
               bounds=Bounds(lb, ub), options={"time_limit": 30})
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    return {"c": x[IC], "d": x[ID_], "soc": x[IS], "q": x[IQ], "r_pos": x[IRP], "r_neg": x[IRN]}


def revenue(bat: Battery, sol: dict, da: np.ndarray, idp: np.ndarray, cap_pos: np.ndarray, cap_neg: np.ndarray) -> dict:
    """Realised revenue of a decision at the given (actual) prices, split by market (EUR per day)."""
    q, c, d = sol["q"], sol["c"], sol["d"]
    out = {"da": float(da @ q), "id": float(idp @ (d - c - q)),
           "afrr": float(4 * (cap_pos @ sol["r_pos"] + cap_neg @ sol["r_neg"])),
           "degradation": float(-bat.deg_cost * d.sum())}
    out["total"] = sum(out.values())
    out["cycles"] = float(d.sum() / bat.energy)
    return out
