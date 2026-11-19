"""Forecast GTA VI PS5 trophy earn rates on Exophase at the case target.

Approach (earners / players decomposition).  A trophy's rate is
E / P: earners over players tracked.  Earners only grow; players
tracked grows as Exophase discovers more players, so a rate can FALL
when tracking outgrows earning (dilution -- this is real and was
observed on launch night, when a refresh took players 638 -> 3868
and almost every rate down).  So per trophy we fit, on the last 24h
of reads (or all reads when fewer), with exponential recency weights
(half-life 12h):

  (a) log earners vs log(1 + hours since release),
  (b) log players vs log(1 + hours since release),
  (c) logit(rate) vs log(1 + hours since release) directly.

The forecast rate is the average in logit space of (a)/(b) and (c).
Fits run on distinct plateaus, not on every read: between refreshes
the page is frozen, so repeats carry no information and a long flat
stretch must not drag slopes to zero.  A trophy every tracked player
holds (100% at the checkpoint) keeps a ~100% median with a long
downside tail, since it survived a 6x player expansion at 100%.
Spreads are honest about refresh shocks: a floor of 0.5 logit units
(the launch-night refresh moved rates by 0.6-1.8 logit in 90 min),
plus a random-walk term in the horizon and the fit residual, with a
longer downside tail than upside tail because refreshes drop rates.

Structural priors enforced on medians: chapter trophies decrease
along the story (Ch1 >= Ch2 >= Ch3 >= Ch4 >= final mission), the
platinum is at most the rarest trophy and at most the 100%-completion
trophy (platinum requires 100%), 100% completion is at most the
final mission.  Ceilings cap extreme extrapolation only.

Standard library only, deterministic, no network, no wall clock.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Story / structural ids (Exophase list order).
CHAPTERS = (3, 4, 5, 6, 7)          # Ch1, Ch2, Ch3, Ch4, final mission
PLATINUM = 1
FULL_COMPLETION = 14
GRIND = frozenset((15, 30, 31, 32))  # 250 stunts, 3 card sets, 100 spots, 50 challenges

CEILING = {PLATINUM: 10.0, FULL_COMPLETION: 12.0}
GRIND_CEILING = 25.0
MAX_RATE = 99.5

SIGMA_FLOOR = 0.5        # logit units: refresh / dilution uncertainty
RW_PER_DAY = 0.5         # random-walk spread per sqrt(day)
DOWN_MULT = 1.3          # downside tail longer: refreshes drop rates
UP_MULT = 0.9
HALF_LIFE_H = 12.0       # recency weight half-life for the fits
WINDOW_H = 24.0          # fit window: last 24h of reads


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    return float(Decimal(repr(x)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP))


def wls(xs, ys, ws):
    """Weighted least squares line; returns (slope, intercept, res_var)."""
    sw = sum(ws)
    if sw <= 0.0 or len(xs) < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-12:
        var = sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / sw
        return 0.0, my, var
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    var = sum(w * (y - (inter + slope * x)) ** 2
              for w, x, y in zip(ws, xs, ys)) / sw
    return slope, inter, var


def forecast_trophy(series, horizon_days, target_x):
    """series: list of (hours, players, rate_pct) with rate not None.

    Returns (median_logit, sigma, down_mult, up_mult).
    """
    if not series:
        return logit(0.5), 1.5, DOWN_MULT, UP_MULT
    last_rate = series[-1][2]
    if last_rate >= 99.9:
        # Essentially every tracked player holds it; new players can
        # only dilute it, so a high median with a long downside tail.
        return logit(99.9), 1.0, DOWN_MULT, UP_MULT
    # Collapse consecutive identical reads to one point per plateau:
    # between Exophase refreshes the page is frozen, so repeats carry
    # no new information, and a long flat stretch must not drag the
    # fitted slope to zero or let one refresh jump count a hundred
    # times.  Each plateau is dated at its first read (the refresh).
    plateaus = []
    for point in series:
        if plateaus and plateaus[-1][1] == point[1] and \
                plateaus[-1][2] == point[2]:
            continue
        plateaus.append(point)
    series = plateaus
    t_last = series[-1][0]
    window = [s for s in series if s[0] >= t_last - WINDOW_H]
    if len(window) < 3:
        window = series
    xs, wts = [], []
    y_e, y_p, y_l = [], [], []
    for hours, players, rate in window:
        x = math.log(max(1.0 + hours, 0.05))
        xs.append(x)
        wts.append(0.5 ** ((t_last - hours) / HALF_LIFE_H))
        earners = max(rate / 100.0 * max(players, 1), 0.5)
        y_e.append(math.log(earners))
        y_p.append(math.log(max(players, 1)))
        y_l.append(logit(rate))
    if len(xs) < 2:
        return logit(last_rate), 1.0, DOWN_MULT, UP_MULT
    s_e, i_e, v_e = wls(xs, y_e, wts)
    s_p, i_p, v_p = wls(xs, y_p, wts)
    s_l, i_l, v_l = wls(xs, y_l, wts)
    s_e = clamp(s_e, 0.0, 2.5)   # earners never shrink
    s_p = clamp(s_p, 0.0, 2.5)   # tracked players never shrink
    s_l = clamp(s_l, -2.0, 3.0)
    e_tgt = math.exp(i_e + s_e * target_x)
    p_tgt = math.exp(i_p + s_p * target_x)
    rate_ab = clamp(100.0 * e_tgt / max(p_tgt, 1.0), 0.01, MAX_RATE)
    rate_c = clamp(inv_logit(i_l + s_l * target_x), 0.01, MAX_RATE)
    med = 0.5 * (logit(rate_ab) + logit(rate_c))
    # Guard against runaway extrapolation from noisy early fits.
    med = clamp(med, logit(last_rate) - 3.5, logit(last_rate) + 3.5)
    resid = min(v_l, 1.0) + 0.25 * min(v_e + v_p, 1.0)
    sigma = math.sqrt(SIGMA_FLOOR ** 2 + (RW_PER_DAY ** 2) * horizon_days
                      + resid)
    # Dilution risk: players growing faster than earners -> longer tail down.
    dil = clamp((s_p - s_e) / 1.5, 0.0, 1.0)
    down = DOWN_MULT + 0.25 * dil
    up = UP_MULT - 0.1 * dil
    return med, sigma, down, up


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target_h = hours_since_release(parse_time(case["target"]))
    target_x = math.log(max(1.0 + target_h, 0.05))
    n = len(trophies)

    # Per-trophy series: (hours, players, rate) in time order.
    med = {}
    spread = {}
    for idx, trophy in enumerate(trophies):
        series = []
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            series.append((hours_since_release(parse_time(read["at"])),
                           read.get("players_tracked") or 1,
                           float(rates[idx])))
        if not series:
            # No read ever listed this trophy: fall back to a rare rate.
            med[trophy["id"]] = logit(0.5)
            spread[trophy["id"]] = (1.5, DOWN_MULT, UP_MULT)
            continue
        m, s, d, u = forecast_trophy(series, horizon_days, target_x)
        med[trophy["id"]] = m
        spread[trophy["id"]] = (s, d, u)

    # Structural priors on medians (percent scale, via logit).
    def pct(tid):
        return inv_logit(med[tid])

    def set_pct(tid, value):
        med[tid] = logit(clamp(value, 0.01, MAX_RATE))

    # Chapters decrease along the story.
    prev = None
    for tid in CHAPTERS:
        if tid in med:
            cur = pct(tid)
            if prev is not None and cur > prev:
                set_pct(tid, prev)
            prev = pct(tid)
    # 100% completion needs the final mission done.
    if FULL_COMPLETION in med and 7 in med:
        if pct(FULL_COMPLETION) > pct(7):
            set_pct(FULL_COMPLETION, pct(7))
    # Platinum is the rarest: <= every other trophy, and <= 100%.
    if PLATINUM in med and len(med) > 1:
        others = [pct(t) for t in med if t != PLATINUM]
        cap = min(others)
        if FULL_COMPLETION in med:
            cap = min(cap, pct(FULL_COMPLETION))
        if pct(PLATINUM) > cap:
            set_pct(PLATINUM, cap)

    # Last observed rate per trophy (for the near-100% special case).
    last_rate_of = {}
    if history:
        last_rates = history[-1].get("rates") or []
        for idx, trophy in enumerate(trophies):
            if idx < len(last_rates) and last_rates[idx] is not None:
                last_rate_of[trophy["id"]] = float(last_rates[idx])

    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        m = med[tid]
        sigma, down, up = spread[tid]
        qs = []
        for z in Z:
            mult = down if z < 0 else up if z > 0 else 1.0
            qs.append(inv_logit(m + z * sigma * mult))
        # A trophy every tracked player holds: the rate can only be
        # diluted by newcomers, and it survived a 6x player expansion
        # at 100%, so the median stays ~100 with a long downside tail.
        if last_rate_of.get(tid, 0.0) >= 99.9:
            qs = [93.0, 99.5, 99.99, 99.99, 99.99]
            rows.append({"id": tid, "q": qs})
            continue
        # Ceilings (caps only, never targets).
        ceil = CEILING.get(tid, GRIND_CEILING if tid in GRIND else MAX_RATE)
        qs = [clamp(q, 0.01, 99.99) for q in qs]
        qs = [min(q, ceil) for q in qs]
        # Round half-up to 4 decimals, then force non-decreasing.
        qs = [round4(q) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
