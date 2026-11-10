#!/usr/bin/env python3
"""
Driving Insights - four-source weighted Drive Score: reference implementation,
10 km simulated trip (ten 1 km buckets) and validation tests.

Pure stdlib. The scoring part (section 2) uses only arithmetic on a bucket record,
so it can be ported 1:1 to Kotlin; testvectors.json is the contract.

Sources: ESP (accelerometer), BMS (consumption), Temperature, Terrain.
Every source keeps a weight > 0 (confidence floor), inactive conditions give a neutral sub-score.
"""
import json, math, random, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ 1. model (shared by simulator and scorer)
M, G, RHO, CDA = 1950.0, 9.81, 1.2, 0.62
ETA_T = 0.88

def clamp(x, lo, hi): return max(lo, min(hi, x))
def lerp(t, a, b): return a + (b - a) * t

COMFORT_C, COMFORT_BAND = 20.0, 3.0     # 20 C is the zero point: no heating or cooling is needed within +-3 K

def climate_k(T):        # climate demand in kelvin: distance from the comfort zone, 0 inside 17..23 C
    return max(0.0, abs(T - COMFORT_C) - COMFORT_BAND)

def p_aux(T):            # kW, base load + heating (below 20 C) or cooling (above 20 C) power
    return 0.6 + (0.060 if T < COMFORT_C else 0.050) * climate_k(T)

def f_T(T):              # cold-driveline factor on traction energy
    return lerp(clamp((10 - T) / 20.0, 0, 1), 1.0, 1.10)

def eta_r(T):            # regen round-trip efficiency vs battery temperature
    return lerp(clamp((10 - T) / 20.0, 0, 1), 0.75, 0.45)

def e_warm(T):           # kWh once per trip start: cabin heat-up (cold) or pull-down (hot), battery conditioning
    return (0.052 if T < COMFORT_C else 0.022) * climate_k(T)

def crr(T):
    return lerp(clamp((10 - T) / 20.0, 0, 1), 0.009, 0.0105)

# ------------------------------------------------------------------ 2. scoring (port this to Kotlin)
BANDS = [(0.10, 1.0), (0.20, 0.8), (0.30, 0.45), (9.9, 0.0)]   # peak a_c upper edge, weight
# Context factor: the same peak force is riskier in the cold (less grip) and on a slope, so the band edges shrink.
CTX_COLD_REF_C, CTX_COLD_PER_K, CTX_GRADE_PER_PCT, CTX_MIN = 5.0, 0.010, 0.020, 0.70
def ctx_factor(T, grade_pct):
    """T in C, grade_pct = (climb + descent) per km / 10, i.e. average absolute grade of the km in percent."""
    return clamp(1.0 - CTX_COLD_PER_K * max(0.0, CTX_COLD_REF_C - T) - CTX_GRADE_PER_PCT * grade_pct, CTX_MIN, 1.0)
def band_weight(ac, f=1.0):
    for edge, w in BANDS:
        if ac < edge * f: return w
    return 0.0

BASE_W = {"ESP": 60.0, "BMS": 20.0, "TMP": 10.0, "TER": 10.0}
FLOOR = 0.2
E_BEST, E_WORST = 10.0, 24.0         # kWh/100km normalised: 100 pts / 0 pts
E_NORM_MIN = 3.0                      # floor of the normalised consumption (inconsistent inputs cannot give a negative value)
TMP_FULL_K, TMP_FULL_H = 3.0, 0.5     # climate demand (K) and steady driving time (h) at which the temperature sub-score counts fully
CLIMATE_FULL_K = 10.0                 # climate demand at which the temperature input counts fully
TER_ACTIVE_DOWN_M = 30.0
WARM_ZONE_KM = 3                      # km at the start of every trip excluded from the climate-use ratio
CAPTURE_REF = 0.8

class Bucket:
    """One km of driving; the only data stored (no timestamp, no position)."""
    FIELDS = ["dist_km", "dur_s", "temp_c", "e_meas_kwh", "e_aux_kwh", "h_up_m", "h_down_m",
              "e_regen_down_kwh", "dv_total", "dv_weighted", "n_trips", "e_aux_run_kwh", "dur_run_s"]
    def __init__(self, **kw):
        for f in self.FIELDS: setattr(self, f, kw.get(f))
    def dict(self): return {f: getattr(self, f) for f in self.FIELDS}

def aggregate(bs):
    """Window = plain sums; additive, so any distance window is a ring-buffer sum."""
    a = {f: 0.0 for f in Bucket.FIELDS if f != "temp_c"}
    tsum = 0.0; tw = 0.0
    for b in bs:
        for f in a: a[f] += (getattr(b, f) or 0.0)
        if b.temp_c is not None:
            tsum += b.temp_c * b.dist_km; tw += b.dist_km
    a["temp_c"] = (tsum / tw) if tw > 0 else None
    return a

def score(a, have=("ESP", "BMS", "TMP", "TER"), params=None):
    """a = aggregated window. Returns dict with sub-scores, weights, drive score."""
    p = dict(m=M, aux=1.0, eta_r=1.0)
    if params: p.update(params)
    d = a["dist_km"]; T = a["temp_c"]
    out = {}
    # --- ESP
    s_esp = 100.0 * a["dv_weighted"] / a["dv_total"] if a["dv_total"] > 0 else None
    # --- BMS normalised consumption
    s_bms = e_norm = e_raw = None
    if "BMS" in have and a["e_meas_kwh"] > 0 and T is not None:
        t_h = a["dur_s"] / 3600.0
        # Reference, not the driver's own values: the climate energy a typical driver needs (heating or cooling
        # above the 20 C base load, plus the warm-up of every trip start) and the descent recovery a good driver
        # reaches (CAPTURE_REF). The driver's extra climate use and missed recovery therefore stay in e_norm.
        e_aux_ref = p["aux"] * (p_aux(T) - p_aux(20.0)) * t_h + p["aux"] * e_warm(T) * (a["n_trips"] or 0.0)
        avail_b = p["m"] * G * ETA_T * p["eta_r"] * eta_r(T) * a["h_down_m"] / 3.6e6
        e_rec = CAPTURE_REF * avail_b
        e_grade = p["m"] * G * a["h_up_m"] / ETA_T / 3.6e6 - e_rec
        e_norm = max(E_NORM_MIN, (a["e_meas_kwh"] - e_aux_ref - e_grade) / d * 100.0 / f_T(T))
        s_bms = 100.0 * clamp((E_WORST - e_norm) / (E_WORST - E_BEST), 0, 1)
    if a["e_meas_kwh"] > 0:
        e_raw = a["e_meas_kwh"] / d * 100.0
    # --- Temperature: driver's climate-use factor, measured on steady driving only.
    # The first WARM_ZONE_KM of every trip (cabin and battery warm-up) are left out of this ratio.
    t_active = False; s_tmp = None; t_run = (a["dur_run_s"] or 0.0) / 3600.0; b_t = 0.0
    if "TMP" in have and T is not None:
        t_active = climate_k(T) > 0 and a["e_aux_kwh"] > 0 and t_run > 0
        if t_active:
            kappa = a["e_aux_run_kwh"] / (p["aux"] * p_aux(T) * t_run)
            s_tmp = clamp(75.0 - 250.0 * (kappa - 1.0), 0, 100)
            b_t = clamp(climate_k(T) / TMP_FULL_K, 0, 1) * clamp(t_run / TMP_FULL_H, 0, 1)     # no jump at the edge of the comfort zone
    # --- Terrain: regen capture on descents, blended in over the first TER_ACTIVE_DOWN_M metres of descent
    r_active = False; s_ter = None; b_r = 0.0
    if "TER" in have and T is not None and a["h_down_m"] > 0:
        avail = p["m"] * G * ETA_T * p["eta_r"] * eta_r(T) * a["h_down_m"] / 3.6e6
        if avail > 0 and a["e_regen_down_kwh"] is not None:
            r_active = True
            s_ter = clamp(100.0 * (a["e_regen_down_kwh"] / avail) / CAPTURE_REF, 0, 100)
            b_r = clamp(a["h_down_m"] / TER_ACTIVE_DOWN_M, 0, 1)
    # --- confidence and weights (all > 0)
    c = {"ESP": 1.0 if s_esp is not None else FLOOR,
         "BMS": clamp(d / 30.0, FLOOR, 1.0) if s_bms is not None else FLOOR,
         # climate-use score needs enough steady driving time (warm-up dominates short trips): full at 30 min
         "TMP": clamp(climate_k(T) / CLIMATE_FULL_K * min(1.0, t_run / 0.5), FLOOR, 1.0) if t_active else FLOOR,
         "TER": (max(FLOOR, 0.8 * b_r) if r_active else FLOOR)}
    raw_w = {k: BASE_W[k] * c[k] for k in BASE_W}
    tot = sum(raw_w.values())
    w = {k: raw_w[k] / tot for k in raw_w}
    # Neutral value of an inactive or still-ramping input = the weighted mean of the ESP and BMS scores, so an
    # input with nothing to judge neither rewards nor punishes (a perfect driver can reach 100).
    base = [(BASE_W[k] * c[k], v) for k, v in (("ESP", s_esp), ("BMS", s_bms)) if v is not None]
    neutral = sum(x * v for x, v in base) / sum(x for x, _ in base) if base else 70.0
    S = {"ESP": s_esp if s_esp is not None else neutral,
         "BMS": s_bms if s_bms is not None else neutral,
         "TMP": neutral if s_tmp is None else neutral + b_t * (s_tmp - neutral),
         "TER": neutral if s_ter is None else neutral + b_r * (s_ter - neutral)}
    ds = sum(w[k] * S[k] for k in w)
    raw_score = None
    if s_esp is not None and e_raw is not None:      # v1.0-style: no normalisation
        i_raw = 100.0 * clamp((E_WORST - e_raw) / (E_WORST - E_BEST), 0, 1)
        raw_score = 0.8 * s_esp + 0.2 * i_raw
    return dict(S=S, w=w, c=c, drive_score=ds, e_norm=e_norm, e_raw=e_raw,
                raw_score=raw_score,
                raw_ie=(100.0 * clamp((E_WORST - e_raw) / (E_WORST - E_BEST), 0, 1) if e_raw else None))

def rating(ds):
    ds = int(ds + 0.5)               # the rating follows the displayed, rounded number
    return "Excellent" if ds >= 90 else "Good" if ds >= 80 else "Balanced" if ds >= 60 else "Needs Improvement"

# ------------------------------------------------------------------ 3. simulator (stand-in for ESP + BMS + HVAC + slope)
PROFILES = {
    #        speed  events/km  band probabilities (eco,normal,sporty,aggr)  regen_capture  climate kappa  accel peak boost
    "Eco":        dict(v=45, n=3, p=(0.75, 0.25, 0.00, 0.00), cap=0.80, kappa=0.85),
    "Normal":     dict(v=50, n=4, p=(0.20, 0.65, 0.15, 0.00), cap=0.55, kappa=1.00),
    "Aggressive": dict(v=62, n=5, p=(0.05, 0.25, 0.35, 0.35), cap=0.30, kappa=1.05),
}
CONDITIONS = {
    "mild-flat":  dict(T=20.0, up=[0]*10, down=[0]*10),
    "mild-hilly": dict(T=20.0, up=[22,8,25,5,20,6,24,4,18,7], down=[4,20,6,24,5,22,3,25,6,20]),
    "cold-flat":  dict(T=-5.0, up=[0]*10, down=[0]*10),
    "cold-hilly": dict(T=-5.0, up=[22,8,25,5,20,6,24,4,18,7], down=[4,20,6,24,5,22,3,25,6,20]),
    "hot-flat":   dict(T=35.0, up=[0]*10, down=[0]*10),
}
EDGES = [(0.04, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.45)]

WARM_SHARE = (0.60, 0.25, 0.15)      # warm-up energy decays over the first three km of a trip

def add_warmup(bs, T):
    """Cold start: warm-up energy (cabin + battery) lands in the first km of the trip, once."""
    ew = e_warm(T)
    for i in range(WARM_ZONE_KM):
        bs[i].e_meas_kwh += ew * WARM_SHARE[i]; bs[i].e_aux_kwh += ew * WARM_SHARE[i]
    bs[0].n_trips = 1.0

def simulate(profile, cond, seed=7, warm=True):
    pr, co = PROFILES[profile], CONDITIONS[cond]
    # maneuvers: own RNG, independent of conditions -> same driver, same peak forces; the band edges depend on context
    rm = random.Random(f"{seed}-{profile}-man")
    rn = random.Random(f"{seed}-{profile}-{cond}-noise")
    T = co["T"]
    buckets = []
    for k in range(10):
        v = pr["v"] * (1 + rm.uniform(-0.08, 0.08)) / 3.6       # m/s
        f_ctx = ctx_factor(T, (co["up"][k] + co["down"][k]) / 10.0)
        dv_tot = dv_w = 0.0
        e_kin_acc = e_kin_brk = 0.0
        for ev in range(pr["n"]):
            for kind in ("acc", "brk"):
                bi = rm.choices(range(4), pr["p"])[0]
                lo, hi = EDGES[bi]; ac = rm.uniform(lo, hi)
                dv = rm.uniform(4, 11) * (1.0 + 0.3 * bi) / 3.6   # m/s, harder manoeuvres change more speed
                dv = min(dv, v * 0.9)
                dv_tot += dv; dv_w += dv * band_weight(ac, f_ctx)
                ke = 0.5 * M * (v ** 2 - (v - dv) ** 2)
                if kind == "acc":
                    e_kin_acc += ke / (ETA_T - 0.10 * ac / 0.3)   # high-power inefficiency
                else:
                    e_kin_brk += ke
        d_m = 1000.0
        # kinetic energy given back by braking: part is recovered by regen, driver-dependent
        e_regen_brk = e_kin_brk * pr["cap"] * eta_r(T) * ETA_T
        e_rr = crr(T) * M * G * d_m / ETA_T
        e_aero = 0.5 * RHO * CDA * v ** 2 * d_m / ETA_T
        up, down = co["up"][k], co["down"][k]
        e_up = M * G * up / ETA_T
        e_down_avail = M * G * down * ETA_T * eta_r(T)
        e_regen_down = e_down_avail * pr["cap"] / 0.8 * 0.8 if False else e_down_avail * pr["cap"]
        # drive-time: cruise + a little stop time
        dur = d_m / v + 4.0 * pr["n"]
        e_aux = pr["kappa"] * p_aux(T) * dur / 3600.0
        e_trac = (e_rr + e_aero) / 3.6e6 + e_kin_acc / 3.6e6 - e_regen_brk / 3.6e6 \
                 + e_up / 3.6e6 - e_regen_down / 3.6e6
        e_trac *= f_T(T) if True else 1.0
        e_meas = (e_trac + e_aux) * (1 + rn.gauss(0, 0.01))              # BMS noise 1 %
        n8 = lambda x: x * (1 + rn.gauss(0, 0.08))                       # ESP-estimated terrain noise 8 %
        buckets.append(Bucket(dist_km=1.0, dur_s=dur, temp_c=T + rn.gauss(0, 0.3),
                              e_meas_kwh=e_meas, e_aux_kwh=e_aux,
                              h_up_m=n8(up) if up else 0.0, h_down_m=n8(down) if down else 0.0,
                              e_regen_down_kwh=e_regen_down / 3.6e6 * (1 + rn.gauss(0, 0.01)),
                              dv_total=dv_tot, dv_weighted=dv_w))
    for b in buckets: b.n_trips = 0.0
    if warm: add_warmup(buckets, T)              # one cold start at the beginning of the trip
    finish(buckets, warm)
    return buckets

def finish(bs, starts_trip):
    """Mark the buckets in the warm-up zone and fill the steady-state aux fields."""
    for i, b in enumerate(bs):
        in_zone = starts_trip and i < WARM_ZONE_KM
        b.e_aux_run_kwh = 0.0 if in_zone else b.e_aux_kwh
        b.dur_run_s = 0.0 if in_zone else b.dur_s

def window(profile, cond, total_km, trip_km, seed=7):
    """total_km of driving made of trips of trip_km each (10 km pattern repeated); every trip pays its own warm-up."""
    import copy
    base = simulate(profile, cond, seed, warm=False); T = CONDITIONS[cond]["T"]; out = []
    for i in range(total_km // 10):
        bs = copy.deepcopy(base); start = (i * 10) % trip_km == 0
        if start: add_warmup(bs, T)
        finish(bs, start); out += bs
    return out

# ------------------------------------------------------------------ 4. validation
RESULTS = {"tests": [], "table": [], "vectors": []}
def check(name, ok, detail=""):
    RESULTS["tests"].append(dict(name=name, passed=bool(ok), detail=detail))
    print(("PASS " if ok else "FAIL ") + name + ("  | " + detail if detail else ""))

def run():
    grid = {}
    for pf in PROFILES:
        for cd in CONDITIONS:
            bs = simulate(pf, cd); a = aggregate(bs); r = score(a)
            grid[(pf, cd)] = (bs, a, r)
            RESULTS["table"].append(dict(profile=pf, condition=cd, dist_km=a["dist_km"],
                S_ESP=round(r["S"]["ESP"], 1), S_BMS=round(r["S"]["BMS"], 1), S_TMP=round(r["S"]["TMP"], 1),
                S_TER=round(r["S"]["TER"], 1), e_raw=round(r["e_raw"], 2), e_norm=round(r["e_norm"], 2),
                weights={k: round(v, 3) for k, v in r["w"].items()},
                drive_score=round(r["drive_score"], 1), rating=rating(r["drive_score"]),
                raw_score=round(r["raw_score"], 1)))

    # T1 all weights > 0
    minw = min(min(r["w"].values()) for _, _, r in grid.values())
    check("T1 every weight > 0 in all 15 trips", minw > 0, f"min weight {minw:.3f}")
    # T2 weights sum to 1
    check("T2 weights sum to 1", all(abs(sum(r["w"].values()) - 1) < 1e-9 for _, _, r in grid.values()))
    # T3 S_ESP independent of conditions
    ok = all(abs(grid[(pf, "mild-flat")][2]["S"]["ESP"] - grid[(pf, "hot-flat")][2]["S"]["ESP"]) < 1e-9 for pf in PROFILES)
    check("T3 S_ESP identical for the same driver when neither cold nor slope is present (20 C versus 35 C, flat)", ok)
    ok = all(grid[(pf, "cold-hilly")][2]["S"]["ESP"] < grid[(pf, "cold-flat")][2]["S"]["ESP"] < grid[(pf, "mild-flat")][2]["S"]["ESP"]
             and grid[(pf, "cold-hilly")][2]["S"]["ESP"] < grid[(pf, "mild-hilly")][2]["S"]["ESP"] < grid[(pf, "mild-flat")][2]["S"]["ESP"]
             for pf in PROFILES)
    check("T3b S_ESP is context-aware: the same manoeuvres score lower in the cold and on slopes (cold-hilly lowest)", ok)
    # T4 ordering Eco > Normal > Aggressive in every condition
    ok = all(grid[("Eco", cd)][2]["drive_score"] > grid[("Normal", cd)][2]["drive_score"] > grid[("Aggressive", cd)][2]["drive_score"]
             for cd in CONDITIONS)
    check("T4 Drive Score ordering Eco > Normal > Aggressive in every condition", ok)
    # T5 condition spread: normalised vs raw (BMS part and total)
    for pf in PROFILES:
        e_raw = [grid[(pf, cd)][2]["e_raw"] for cd in CONDITIONS]
        e_nrm = [grid[(pf, cd)][2]["e_norm"] for cd in CONDITIONS]
        ds = [grid[(pf, cd)][2]["drive_score"] for cd in CONDITIONS]
        rs = [grid[(pf, cd)][2]["raw_score"] for cd in CONDITIONS]
        sp = lambda x: max(x) - min(x)
        check(f"T5 {pf}: consumption spread across the 4 conditions shrinks after normalisation (kWh/100 km)",
              sp(e_nrm) < 0.5 * sp(e_raw), f"raw {sp(e_raw):.1f} -> normalised {sp(e_nrm):.1f}")
        check(f"T5 {pf}: Drive Score spread across conditions <= 12 pts (same driver, bounded context effect)",
              sp(ds) <= 12.0, f"v1.0-style raw score {sp(rs):.1f} pts -> Drive Score {sp(ds):.1f} pts")
    # T6 additivity: merging two 5 km halves equals the 10 km window
    bs = grid[("Normal", "cold-hilly")][0]
    a_all = aggregate(bs); a_split = aggregate(bs[:5]); a_b = aggregate(bs[5:])
    summed = {f: a_split[f] + a_b[f] for f in Bucket.FIELDS if f != "temp_c"}
    check("T6 window additivity (sum of 2x5 km buckets == 10 km window)",
          all(abs(summed[f] - a_all[f]) < 1e-9 for f in summed))
    # T7 robustness to model mismatch +-20 %
    worst = 0.0; order_ok = True
    for dm in (0.8, 1.2):
        for da in (0.8, 1.2):
            for de in (0.8, 1.2):
                prm = dict(m=M * dm, aux=da, eta_r=de)
                for cd in CONDITIONS:
                    sc = {pf: score(grid[(pf, cd)][1], params=prm)["drive_score"] for pf in PROFILES}
                    base = {pf: grid[(pf, cd)][2]["drive_score"] for pf in PROFILES}
                    worst = max(worst, max(abs(sc[pf] - base[pf]) for pf in PROFILES))
                    order_ok &= sc["Eco"] > sc["Normal"] > sc["Aggressive"]
    check("T7 +-20 % model error (mass, aux, regen eff.): ordering kept", order_ok)
    check("T7b +-20 % model error: Drive Score moves <= 10 pts (worst corner of 8 combinations)", worst <= 10.0, f"max shift {worst:.1f} pts")
    # T8 missing data fallbacks
    a = grid[("Normal", "cold-hilly")][1]
    fb_ok = True; det = []
    for name, have in (("no BMS", ("ESP", "TMP", "TER")), ("no temperature", ("ESP", "BMS", "TER")),
                       ("no terrain", ("ESP", "BMS", "TMP")), ("ESP only", ("ESP",))):
        r = score(a, have=have)
        ok = min(r["w"].values()) > 0 and math.isfinite(r["drive_score"])
        fb_ok &= ok; det.append(f"{name}: {r['drive_score']:.1f}")
    check("T8 missing-source fallbacks keep all weights > 0 and a finite score", fb_ok, "; ".join(det))
    # T9 inactive conditions are neutral: mild-flat -> temperature + terrain inactive
    r = grid[("Normal", "mild-flat")][2]
    nb = (60 * r["c"]["ESP"] * r["S"]["ESP"] + 20 * r["c"]["BMS"] * r["S"]["BMS"]) / (60 * r["c"]["ESP"] + 20 * r["c"]["BMS"])
    check("T9 inactive inputs take the mean of the ESP and BMS scores and the floor weight, so the total equals that mean",
          abs(r["S"]["TMP"] - nb) < 1e-9 and abs(r["S"]["TER"] - nb) < 1e-9
          and r["c"]["TMP"] == FLOOR and r["c"]["TER"] == FLOOR and abs(r["drive_score"] - nb) < 1e-9,
          f"S_TMP {r['S']['TMP']:.1f}, S_TER {r['S']['TER']:.1f}, total {r['drive_score']:.1f}")
    # T10 determinism
    a1 = score(aggregate(simulate("Eco", "cold-hilly")))["drive_score"]
    a2 = score(aggregate(simulate("Eco", "cold-hilly")))["drive_score"]
    check("T10 deterministic (same seed -> same score)", a1 == a2)
    # T11 sanity of e_norm
    mf = grid[("Normal", "mild-flat")][2]["e_norm"]
    check("T11 normalised consumption plausible (Normal, mild flat 9-20 kWh/100 km)", 9 <= mf <= 20, f"{mf:.2f}")
    # T12-T14 warm-up: a short trip pays for heating the cabin and the battery, a long trip pays once
    raw_ok = norm_ok = same_ok = True; maxd10 = 0.0; maxsame = 0.0; maxnorm = 0.0; minratio = 9.0
    for pf in PROFILES:
        for cd in ("cold-flat", "cold-hilly"):
            r10 = grid[(pf, cd)][2]
            r200 = score(aggregate(window(pf, cd, 200, 200)))
            r20 = score(aggregate(window(pf, cd, 200, 10)))
            minratio = min(minratio, r10["e_raw"] / r200["e_raw"])
            maxnorm = max(maxnorm, abs(r10["e_norm"] - r200["e_norm"]))
            maxsame = max(maxsame, abs(r20["drive_score"] - r200["drive_score"]))
            maxd10 = max(maxd10, abs(r10["drive_score"] - r200["drive_score"]))
    check("T12 cold short trip uses >= 20 % more energy per km than one long trip (warm-up), measured consumption", minratio >= 1.2, f"10 km / 200 km consumption ratio, minimum {minratio:.2f}")
    check("T13 normalised consumption is the same for a 10 km trip and a 200 km trip (warm-up removed)", maxnorm < 0.15, f"max difference {maxnorm:.2f} kWh/100 km")
    check("T14 200 km as one trip and as 20 short trips give the same Drive Score (<= 1 pt)", maxsame <= 1.0, f"max difference {maxsame:.2f} pts")
    check("T15 10 km versus 200 km Drive Score differs <= 6 pts (only the weights move with data volume)", maxd10 <= 6.0, f"max difference {maxd10:.1f} pts")
    # T16-T18 temperature: 20 C is the zero point, no heating or cooling is needed inside +-3 K
    def flat(pf, T, total=10, trip=10):
        key = f"T{T}"; CONDITIONS[key] = dict(T=float(T), up=[0] * 10, down=[0] * 10)
        try:
            bs = simulate(pf, key) if total == 10 else window(pf, key, total, trip)
            return score(aggregate(bs))
        finally:
            del CONDITIONS[key]
    base = flat("Normal", 20)
    inband = [flat("Normal", t) for t in (17, 18, 22, 23)]
    dcons = max(abs(r["e_raw"] - base["e_raw"]) for r in inband)
    dscore = max(abs(r["drive_score"] - base["drive_score"]) for r in inband)
    check("T16 inside the comfort zone (17 to 23 C) temperature has no effect on measured consumption or score",
          dcons < 0.35 and dscore < 0.5, f"max change {dcons:.2f} kWh/100 km, {dscore:.2f} pts")
    check("T16b temperature weight stays at its minimum inside the comfort zone",
          all(r["w"]["TMP"] <= base["w"]["TMP"] + 1e-9 for r in inband), f"{base['w']['TMP']*100:.1f} %")
    cold, hot = flat("Normal", -5), flat("Normal", 35)
    check("T17 heating (-5 C) and cooling (35 C) both raise measured consumption by >= 15 % versus 20 C",
          cold["e_raw"] >= 1.15 * base["e_raw"] and hot["e_raw"] >= 1.15 * base["e_raw"],
          f"20 C {base['e_raw']:.1f}, -5 C {cold['e_raw']:.1f}, 35 C {hot['e_raw']:.1f} kWh/100 km")
    dc, dh = base["drive_score"] - cold["drive_score"], abs(hot["drive_score"] - base["drive_score"])
    check("T17b cooling (35 C) leaves the score within 3 pts; heating weather (-5 C) lowers it by 1 to 12 pts (grip context)",
          dh <= 3.0 and 1.0 <= dc <= 12.0,
          f"20 C {base['drive_score']:.1f}, -5 C {cold['drive_score']:.1f}, 35 C {hot['drive_score']:.1f}")
    # T18 context factor bounds and monotonicity
    fs = [ctx_factor(t, g) for t in (-20, -5, 5, 20, 35) for g in (0, 1, 3, 6)]
    check("T18 context factor stays within 0.70 to 1.00, equals 1.00 at 5 C and above on flat roads, and never rises with worse conditions",
          all(CTX_MIN <= f <= 1.0 for f in fs) and ctx_factor(5, 0) == 1.0 and ctx_factor(35, 0) == 1.0
          and ctx_factor(-5, 3) < ctx_factor(-5, 0) < ctx_factor(5, 0) and ctx_factor(5, 3) < ctx_factor(5, 0),
          f"cold-hilly {ctx_factor(-5, 2.8):.3f}")
    # T19 worse regen capture or more climate use never raises the Drive Score (review: reference, not driver values)
    import copy
    def with_driver(pf, cd, cap=None, kappa=None):
        bs = simulate(pf, cd); T0 = CONDITIONS[cd]["T"]
        for b in bs:
            if cap is not None: b.e_regen_down_kwh *= cap
            if kappa is not None:
                extra = (kappa - 1.0) * b.e_aux_kwh; b.e_aux_kwh += extra; b.e_meas_kwh += extra
        return score(aggregate(bs))["drive_score"]
    caps = [with_driver("Normal", "cold-hilly", cap=c) for c in (1.4, 1.0, 0.5, 0.1, 0.0)]
    check("T19 less regen capture on descents never raises the Drive Score (cap x1.4, x1, x0.5, x0.1, x0)",
          all(caps[i] >= caps[i + 1] - 1e-9 for i in range(4)), ", ".join(f"{x:.1f}" for x in caps))
    ks = [with_driver("Normal", "cold-hilly", kappa=k) for k in (0.8, 1.0, 1.3, 1.6)]
    check("T19b more climate use never raises the Drive Score (kappa x0.8, x1, x1.3, x1.6)",
          all(ks[i] >= ks[i + 1] - 1e-9 for i in range(3)), ", ".join(f"{x:.1f}" for x in ks))
    # T20 a perfect driver on a mild flat road can reach 100; no jump at the 30 m descent threshold
    pa = aggregate(simulate("Eco", "mild-flat")); pa["dv_weighted"] = pa["dv_total"]; pa["e_meas_kwh"] = 10.0 * pa["dist_km"] / 100 * 1.0
    pr_ = score(pa)
    check("T20 a perfect ESP and BMS driver on a mild flat road reaches 100", pr_["drive_score"] > 99.5, f"{pr_['drive_score']:.1f}")
    prev = None; jump = 0.0
    for dm in range(0, 61):
        a2 = aggregate(simulate("Normal", "mild-hilly")); sc = a2["h_down_m"]
        a2["h_down_m"] = float(dm); a2["h_up_m"] = float(dm); a2["e_regen_down_kwh"] = 0.30 * (M * G * ETA_T * eta_r(20) * dm / 3.6e6)
        v = score(a2)["drive_score"]
        if prev is not None: jump = max(jump, abs(v - prev))
        prev = v
    check("T21 no jump when descent grows from 0 to 60 m (step <= 0.6 pts per metre)", jump <= 0.6, f"max step {jump:.2f} pts")
    # T22 rating follows the rounded score
    check("T22 rating thresholds apply to the rounded score (89.9 -> 90 Excellent, 79.6 -> 80 Good, 59.5 -> 60 Balanced)",
          rating(89.9) == "Excellent" and rating(79.6) == "Good" and rating(59.5) == "Balanced" and rating(59.4) == "Needs Improvement")
    # T23 normalised consumption is never negative, even for an inconsistent input (12 kWh/100 km on a 50 m/km climb)
    a3 = aggregate(simulate("Eco", "mild-hilly")); a3["h_up_m"] *= 4; a3["h_down_m"] *= 4
    check("T23 normalised consumption never below the 3 kWh/100 km floor", score(a3)["e_norm"] >= E_NORM_MIN, f"{score(a3)['e_norm']:.1f}")
    return grid

def export(grid):
    with open(os.path.join(HERE, "trip10_buckets.csv"), "w") as f:
        f.write("profile,condition,km," + ",".join(Bucket.FIELDS) + "\n")
        for (pf, cd), (bs, _, _) in grid.items():
            for i, b in enumerate(bs, 1):
                f.write(f"{pf},{cd},{i}," + ",".join(f"{getattr(b, k):.6f}" for k in Bucket.FIELDS) + "\n")
    vec = []
    for (pf, cd), (bs, a, r) in grid.items():
        vec.append(dict(id=f"{pf}/{cd}", buckets=[b.dict() for b in bs],
                        expected=dict(S=r["S"], weights=r["w"], drive_score=r["drive_score"],
                                      e_norm=r["e_norm"], rating=rating(r["drive_score"])), tolerance=1e-6))
    wins = []
    for pf in ("Eco", "Normal", "Aggressive"):
        for cd in ("mild-flat", "cold-hilly"):
            for total, trip in ((200, 200), (200, 10), (1000, 200)):
                a = aggregate(window(pf, cd, total, trip)); r = score(a)
                wins.append(dict(id=f"{pf}/{cd}/{total}km-in-{total // trip}-trips", window=a,
                                 expected=dict(S=r["S"], weights=r["w"], drive_score=r["drive_score"], e_raw=r["e_raw"], e_norm=r["e_norm"]), tolerance=1e-6))
    json.dump(dict(trips=vec, windows=wins), open(os.path.join(HERE, "testvectors.json"), "w"), indent=1)
    json.dump(RESULTS, open(os.path.join(HERE, "validation_results.json"), "w"), indent=1)

if __name__ == "__main__":
    g = run()
    export(g)
    print()
    print(f"{'profile':11}{'condition':12}{'e_raw':>6}{'e_norm':>7}{'ESP':>6}{'BMS':>6}{'TMP':>6}{'TER':>6}{'DriveScore':>11}  {'rating':18}{'raw v1.0':>9}")
    for t in RESULTS["table"]:
        print(f"{t['profile']:11}{t['condition']:12}{t['e_raw']:6.1f}{t['e_norm']:7.1f}{t['S_ESP']:6.1f}{t['S_BMS']:6.1f}"
              f"{t['S_TMP']:6.1f}{t['S_TER']:6.1f}{t['drive_score']:11.1f}  {t['rating']:18}{t['raw_score']:9.1f}")
    npass = sum(t["passed"] for t in RESULTS["tests"])
    print(f"\n{npass}/{len(RESULTS['tests'])} tests passed")
    sys.exit(0 if npass == len(RESULTS["tests"]) else 1)
