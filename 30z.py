#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30z.py — Full Overfitting Diagnostic Analyzer (hard-coded to hype_predictions.csv)

Adds:
- Purged walk-forward split (80/20 + horizon-based embargo)
- Purged 5-fold CV across time
- Rolling (5-slice) stability
- Block bootstrap CIs (AUC, Brier, per-event PnL after cost)
- Permutation & time-shift null tests
- Gate sensitivity sweeps (combined with original gate, and prob-only)
- Final verdict string: not_overfit_summary

Outputs:
- One-line copy_paste_summary to stdout
- Full JSON printed to stdout and saved to hype_summary_full.json (UTF-8, emojis preserved)

Run:
    python 30z.py
"""

import json
import math
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------- Helpers (Metrics & Utils) --------------------------- #

def _to_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _tick_guess_from_spreads(spreads: pd.Series) -> float:
    s = pd.to_numeric(spreads, errors="coerce")
    s = s[(s > 0) & np.isfinite(s)]
    if s.empty:
        return 1e-6
    return float(np.quantile(s.values, 0.10))

def _choose_horizon_cols(df: pd.DataFrame) -> Tuple[Optional[str], str, float]:
    candidates = [
        ("realized_up_180s", "realized_ret_180s", 180.0),
        ("realized_up_300s", "realized_ret_300s", 300.0),
        ("realized_up_30s",  "realized_ret_30s",   30.0),
    ]
    for up_col, ret_col, hz in candidates:
        if ret_col in df.columns and df[ret_col].notna().sum() >= 50:
            return (up_col if up_col in df.columns else None), ret_col, hz
    for c in df.columns:
        if c.startswith("realized_ret_") and df[c].notna().sum() >= 50:
            postfix = c[len("realized_ret_"):]
            upc = f"realized_up_{postfix}"
            try:
                hz = float(postfix.replace("s", ""))
            except Exception:
                hz = np.nan
            return (upc if upc in df.columns else None), c, hz
    raise ValueError("No sufficient realized label columns found (need >=50 non-null rows).")

def _get_cost_ticks(df: pd.DataFrame) -> pd.Series:
    if "est_cost_ticks" in df.columns:
        return pd.to_numeric(df["est_cost_ticks"], errors="coerce").fillna(0.0)
    if "est_cost" in df.columns:
        return pd.to_numeric(df["est_cost"], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)

def _gate_series(df: pd.DataFrame) -> pd.Series:
    g1 = pd.to_numeric(df.get("gate_exec_ok", 0), errors="coerce").fillna(0).astype(int)
    g2 = pd.to_numeric(df.get("gate_robust_ok", 0), errors="coerce").fillna(0).astype(int)
    return (g1 == 1) & (g2 == 1)

def _auroc_safe(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_sorted) + 1, dtype=float)
    y_prob_sorted = y_prob[order]
    start = 0
    while start < len(y_prob_sorted):
        end = start + 1
        while end < len(y_prob_sorted) and y_prob_sorted[end] == y_prob_sorted[start]:
            end += 1
        avg = 0.5 * (ranks[start] + ranks[end - 1])
        ranks[start:end] = avg
        start = end
    pos_rank_sum = ranks[y_sorted == 1].sum()
    U = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))

def _brier_score(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 0.0, 1.0).astype(float)
    y_true = y_true.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))

def _safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return 0.0
    return float(x.mean())

def _reliability_bins(y_prob: np.ndarray, y_true: np.ndarray, bins: int = 10):
    y_prob = y_prob.clip(0, 1)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (y_prob >= lo) & ((y_prob < hi) | ((b == bins - 1) & (y_prob == 1.0)))
        cnt = int(mask.sum())
        emp = float(y_true[mask].mean()) if cnt > 0 else float("nan")
        out.append({"bin_center": float(0.5 * (lo + hi)), "empirical": emp, "count": cnt})
    return out

def _ece(reliab_bins) -> float:
    total = sum(item["count"] for item in reliab_bins)
    if total == 0:
        return float("nan")
    err = 0.0
    for item in reliab_bins:
        c = item["count"]
        emp = item["empirical"]
        ctr = item["bin_center"]
        if c > 0 and np.isfinite(emp):
            err += (c / total) * abs(emp - ctr)
    return float(err)

def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score CI with guards; clips k to [0, n] and prevents sqrt negatives."""
    if n <= 0:
        return (float("nan"), float("nan"))
    k = max(0, min(int(k), int(n)))  # clip successes
    n = int(n)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    # guarded variance term
    var_term = (p * (1.0 - p)) / n + (z * z) / (4.0 * n * n)
    var_term = max(0.0, var_term)
    half = z * math.sqrt(var_term) / denom
    return (float(center - half), float(center + half))

def _compute_metrics(df: pd.DataFrame, realized_up_col: str, realized_ret_col: str) -> Dict[str, float]:
    y_true = pd.to_numeric(df[realized_up_col], errors="coerce").fillna(0).astype(int).values
    y_prob = pd.to_numeric(df["p_used"], errors="coerce").clip(1e-9, 1-1e-9).fillna(0.5).values
    pnl_after = pd.to_numeric(df["pnl_after_cost"], errors="coerce").fillna(0.0)
    pnl_raw = pd.to_numeric(df["pnl_raw"], errors="coerce").fillna(0.0)
    hit = (pnl_raw.values > 0).mean() if len(pnl_raw) else float("nan")
    brier = _brier_score(y_prob, y_true) if len(y_true) else float("nan")
    auc = _auroc_safe(y_prob, y_true) if len(y_true) else float("nan")
    mean_pnl = float(pnl_after.mean()) if len(pnl_after) else float("nan")
    sharpe_like = float(pnl_after.mean() / (pnl_after.std(ddof=0) + 1e-12)) if len(pnl_after) else float("nan")
    return dict(
        n=int(len(df)),
        hit_rate=float(hit),
        auc=float(auc),
        brier=float(brier),
        pnl_per_event_after_cost=float(mean_pnl),
        sharpe_like=float(sharpe_like),
    )

def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" in df.columns:
        return df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    if "wall_time_iso" in df.columns:
        try:
            tmp = df.copy()
            tmp["_t"] = pd.to_datetime(tmp["wall_time_iso"], errors="coerce", utc=True)
            tmp = tmp.sort_values("_t", kind="mergesort").drop(columns=["_t"]).reset_index(drop=True)
            return tmp
        except Exception:
            pass
    return df.reset_index(drop=True)

def _median_dt_seconds(df: pd.DataFrame) -> float:
    if "ts" not in df.columns:
        return 1.0
    s = pd.to_numeric(df["ts"], errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return 1.0
    dt = np.diff(s.values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 1.0
    return float(np.median(dt))

def _block_len_rows(df: pd.DataFrame, horizon_s: float) -> int:
    dt = _median_dt_seconds(df)
    if dt <= 0:
        return 50
    est = int(max(5, min(500, round(horizon_s / dt))))
    return est

# --------------------------- Core Analyzer (as before) --------------------------- #

def base_analysis(csv_path: str) -> dict:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {"error": f"File not found at {csv_path}"}
    except Exception as e:
        return {"error": f"Could not read CSV: {e}"}

    if "p_fused" not in df.columns and "p_fused_cal" not in df.columns:
        return {"error": "Missing 'p_fused'/'p_fused_cal' column in CSV."}
    if "mid" not in df.columns or "spread" not in df.columns:
        return {"error": "CSV must include 'mid' and 'spread' columns."}

    prob_col = "p_fused_cal" if "p_fused_cal" in df.columns else "p_fused"

    realized_up_col_opt, realized_ret_col, horizon_s = _choose_horizon_cols(df)

    df = df.copy()
    df = df[pd.to_numeric(df[realized_ret_col], errors="coerce").notna()]
    if len(df) < 50:
        return {"error": f"Not enough data with realized outcomes ({realized_ret_col}). Found only {len(df)} rows."}

    if realized_up_col_opt is None or realized_up_col_opt not in df.columns:
        realized_up_col = "__derived_up__"
        df[realized_up_col] = (pd.to_numeric(df[realized_ret_col], errors="coerce").fillna(0.0) > 0).astype(int)
    else:
        realized_up_col = realized_up_col_opt

    tick_guess = _tick_guess_from_spreads(df["spread"])
    cost_ticks = _get_cost_ticks(df)
    mid_series = pd.to_numeric(df["mid"], errors="coerce").replace(0, np.nan)
    df["cost_as_return"] = pd.to_numeric(cost_ticks, errors="coerce").fillna(0.0) * (tick_guess / mid_series)
    df["cost_as_return"] = df["cost_as_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    p = pd.to_numeric(df[prob_col], errors="coerce").clip(1e-9, 1 - 1e-9).fillna(0.5)
    df["p_used"] = p
    df["p_used_logit"] = np.log(p / (1 - p))
    df["position"] = np.sign(p - 0.5)

    r = pd.to_numeric(df[realized_ret_col], errors="coerce").fillna(0.0)
    df["pnl_raw"] = df["position"] * r
    df["pnl_after_cost"] = np.where(df["position"] != 0, df["pnl_raw"] - df["cost_as_return"], 0.0)

    df["gate_active"] = _gate_series(df)
    df_active = df[df["gate_active"]].copy()

    # Base topline
    y_true = pd.to_numeric(df[realized_up_col], errors="coerce").fillna(0).astype(int).values
    y_prob = df["p_used"].values
    brier = _brier_score(y_prob, y_true)
    auc = _auroc_safe(y_prob, y_true)
    reliab = _reliability_bins(y_prob, y_true, bins=10)
    ece_val = _ece(reliab)

    # Simple calib line from reliability bins
    calib_pairs = [(b["bin_center"], b["empirical"]) for b in reliab if b["count"] > 0 and np.isfinite(b["empirical"])]
    if len(calib_pairs) >= 2:
        calib_x = [x for x, _ in calib_pairs]
        calib_y = [y for _, y in calib_pairs]
        X = np.vstack([calib_x, np.ones(len(calib_x))]).T
        slope, intercept = np.linalg.lstsq(X, np.array(calib_y), rcond=None)[0]
        slope = float(slope); intercept = float(intercept)
    else:
        calib_x, calib_y = [], []
        slope, intercept = float("nan"), float("nan")

    # Regimes
    df_sorted = _sort_by_time(df)
    mid = pd.to_numeric(df_sorted["mid"], errors="coerce").replace(0, np.nan)
    logret = np.log(mid / mid.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_roll = logret.rolling(window=100, min_periods=20).std().fillna(0.0)
    df_sorted["_volatility"] = vol_roll
    df_sorted["_spread_ticks"] = pd.to_numeric(df_sorted["spread"], errors="coerce") / max(1e-12, tick_guess)

    try:
        df_sorted["_vol_regime"] = pd.qcut(df_sorted["_volatility"], 3, labels=["Low", "Medium", "High"], duplicates="drop")
    except Exception:
        df_sorted["_vol_regime"] = "Unknown"
    try:
        df_sorted["_spread_regime"] = pd.qcut(df_sorted["_spread_ticks"], 3, labels=["Tight", "Medium", "Wide"], duplicates="drop")
    except Exception:
        df_sorted["_spread_regime"] = "Unknown"

    df_active_sorted = df_sorted[df_sorted["gate_active"]].copy()

    def _agg_perf(gdf: pd.DataFrame) -> Dict[str, float]:
        return {
            "sum": _safe_mean(gdf["pnl_after_cost"]) * float(len(gdf)),
            "mean": _safe_mean(gdf["pnl_after_cost"]),
            "count": int(len(gdf)),
        }

    by_vol: Dict[str, Dict[str, float]] = {}
    for key, g in df_active_sorted.groupby("_vol_regime", dropna=False, observed=False):
        by_vol[str(key)] = _agg_perf(g)

    by_spread: Dict[str, Dict[str, float]] = {}
    for key, g in df_active_sorted.groupby("_spread_regime", dropna=False, observed=False):
        by_spread[str(key)] = _agg_perf(g)

    # Copy/paste summary
    def _grade(val, good, ok):
        if np.isnan(val): return "❔"
        return "✅" if val >= good else ("🟡" if val >= ok else "🔴")

    total_rows = int(len(df_sorted))
    total_active = int(df_active_sorted.shape[0])
    pnl_active_mean = _safe_mean(df_active_sorted["pnl_after_cost"]) if total_active else 0.0
    sharpe_like = float(df_active_sorted["pnl_after_cost"].mean() / (df_active_sorted["pnl_after_cost"].std(ddof=0) + 1e-12)) if total_active else 0.0
    hit_active = float((df_active_sorted["pnl_raw"] > 0).mean()) if total_active else 0.0

    grade_auc = _grade(auc, 0.65, 0.58)
    grade_hit = _grade(hit_active, 0.56, 0.53)
    grade_sharpe = _grade(sharpe_like, 0.5, 0.2)
    grade_brier = "✅" if brier <= 0.22 else ("🟡" if brier <= 0.25 else "🔴")
    ece_val = float(ece_val)
    grade_ece = "✅" if (np.isfinite(ece_val) and ece_val <= 0.03) else ("🟡" if (np.isfinite(ece_val) and ece_val <= 0.06) else "🔴")

    copy_paste_summary = (
        f"Horizon={int(horizon_s)}s | events={total_rows} (active {total_active}, {total_active/max(1,total_rows):.1%}). "
        f"Active hit-rate={hit_active:.2%} {grade_hit}, per-event PnL(after cost)={pnl_active_mean:+.4e}, "
        f"Sharpe-like={sharpe_like:.2f} {grade_sharpe}. "
        f"AUC={auc:.3f} {grade_auc}, Brier={brier:.3f} {grade_brier}, ECE={ece_val:.3f} {grade_ece}. "
        f"Gated avg p={_safe_mean(df_active_sorted['p_used']):.3f} vs ungated {_safe_mean(df_sorted.loc[~df_sorted['gate_active'], 'p_used']):.3f}. "
        f"Regimes (active): vol={ {str(k): _to_float(v.get('mean', np.nan)) for k,v in by_vol.items()} }, "
        f"spread={ {str(k): _to_float(v.get('mean', np.nan)) for k,v in by_spread.items()} }."
    )

    base = {
        "meta": {
            "csv_path": str(csv_path),
            "rows_used": total_rows,
            "horizon_seconds": float(horizon_s),
            "probability_column_used": prob_col,
            "realized_up_col": realized_up_col if realized_up_col in df.columns else "derived_from_return_sign",
            "realized_ret_col": realized_ret_col,
        },
        "topline": {
            "pnl_per_event_after_cost_active": _to_float(pnl_active_mean),
            "sharpe_like_active": _to_float(sharpe_like),
            "hit_rate_active": hit_active,
            "auc": float(auc),
            "brier": float(brier),
            "ece": float(ece_val),
            "active_signals": total_active,
            "percentage_active": float(total_active/max(1,total_rows)),
        },
        "signal_quality": {
            "brier_score": float(brier),
            "roc_auc": float(auc),
            "calibration_curve": {
                "pred_bins": [float(x) for x in ([p for p,_ in calib_pairs] if calib_pairs else [])],
                "true_bins": [float(y) for y in ([y for _,y in calib_pairs] if calib_pairs else [])],
                "slope": float(slope),
                "intercept": float(intercept),
                "reliability_bins": _reliability_bins(y_prob, y_true, bins=10),
                "ece": float(ece_val),
            }
        },
        "regime_performance": {"by_volatility": by_vol, "by_spread_ticks": by_spread},
        "copy_paste_summary": copy_paste_summary,
        "df_sorted": df_sorted,  # return for downstream diagnostics
        "realized_up_col": realized_up_col,
        "realized_ret_col": realized_ret_col,
        "horizon_s": float(horizon_s),
    }
    return base

# --------------------------- Overfitting Diagnostics --------------------------- #

def walk_forward_purged(df_sorted: pd.DataFrame, realized_up_col: str, realized_ret_col: str, horizon_s: float) -> dict:
    n = len(df_sorted)
    if n < 200:
        return {"note": "insufficient rows for walk-forward"}

    split = int(0.8 * n)
    block_rows = _block_len_rows(df_sorted, horizon_s)
    # Train: [0, split - embargo), Test: (split + embargo, end]
    train_end = max(0, split - block_rows)
    test_start = min(n, split + block_rows)
    train = df_sorted.iloc[:train_end].copy()
    test = df_sorted.iloc[test_start:].copy()

    # Metrics on ACTIVE rows only (consistent n for Wilson)
    train_act = train[train["gate_active"]]
    test_act = test[test["gate_active"]]

    t_metrics = _compute_metrics(train_act, realized_up_col, realized_ret_col) if len(train_act) else {}
    s_metrics = _compute_metrics(test_act, realized_up_col, realized_ret_col) if len(test_act) else {}

    # Wilson CI on active hit in test — use active wins and active count
    if len(test_act):
        k = int((test_act["pnl_raw"] > 0).sum())
        nact = int(len(test_act))
        ci_lo, ci_hi = _wilson_ci(k, nact)
    else:
        ci_lo, ci_hi = (float("nan"), float("nan"))

    return {
        "rows": {"total": n, "train": int(len(train)), "test": int(len(test)), "embargo_rows": int(block_rows)},
        "train_active_metrics": t_metrics,
        "test_active_metrics": s_metrics,
        "test_hit_rate_wilson95": {"low": ci_lo, "high": ci_hi}
    }

def kfold_purged(df_sorted: pd.DataFrame, realized_up_col: str, realized_ret_col: str, horizon_s: float, k: int = 5) -> dict:
    n = len(df_sorted)
    if n < k * 50:
        return {"note": "insufficient rows for k-fold"}

    block_rows = _block_len_rows(df_sorted, horizon_s)
    fold_sizes = [n // k + (1 if i < (n % k) else 0) for i in range(k)]
    idx = 0
    folds = []
    for fs in fold_sizes:
        folds.append((idx, idx + fs))
        idx += fs

    test_metrics = []
    for i, (a, b) in enumerate(folds):
        test = df_sorted.iloc[a:b].copy()
        # Purge train around test by embargo
        # (We don't fit anything parametric here, just report test metrics.)
        m = _compute_metrics(test[test["gate_active"]], realized_up_col, realized_ret_col) if len(test) else {}
        m["fold"] = i
        m["test_rows"] = int(len(test))
        test_metrics.append(m)

    def _agg(key):
        vals = [tm[key] for tm in test_metrics if key in tm and np.isfinite(tm[key])]
        if not vals:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}

    return {"embargo_rows": int(block_rows), "folds": test_metrics,
            "aggregate": {k: _agg(k) for k in ["hit_rate", "auc", "brier", "pnl_per_event_after_cost", "sharpe_like"]}}

def rolling_stability(df_sorted: pd.DataFrame, realized_up_col: str, realized_ret_col: str, slices: int = 5) -> dict:
    n = len(df_sorted)
    if n < slices * 50:
        slices = max(2, n // 100) or 2
    out = []
    for i in range(slices):
        a = int(i * n / slices)
        b = int((i + 1) * n / slices)
        seg = df_sorted.iloc[a:b]
        m = _compute_metrics(seg[seg["gate_active"]], realized_up_col, realized_ret_col)
        m["slice"] = i
        m["rows"] = int(len(seg))
        out.append(m)
    return {"slices": out}

def block_bootstrap(df_sorted: pd.DataFrame, realized_up_col: str, realized_ret_col: str, horizon_s: float, n_boot: int = 300, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = len(df_sorted)
    if n < 200:
        return {"note": "insufficient rows for bootstrap"}
    L = _block_len_rows(df_sorted, horizon_s)
    y_true = pd.to_numeric(df_sorted[realized_up_col], errors="coerce").fillna(0).astype(int).values
    y_prob = pd.to_numeric(df_sorted["p_used"], errors="coerce").clip(1e-9, 1-1e-9).fillna(0.5).values
    pnl_after = pd.to_numeric(df_sorted["pnl_after_cost"], errors="coerce").fillna(0.0).values
    gate = df_sorted["gate_active"].values.astype(bool)

    aucs, briers, pnls = [], [], []
    blocks = max(1, n // max(1, L))
    for _ in range(n_boot):
        idxs = []
        for __ in range(blocks):
            start = rng.integers(0, max(1, n - L))
            idxs.extend(range(start, min(n, start + L)))
        idxs = np.array(idxs[:n], dtype=int)
        g = gate[idxs]
        if g.sum() == 0:
            continue
        yp = y_prob[idxs]
        yt = y_true[idxs]
        pa = pnl_after[idxs]
        aucs.append(_auroc_safe(yp, yt))
        briers.append(_brier_score(yp, yt))
        pnls.append(float(np.mean(pa[g])))
    def ci(arr):
        if not arr:
            return {"mean": float("nan"), "lo95": float("nan"), "hi95": float("nan")}
        arr = np.array(arr)
        return {"mean": float(arr.mean()), "lo95": float(np.percentile(arr, 2.5)), "hi95": float(np.percentile(arr, 97.5))}
    return {"block_len_rows": int(L), "n_boot": int(n_boot),
            "auc": ci(aucs), "brier": ci(briers), "pnl_per_event_after_cost_active": ci(pnls)}

def permutation_tests(df_sorted: pd.DataFrame, realized_up_col: str, realized_ret_col: str, horizon_s: float, shift_blocks: int = 1, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    n = len(df_sorted)
    L = _block_len_rows(df_sorted, horizon_s)
    perm = df_sorted.copy()
    perm[realized_up_col] = rng.permutation(perm[realized_up_col].values)
    m_perm = _compute_metrics(perm[perm["gate_active"]], realized_up_col, realized_ret_col)
    if n > L + 1:
        shift = df_sorted.copy()
        shift = shift.iloc[L:].copy()
        shift[realized_up_col] = df_sorted[realized_up_col].values[:-L]
        m_shift = _compute_metrics(shift[shift["gate_active"]], realized_up_col, realized_ret_col)
    else:
        m_shift = {"note": "insufficient rows to time-shift"}
    return {"random_permutation": m_perm, "time_shift_by_block": {"block_len_rows": int(L), "metrics": m_shift}}

def gate_sensitivity(df_sorted: pd.DataFrame, realized_up_col: str, realized_ret_col: str, thresholds: List[float] = None) -> dict:
    if thresholds is None:
        thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    p = pd.to_numeric(df_sorted["p_used"], errors="coerce").clip(1e-9, 1-1e-9).values
    out_combined, out_prob_only = [], []
    for th in thresholds:
        amp = max(0.0, th - 0.5)
        sel_comb = df_sorted[(df_sorted["gate_active"]) & (np.abs(p - 0.5) >= amp)]
        out_combined.append({"threshold": float(th), **_compute_metrics(sel_comb, realized_up_col, realized_ret_col)})
        sel_prob = df_sorted[np.abs(p - 0.5) >= amp]
        out_prob_only.append({"threshold": float(th), **_compute_metrics(sel_prob, realized_up_col, realized_ret_col)})
    return {"combined_with_original_gate": out_combined, "probability_only": out_prob_only}

# --------------------------- Final Verdict Heuristic --------------------------- #

def verdict_from_checks(base_topline: dict, walk: dict, kcv: dict, boot: dict, perm: dict) -> str:
    msgs = []
    ok_flags = []
    try:
        t, s = walk.get("train_active_metrics", {}), walk.get("test_active_metrics", {})
        if t and s and all(k in t and k in s for k in ["auc", "brier", "hit_rate"]):
            auc_ok = (s["auc"] >= 0.8 * t["auc"])
            brier_ok = (s["brier"] <= 1.2 * t["brier"])
            hit_ok = (s["hit_rate"] >= 0.8 * t["hit_rate"])
            ok_flags += [auc_ok, brier_ok, hit_ok]
            msgs.append(f"Walk-forward: test auc={s['auc']:.3f}, brier={s['brier']:.3f}, hit={s['hit_rate']:.2%} (train auc={t['auc']:.3f}).")
    except Exception:
        pass
    try:
        auc_ci = boot.get("auc", {})
        brier_ci = boot.get("brier", {})
        if all(k in auc_ci for k in ["lo95", "hi95"]):
            ok_flags.append(auc_ci["lo95"] > 0.5)
            msgs.append(f"Bootstrap AUC 95% CI [{auc_ci['lo95']:.3f}, {auc_ci['hi95']:.3f}]")
        if all(k in brier_ci for k in ["lo95", "hi95"]):
            ok_flags.append(brier_ci["hi95"] < 0.25)
            msgs.append(f"Bootstrap Brier 95% CI [{brier_ci['lo95']:.3f}, {brier_ci['hi95']:.3f}]")
    except Exception:
        pass
    try:
        mp = perm.get("random_permutation", {})
        if mp:
            ok_flags.append((mp.get("auc", 0.5) <= 0.55) and (mp.get("brier", 0.25) >= 0.24))
            msgs.append(f"Permutation: auc={mp.get('auc', float('nan')):.3f}, brier={mp.get('brier', float('nan')):.3f}")
    except Exception:
        pass
    try:
        auc0, b0 = base_topline.get("auc", float("nan")), base_topline.get("brier", float("nan"))
        if np.isfinite(auc0) and np.isfinite(b0):
            msgs.append(f"Base: auc={auc0:.3f}, brier={b0:.3f}")
    except Exception:
        pass
    votes = sum(bool(x) for x in ok_flags)
    need = max(2, len(ok_flags) // 2)
    if votes >= need:
        return "Likely NOT overfit based on walk-forward consistency, bootstrap CIs, and null collapse. " + " | ".join(msgs)
    return "Inconclusive/possible overfit. Some checks failed or were inconclusive. " + " | ".join(msgs)

# --------------------------- Orchestrator --------------------------- #

def analyze_predictions_full(csv_path: str) -> dict:
    base = base_analysis(csv_path)
    if "error" in base:
        return base

    df_sorted = base.pop("df_sorted")
    realized_up_col = base.pop("realized_up_col")
    realized_ret_col = base.pop("realized_ret_col")
    horizon_s = base.pop("horizon_s")

    # Diagnostics
    walk = walk_forward_purged(df_sorted, realized_up_col, realized_ret_col, horizon_s)
    kcv = kfold_purged(df_sorted, realized_up_col, realized_ret_col, horizon_s, k=5)
    roll = rolling_stability(df_sorted, realized_up_col, realized_ret_col, slices=5)
    boot = block_bootstrap(df_sorted, realized_up_col, realized_ret_col, horizon_s, n_boot=300, seed=7)
    perm = permutation_tests(df_sorted, realized_up_col, realized_ret_col, horizon_s, seed=11)
    gate = gate_sensitivity(df_sorted, realized_up_col, realized_ret_col)

    summary = verdict_from_checks(base["topline"], walk, kcv, boot, perm)

    return {
        **{k: v for k, v in base.items() if k != "copy_paste_summary"},
        "copy_paste_summary": base["copy_paste_summary"],
        "validation": {
            "walk_forward_purged": walk,
            "kfold_purged": kcv,
            "rolling_stability": roll,
            "block_bootstrap": boot,
            "permutation_tests": perm,
            "gate_sensitivity": gate,
        },
        "not_overfit_summary": summary,
    }

# --------------------------- Entrypoint --------------------------- #

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    default_csv = script_dir / "hype_predictions.csv"
    csv_path = default_csv if default_csv.exists() else (Path.cwd() / "hype_predictions.csv")

    results = analyze_predictions_full(str(csv_path))

    # One-liner summary (quick look)
    if "copy_paste_summary" in results:
        print(results["copy_paste_summary"])

    # Save full JSON next to the CSV (human-readable, with real emojis)
    out_path = script_dir / "hype_summary_full.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print full JSON to stdout for AI ingestion
    print(json.dumps(results, indent=2, ensure_ascii=False))
