#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S12: Layered, math-only 30s price prediction (non-AI) — CSV logger
FIXED (2025-08-17):
  - Single-clock timeline: labels & features use the same (exchange) timestamp.
  - Removed wall-clock mid sampler that polluted the label timeline.
  - OFI normalization uses rolling USD notional (EWMA) + winsorization.
  - Adaptive drift→prob scaling via percentile scale + temperature.
  - LVP update clipping & throttle to prevent blow-ups.
  - Lightweight reliability calibrator; gate on calibrated probability.
  - Longer warmup before gating.

Target: Hyperliquid 'HYPE' perp. No CLI args required.
Python: 3.10+
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Any

try:
    import websockets  # type: ignore
    HAS_WS = True
except ImportError:
    HAS_WS = False


# ------------------------------- Utilities ---------------------------------- #

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def ewma(prev: float, x: float, alpha: float) -> float:
    return alpha * x + (1 - alpha) * prev

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def winsorize(x: float, lo: float, hi: float) -> float:
    return clamp(x, lo, hi)

def percentile_abs(window: Deque[float], p: float) -> float:
    """Percentile of absolute values from a deque without NumPy."""
    if not window:
        return 1.0
    arr = sorted(abs(x) for x in window)
    if len(arr) < 50:
        return 10.0
    k = clamp(int(p * (len(arr) - 1)), 0, len(arr) - 1)
    return max(1.0, arr[k])

def now_s() -> float:
    return time.time()

def iso_utc(ts: float | None = None) -> str:
    if ts is None:
        ts = now_s()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

def epoch_any_to_s(t: Any) -> float:
    """
    Robustly convert various epoch units to seconds. If it's obviously not an epoch,
    fall back to wall clock to keep the internal runtime sane.

    Correct order matters: ns > µs > ms > s.
    """
    try:
        t = float(t)
    except Exception:
        return now_s()
    # ns -> s  (e.g., 1690000000000000000)
    if t > 9e17:
        return t / 1e9
    # μs -> s  (e.g., 1690000000000000)
    if t > 9e14:
        return t / 1e6
    # ms -> s  (e.g., 1690000000000)
    if t > 9e11:
        return t / 1e3
    # s since 1970 (plausible window: >= 2001-09-09)
    if t >= 1_000_000_000:
        return t
    # Not a real epoch (e.g., a small counter) -> wall time
    return now_s()

def _parse_levels(levels):
    """
    Accept both shapes:
      - dicts: {"px": "...", "sz": "..."}
      - arrays: [px, sz, ...]
    """
    def _px_sz(L):
        if isinstance(L, dict):
            return float(L.get("px", 0.0)), float(L.get("sz", 0.0))
        return float(L[0]), float(L[1])
    bids = [[*_px_sz(L)] for L in (levels[0] if levels and levels[0] else [])]
    asks = [[*_px_sz(L)] for L in (levels[1] if levels and len(levels) > 1 and levels[1] else [])]
    return bids, asks


# ----------------------------- Market State --------------------------------- #

@dataclass
class OrderBookState:
    best_bid: float = 0.0
    best_bid_sz: float = 0.0
    best_ask: float = 0.0
    best_ask_sz: float = 0.0
    tick: float = 0.001  # HYPE tick

    def update_from_book(self, bids: List[List[float]], asks: List[List[float]]):
        if bids:
            self.best_bid, self.best_bid_sz = float(bids[0][0]), float(bids[0][1])
        if asks:
            self.best_ask, self.best_ask_sz = float(asks[0][0]), float(asks[0][1])

    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return max(self.tick, self.best_ask - self.best_bid)
        return self.tick

    @property
    def mid(self) -> float:
        if self.best_bid and self.best_ask:
            return 0.5 * (self.best_bid + self.best_ask)
        return 0.0

    @property
    def qib1(self) -> float:
        denom = self.best_bid_sz + self.best_ask_sz
        if denom <= 0:
            return 0.0
        return (self.best_bid_sz - self.best_ask_sz) / denom

    @property
    def microprice(self) -> float:
        denom = self.best_bid_sz + self.best_ask_sz
        if denom <= 0:
            return self.mid
        return (self.best_ask * self.best_bid_sz + self.best_bid * self.best_ask_sz) / denom

    @property
    def micro_pressure(self) -> float:
        if self.tick <= 0:
            return 0.0
        return (self.microprice - self.mid) / self.tick


# ------------------------------- Features ----------------------------------- #

class RollingOFI:
    """Order-flow imbalance over a rolling window (seconds)."""
    def __init__(self, window_s: float = 5.0):
        self.window = window_s
        self.q: Deque[Tuple[float, float]] = deque()
        self.sum = 0.0

    def add(self, ts: float, signed_size: float):
        self.q.append((ts, signed_size))
        self.sum += signed_size
        self._evict(ts)

    def value(self, ts: float) -> float:
        self._evict(ts)
        return self.sum

    def _evict(self, ts: float):
        while self.q and ts - self.q[0][0] > self.window:
            _, v = self.q.popleft()
            self.sum -= v


class RollingCount:
    def __init__(self, window_s: float = 5.0):
        self.window = window_s
        self.q: Deque[Tuple[float, int]] = deque()
        self.count = 0

    def add(self, ts: float, count: int = 1):
        self.q.append((ts, count))
        self.count += count
        self._evict(ts)

    def value(self, ts: float) -> int:
        self._evict(ts)
        return self.count

    def _evict(self, ts: float):
        while self.q and ts - self.q[0][0] > self.window:
            _, c = self.q.popleft()
            self.count -= c


class RollingVol:
    """Realized vol estimator (EWMA of squared returns)."""
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.var = 0.0
        self.last_mid: Optional[float] = None

    def update(self, mid: float):
        if self.last_mid is not None and self.last_mid > 0 and mid > 0:
            r = math.log(mid / self.last_mid)
            self.var = ewma(self.var, r * r, self.alpha)
        self.last_mid = mid

    def sigma(self) -> float:
        return math.sqrt(max(self.var, 1e-12))


class LabelBuffer:
    """Hold mid-prices & emit realized 30s returns when available (same exchange clock)."""
    def __init__(self, horizon_s: float = 30.0):
        self.h = horizon_s
        self.q: Deque[Tuple[float, float]] = deque()  # (ts, mid)

    def add_mid(self, ts: float, mid: float):
        self.q.append((ts, mid))
        self._evict_old(ts)

    def realized_return(self, ts_now: float) -> Optional[Tuple[float, float]]:
        # prune too-old origins
        while self.q and ts_now - self.q[0][0] > self.h + 1.0:
            self.q.popleft()
        target_ts = ts_now - self.h
        origin_mid = None
        origin_ts = None
        for (ts, mid) in reversed(self.q):
            if ts <= target_ts:
                origin_mid = mid
                origin_ts = ts
                break
        if origin_mid is None:
            return None
        end_mid = self.q[-1][1]
        if origin_mid <= 0 or end_mid <= 0:
            return None
        r = math.log(end_mid / origin_mid)
        return (origin_ts, r)

    def _evict_old(self, ts: float):
        while self.q and ts - self.q[0][0] > self.h + 60.0:
            self.q.popleft()


class EWMANorm:
    """EWMA scale tracker (used for rolling USD notional normalization)."""
    def __init__(self, alpha: float = 0.02, init: float = 1000.0):
        self.alpha, self.v = alpha, init
    def update(self, x: float):
        self.v = ewma(self.v, x, self.alpha)


# ------------------------------- HC-QR -------------------------------------- #

@dataclass
class HawkesParams:
    mu_up: float = 0.02
    mu_dn: float = 0.02
    a_up: float = 0.08
    a_dn: float = 0.08
    beta: float = 1.0 / 3.0  # ~3s decay


@dataclass
class HCQRState:
    lam_up: float = 0.02
    lam_dn: float = 0.02
    last_ts: Optional[float] = None


class HCQR:
    def __init__(self, params: HawkesParams, step_s: float = 0.5, horizon_s: float = 30.0):
        self.p = params
        self.step = step_s
        self.T = horizon_s
        self.state = HCQRState(lam_up=params.mu_up, lam_dn=params.mu_dn)

    def update_event(self, ts: float, side: str):
        s = self.state
        if s.last_ts is None:
            s.last_ts = ts
        dt = max(0.0, ts - s.last_ts)
        decay = math.exp(-self.p.beta * dt)
        s.lam_up = self.p.mu_up + (s.lam_up - self.p.mu_up) * decay
        s.lam_dn = self.p.mu_dn + (s.lam_dn - self.p.mu_dn) * decay
        if side == "buy":
            s.lam_up += self.p.a_up
        elif side == "sell":
            s.lam_dn += self.p.a_dn
        s.last_ts = ts

    def prob_up_within_T(self) -> float:
        s = self.state
        if s.last_ts is None:
            return 0.5
        lam_u = s.lam_up
        lam_d = s.lam_dn
        p_up = 0.0
        surv = 1.0
        u = 0.0
        while u < self.T:
            lam_u = self.p.mu_up + (lam_u - self.p.mu_up) * math.exp(-self.p.beta * self.step)
            lam_d = self.p.mu_dn + (lam_d - self.p.mu_dn) * math.exp(-self.p.beta * self.step)
            total = max(1e-9, lam_u + lam_d)
            du = min(self.step, self.T - u)
            chunk = lam_u / total * (1.0 - math.exp(-total * du)) if total > 0 else 0.0
            p_up += surv * chunk
            surv *= math.exp(-total * du)
            u += du
        p_up += 0.5 * surv
        return clamp(p_up, 0.0, 1.0)


# ----------------------- LVP: Local Volterra Predictor ---------------------- #

def soft_threshold_l1(vec: List[float], budget: float) -> List[float]:
    v = [abs(x) for x in vec]
    s = sum(v)
    if s <= budget or budget <= 0:
        return vec[:]
    lo, hi = 0.0, max(v) if v else 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s_mid = sum(max(0.0, vi - mid) for vi in v)
        if s_mid > budget:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    out = []
    for x in vec:
        ax = abs(x)
        shrink = max(0.0, ax - lam)
        out.append(math.copysign(shrink, x))
    return out

class LVP:
    def __init__(self, K: int = 10, lam: float = 0.95, l1_budget: float = 5.0, tick: float = 0.01):
        self.K = K
        self.lam = lam
        self.tick = tick
        self.w1 = [0.0] * K
        self.w2 = [0.0] * K
        self.P = [[0.0]* (2*K) for _ in range(2*K)]
        for i in range(2*K):
            self.P[i][i] = 1000.0
        self.l1_budget = l1_budget
        self.buf: Deque[float] = deque(maxlen=K)

    def push_ofi(self, x: float):
        self.buf.appendleft(float(x))

    def predict_drift_ticks(self) -> float:
        x1 = list(self.buf) + [0.0] * (self.K - len(self.buf))
        x2 = [v * v for v in x1]
        drift = sum(wi * xi for wi, xi in zip(self.w1, x1)) + sum(wi * xi for wi, xi in zip(self.w2, x2))
        return drift

    def update(self, realized_ret_ticks: float):
        x1 = list(self.buf) + [0.0] * (self.K - len(self.buf))
        x2 = [v * v for v in x1]
        phi = x1 + x2
        Pf = [sum(self.P[i][j] * phi[j] for j in range(len(phi))) for i in range(len(phi))]
        denom = self.lam + sum(phi[i] * Pf[i] for i in range(len(phi)))
        if denom <= 1e-12:
            return
        Kvec = [v / denom for v in Pf]
        y_hat = sum(self.w1[i] * x1[i] for i in range(self.K)) + sum(self.w2[i] * x2[i] for i in range(self.K))
        err = realized_ret_ticks - y_hat
        # gradient/innovation clipping & learning-rate throttle
        err = clamp(err, -5.0, 5.0)
        eta = 0.6
        upd = [k * err for k in Kvec]
        for i in range(self.K):
            self.w1[i] += eta * upd[i]
            self.w2[i] += eta * upd[self.K + i]
        for i in range(len(phi)):
            for j in range(len(phi)):
                self.P[i][j] = (self.P[i][j] - Kvec[i] * Pf[j]) / self.lam
        merged = soft_threshold_l1(self.w1 + self.w2, self.l1_budget)
        self.w1 = merged[:self.K]
        self.w2 = merged[self.K:]


# ------------------------ RRF: Reduced-Rank / Ridge ------------------------- #

class RRF:
    """
    Ridge on multi-lag features, updated with realized labels.
    Features: OFI lags, |OFI| lags, QI, micro pressure, spread, MO asymmetry.
    """
    def __init__(self, K_ofi: int = 10, lam_ridge: float = 1e-2, alpha_cov: float = 0.1):
        self.K = K_ofi
        self.lam = lam_ridge
        self.alpha = alpha_cov
        D = 2*self.K + 4
        self.D = D
        self.Czz = [[0.0]*D for _ in range(D)]
        self.Czy = [0.0]*D
        self.w = [0.0]*D
        self.buf_ofi: Deque[float] = deque(maxlen=self.K)
        self.qi = 0.0
        self.micro_p = 0.0
        self.spread = 0.01
        self.mo_buy = RollingCount(5.0)
        self.mo_sell = RollingCount(5.0)

    def push_context(self, qi: float, micro_p: float, spread: float):
        self.qi = qi
        self.micro_p = micro_p
        self.spread = spread

    def push_ofi(self, ofi: float, ts: float, side: Optional[str] = None):
        self.buf_ofi.appendleft(ofi)
        if side == "buy":
            self.mo_buy.add(ts, 1)
        elif side == "sell":
            self.mo_sell.add(ts, 1)

    def _feature(self, ts: float) -> List[float]:
        x = list(self.buf_ofi) + [0.0]*(self.K - len(self.buf_ofi))
        xa = [abs(v) for v in x]
        mo_b = self.mo_buy.value(ts)
        mo_s = self.mo_sell.value(ts)
        mo_asym = (mo_b - mo_s) / (mo_b + mo_s) if mo_b + mo_s > 0 else 0.0
        return x + xa + [self.qi, self.micro_p, self.spread, mo_asym]

    def predict_drift_ticks(self, ts: float, tick: float) -> float:  # tick unused; kept for compatibility
        z = self._feature(ts)
        return sum(wi * zi for wi, zi in zip(self.w, z))

    def update(self, ts_label: float, realized_ret_ticks: float):
        z = self._feature(ts_label)
        a = self.alpha
        for i in range(self.D):
            for j in range(self.D):
                self.Czz[i][j] = (1-a)*self.Czz[i][j] + a*(z[i]*z[j])
        for i in range(self.D):
            self.Czy[i] = (1-a)*self.Czy[i] + a*(z[i]*realized_ret_ticks)
        C = [[self.Czz[i][j] for j in range(self.D)] for i in range(self.D)]
        for i in range(self.D):
            C[i][i] += self.lam
        b = self.Czy[:]
        for _ in range(5):
            for i in range(self.D):
                s = b[i] - sum(C[i][j]*self.w[j] for j in range(self.D) if j != i)
                self.w[i] = s / max(1e-9, C[i][i])


# ---------------------------- Fusion & Layers ------------------------------- #

def pool_adjacent_violators(p10: float, p30: float, p60: float) -> Tuple[float, float, float]:
    arr = [p10, p30, p60]
    if arr[0] > arr[1]:
        m = 0.5 * (arr[0] + arr[1]); arr[0] = arr[1] = m
    if arr[1] > arr[2]:
        m = 0.5 * (arr[1] + arr[2]); arr[1] = arr[2] = m
    if arr[0] > arr[1]:
        m = 0.5 * (arr[0] + arr[1]); arr[0] = arr[1] = m
    return tuple(clamp(v, 0.0, 1.0) for v in arr)

@dataclass
class FusionState:
    cov: List[List[float]] = field(default_factory=lambda: [[0.0]*4 for _ in range(4)])
    mean: List[float] = field(default_factory=lambda: [0.0]*4)
    alpha: float = 0.1  # faster adaptation

def precision_weighted_logodds(logits: List[float], fs: FusionState) -> float:
    K = len(logits)
    a = fs.alpha
    for i in range(K):
        fs.mean[i] = (1-a)*fs.mean[i] + a*logits[i]
    d = [logits[i] - fs.mean[i] for i in range(K)]
    for i in range(K):
        for j in range(K):
            fs.cov[i][j] = (1-a)*fs.cov[i][j] + a*(d[i]*d[j])
    C = [[fs.cov[i][j] for j in range(K)] for i in range(K)]
    for i in range(K):
        C[i][i] += 1e-3
    x = [0.0]*K
    for _ in range(40):
        for i in range(K):
            s = 1.0 - sum(C[i][j]*x[j] for j in range(K) if j != i)
            x[i] = s / max(1e-9, C[i][i])
    num = sum(x[i]*logits[i] for i in range(K))
    den = sum(x)
    return (num/den) if abs(den) > 1e-9 else statistics.fmean(logits)

def brownian_barrier_prob(mu: float, sigma: float, A: float) -> float:
    if sigma <= 1e-9:
        return 0.5 if abs(mu) < 1e-9 else (1.0 if mu > 0 else 0.0)
    arg = 2.0 * mu * A / (sigma * sigma + 1e-12)
    return sigmoid(arg)

def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1-1e-6)
    return math.log(p/(1.0-p))

def inv_logit(l: float) -> float:
    return sigmoid(l)


class ReliabilityCalibrator:
    """20-bin frequency calibrator with linear interpolation."""
    def __init__(self, bins: int = 20, warm: int = 150):
        self.bins = bins
        self.warm = warm
        self.counts = [0]*bins
        self.wins = [0]*bins
        self.samples = 0
    def _idx(self, p: float) -> int:
        p = clamp(p, 1e-6, 1-1e-6)
        return min(self.bins-1, int(p * self.bins))
    def add(self, p: float, outcome_up: int):
        i = self._idx(p)
        self.counts[i] += 1
        self.wins[i] += 1 if outcome_up else 0
        self.samples += 1
    def calibrate(self, p: float) -> float:
        if self.samples < self.warm:
            return p
        xs, ys = [], []
        for i in range(self.bins):
            n = self.counts[i]
            if n > 0:
                xs.append((i+0.5)/self.bins)
                ys.append(self.wins[i]/n)
        if len(xs) < 2:
            return p
        # linear interpolation among available bin midpoints
        for j in range(1, len(xs)):
            if p <= xs[j]:
                x0,x1,y0,y1 = xs[j-1],xs[j],ys[j-1],ys[j]
                t = (p-x0)/max(1e-9,(x1-x0))
                return clamp(y0 + t*(y1-y0), 0.0, 1.0)
        return ys[-1]


# ------------------------------ Main Engine -------------------------------- #

@dataclass
class EngineConfig:
    symbol: str = "HYPE-PERP"
    tick: float = 0.001
    fee_ticks: float = 0.25
    extra_slip_ticks: float = 0.5
    ofi_window_s: float = 5.0
    horizons_s: Tuple[float, float, float] = (10.0, 30.0, 60.0)
    csv_path: str = "hype_predictions.csv"
    warmup_s: float = 300.0  # 5 minutes
    log_every_ms: int = 150
    hawkes: HawkesParams = field(default_factory=HawkesParams)
    lvp_K: int = 10
    lvp_forgetting: float = 0.95
    lvp_l1_budget: float = 5.0
    rrf_K: int = 10
    rrf_ridge: float = 1e-2
    cov_alpha: float = 0.1
    robust_win: int = 100
    winsor_lo: float = -0.02
    winsor_hi: float = 0.02
    conformal_alpha: float = 0.15


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.ob = OrderBookState(tick=cfg.tick)
        self.ofi = RollingOFI(cfg.ofi_window_s)
        self.vol = RollingVol(alpha=0.05)
        self.hcqr = HCQR(cfg.hawkes, step_s=0.5, horizon_s=cfg.horizons_s[1])
        self.lvp = LVP(K=cfg.lvp_K, lam=cfg.lvp_forgetting, l1_budget=cfg.lvp_l1_budget, tick=cfg.tick)
        self.rrf = RRF(K_ofi=cfg.rrf_K, lam_ridge=cfg.rrf_ridge, alpha_cov=0.1)
        self.fusion_state = FusionState(alpha=cfg.cov_alpha)
        self.label_buf = LabelBuffer(horizon_s=cfg.horizons_s[1])

        self.last_emit_real_ms = 0  # throttle by wall clock
        self.start_time = now_s()
        self.last_fused_p = 0.5
        self.trades_processed_since_summary = 0
        self.csv_rows_written = 0
        self.robust_window: Deque[float] = deque(maxlen=cfg.robust_win)
        self.calib_scores: Deque[float] = deque(maxlen=300)
        self.drift_abs: Deque[float] = deque(maxlen=300)
        self.reliab = ReliabilityCalibrator(bins=20, warm=150)
        self.usd_norm = EWMANorm(alpha=0.02, init=1000.0)  # typical USD trade size

        self._summary_started = False

        # CSV
        self.csv_file = open(cfg.csv_path, "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self._write_header()

    def _write_header(self):
        self.csv.writerow([
            "wall_time_iso","ts","symbol","mid","spread","qib1","micro_pressure",
            "ofi_w","mo_buys","mo_sells",
            "p_hcqr","p_lvp","p_rrf","p_barrier",
            "p10","p30","p60","p_fused","p_fused_cal","hz_chosen",
            "gate_exec_ok","gate_robust_ok",
            "expected_move_ticks","est_cost_ticks",
            "realized_ret_30s","realized_up_30s"
        ])
        self.csv_file.flush()

    # ---------------------------- Event Handling ---------------------------- #

    def on_book(self, ts: float, bids: List[List[float]], asks: List[List[float]]):
        self.ob.update_from_book(bids, asks)
        self.vol.update(self.ob.mid)
        self.label_buf.add_mid(ts, self.ob.mid)

        # Realize labels on the SAME (exchange) clock as features
        lab = self.label_buf.realized_return(ts)
        if lab is not None:
            _, r = lab
            self.robust_window.append(winsorize(r, self.cfg.winsor_lo, self.cfg.winsor_hi))
            price = self.ob.mid
            tick = self.ob.tick
            if price > 0 and tick > 0:
                realized_ticks = r * (price / tick)
                self.lvp.update(realized_ticks)
                self.rrf.update(ts, realized_ticks)

    def on_trade(self, ts: float, price: float, size: float, side: Optional[str]):
        self.trades_processed_since_summary += 1
        if side is None:
            side = "buy" if price >= self.ob.microprice else "sell"

        # Rolling USD notional normalization + winsorization to keep OFI well-scaled
        notional = max(0.0, size * price)
        self.usd_norm.update(notional)
        scale = max(10.0, self.usd_norm.v)  # floor to avoid tiny denominators
        signed = (size if side == "buy" else -size) * price / scale
        signed = winsorize(signed, -0.5, 0.5)

        self.ofi.add(ts, signed)
        self.hcqr.update_event(ts, side)
        self.lvp.push_ofi(signed)
        self.rrf.push_ofi(signed, ts, side=side)

    # --------------------------- Prediction Stack --------------------------- #

    def _barrier_prob(self, drift_ticks: float, sigma: float) -> float:
        A = 1.0  # ticks
        price = self.ob.mid
        tick = self.ob.tick
        if price <= 0 or tick <= 0:
            return 0.5
        sig_ticks = (price / tick) * sigma
        mu = drift_ticks
        return brownian_barrier_prob(mu, max(1e-6, sig_ticks), A)

    def _exec_cost_ticks(self) -> float:
        spread_ticks = max(1.0, self.ob.spread / self.ob.tick) if self.ob.tick > 0 else 1.0
        thin_penalty = 0.5 if (self.ob.best_bid_sz + self.ob.best_ask_sz) <= 0 else 0.0
        return self.cfg.fee_ticks + self.cfg.extra_slip_ticks + 0.5*max(0.0, spread_ticks-1.0) + thin_penalty

    def _robust_edge_ok(self) -> bool:
        if len(self.robust_window) < max(20, int(self.cfg.robust_win*0.2)):
            return True
        m = statistics.fmean(self.robust_window)
        price = self.ob.mid
        tick = self.ob.tick
        if price <= 0 or tick <= 0:
            return False
        return (m > (self._exec_cost_ticks() * (tick / price)) / 3.0)

    def _conformal_ok(self, p_fused: float) -> bool:
        # No trades during warmup
        if now_s() - self.start_time < self.cfg.warmup_s:
            return False
        if len(self.calib_scores) < 50:
            return p_fused >= 0.55
        al = clamp(self.cfg.conformal_alpha, 0.05, 0.3)
        q_idx = int((1.0 - al) * (len(self.calib_scores)-1))
        thresh = sorted(self.calib_scores)[q_idx]
        score_now = max(p_fused, 1.0 - p_fused)
        return score_now >= thresh

    def predict_all(self, ts: float) -> Dict[str, Any]:
        p_hcqr = self.hcqr.prob_up_within_T()
        drift_lvp = self.lvp.predict_drift_ticks()
        drift_rrf = self.rrf.predict_drift_ticks(ts, self.ob.tick)

        self.drift_abs.append(abs(drift_lvp)); self.drift_abs.append(abs(drift_rrf))
        scale = percentile_abs(self.drift_abs, 0.90)
        temperature = 2.0  # tune 1.5–3.0 as needed
        p_lvp = sigmoid(drift_lvp / (scale * temperature))
        p_rrf = sigmoid(drift_rrf / (scale * temperature))

        sigma = self.vol.sigma()
        p_barrier = self._barrier_prob(0.5 * drift_lvp + 0.5*drift_rrf, sigma)

        logits = [logit(p_hcqr), logit(p_lvp), logit(p_rrf), logit(p_barrier)]
        l_star = precision_weighted_logodds(logits, self.fusion_state)
        p_fused = inv_logit(l_star)
        p_fused_cal = self.reliab.calibrate(p_fused)

        (h10, h30, h60) = self.cfg.horizons_s
        p10_raw = inv_logit(l_star * (h10/h30))
        p30_raw = p_fused
        p60_raw = inv_logit(l_star * (h60/h30))
        p10, p30, p60 = pool_adjacent_violators(p10_raw, p30_raw, p60_raw)

        est_move30_ticks = (0.5 * drift_lvp + 0.5*drift_rrf)
        est_cost_ticks = self._exec_cost_ticks()
        chosen = 30

        return dict(
            p_hcqr=p_hcqr, p_lvp=p_lvp, p_rrf=p_rrf, p_barrier=p_barrier,
            p10=p10, p30=p30, p60=p60, p_fused=p_fused, p_fused_cal=p_fused_cal,
            est_move_ticks=est_move30_ticks, est_cost_ticks=est_cost_ticks, hz_chosen=chosen
        )

    # --------------------------- Logging & Summary -------------------------- #

    def maybe_emit_csv(self, msg_ts: float, extras: Dict[str, Any]):
        # Throttle by wall clock (IO control)
        now_ms = int(time.time() * 1000)
        if now_ms - self.last_emit_real_ms < self.cfg.log_every_ms:
            return
        self.last_emit_real_ms = now_ms

        exec_ok = self._conformal_ok(extras["p_fused_cal"])
        robust_ok = self._robust_edge_ok()

        # Realize labels on exchange clock
        lab = self.label_buf.realized_return(msg_ts)
        realized_r, realized_up = None, None
        if lab:
            _, r = lab
            realized_r, realized_up = r, 1 if r > 0 else 0
            # correctness score using calibrated prob
            p_correct = extras["p_fused_cal"] if realized_up else (1.0 - extras["p_fused_cal"])
            self.calib_scores.append(p_correct)
            # train calibrator on raw fused p vs outcome
            self.reliab.add(extras["p_fused"], realized_up)

        self.csv.writerow([
            iso_utc(), f"{msg_ts:.6f}", self.cfg.symbol, f"{self.ob.mid:.8f}", f"{self.ob.spread:.8f}",
            f"{self.ob.qib1:.6f}", f"{self.ob.micro_pressure:.6f}", f"{self.ofi.value(msg_ts):.4f}",
            self.rrf.mo_buy.value(msg_ts), self.rrf.mo_sell.value(msg_ts),
            f"{extras['p_hcqr']:.6f}", f"{extras['p_lvp']:.6f}", f"{extras['p_rrf']:.6f}",
            f"{extras['p_barrier']:.6f}", f"{extras['p10']:.6f}", f"{extras['p30']:.6f}",
            f"{extras['p60']:.6f}", f"{extras['p_fused']:.6f}", f"{extras['p_fused_cal']:.6f}", int(extras["hz_chosen"]),
            int(exec_ok), int(robust_ok), f"{extras['est_move_ticks']:.4f}",
            f"{extras['est_cost_ticks']:.4f}",
            "" if realized_r is None else f"{realized_r:.8f}",
            "" if realized_up is None else int(realized_up),
        ])
        self.csv_file.flush()
        self.csv_rows_written += 1

    async def run_summary_printer(self):
        await asyncio.sleep(10)
        while True:
            await asyncio.sleep(30)
            uptime_m = (now_s() - self.start_time) / 60.0
            mid = self.ob.mid
            spread_ticks = self.ob.spread / self.ob.tick if self.ob.tick > 0 else 0
            recent_returns = list(self.robust_window)[-20:]
            avg_ret = statistics.fmean(recent_returns) if recent_returns else 0.0
            win_rate = (sum(1 for r in recent_returns if r > 0) / len(recent_returns)) if recent_returns else 0.0

            print("\n" + "="*60)
            print(f"📊 TERMINAL SUMMARY @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Uptime: {uptime_m:.2f} minutes | Symbol: {self.cfg.symbol}")
            print(f"   CSV Rows Written: {self.csv_rows_written}")
            print("-"*60)
            print("📈 Market State:")
            print(f"   Mid Price : {mid:,.5f}")
            print(f"   Spread    : {spread_ticks:.2f} ticks")
            print(f"   Inst. Vol : {self.vol.sigma() * 100:.4f}% (log-ret)")
            print("-"*60)
            print("🧠 Prediction Signal:")
            print(f"   Last Fused P(up, cal) : {self.last_fused_p:.4f}")
            print(f"   Trades Last 30s       : {self.trades_processed_since_summary}")
            print("-"*60)
            print("📋 Performance (last ~20 realized 30s labels):")
            print(f"   Avg Realized Return: {avg_ret:+.6f}")
            print(f"   Win Rate (ret > 0) : {win_rate:.2%}")
            print("="*60 + "\n")

            self.trades_processed_since_summary = 0

    # ------------------------------ Run Loop -------------------------------- #

    async def run_hyperliquid_ws(self):
        if not HAS_WS:
            print("ERROR: websockets is not installed. Run: pip install websockets", file=sys.stderr)
            return

        url = "wss://api.hyperliquid.xyz/ws"
        coin = self.cfg.symbol.split('-')[0]

        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:  # type: ignore
            print(f"Connecting to Hyperliquid for {coin}...")
            await ws.send(json.dumps({
                "method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}
            }))
            await ws.send(json.dumps({
                "method": "subscribe", "subscription": {"type": "trades", "coin": coin}
            }))
            print("Subscribed. Logging to:", self.cfg.csv_path)

            # start summary printer once
            if not self._summary_started:
                self._summary_started = True
                asyncio.create_task(self.run_summary_printer())

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if "channel" in msg:
                    channel, data = msg.get("channel"), msg.get("data")
                    if not data:
                        continue

                    if channel == "l2Book":
                        ts = epoch_any_to_s((data.get("time")))
                        bids, asks = _parse_levels(data.get("levels") or [])
                        self._handle_book(ts, bids, asks)

                    elif channel == "trades":
                        for trade in data:
                            side_raw = str(trade.get("side", "")).upper()
                            side = "buy" if side_raw in ("B", "BUY") else ("sell" if side_raw in ("S", "SELL") else None)
                            ts = epoch_any_to_s(trade.get("time"))
                            price = float(trade.get('px', 0.0))
                            size = float(trade.get('sz', 0.0))
                            self._handle_trade(ts, price, size, side)

                elif "ping" in msg:
                    await ws.send(json.dumps({"pong": msg["ping"]}))

    def _handle_book(self, ts: float, bids: List[List[float]], asks: List[List[float]]):
        self.on_book(ts, bids, asks)
        if self.ob.mid > 0 and self.ob.spread > 0:
            extras = self.predict_all(ts)
            # track calibrated fused p for the summary
            self.last_fused_p = extras["p_fused_cal"]
            self.rrf.push_context(self.ob.qib1, self.ob.micro_pressure, self.ob.spread)
            self.maybe_emit_csv(ts, extras)

    def _handle_trade(self, ts: float, price: float, size: float, side: Optional[str]):
        self.on_trade(ts, price, size, side)
        if self.ob.mid > 0 and self.ob.spread > 0:
            extras = self.predict_all(ts)
            self.last_fused_p = extras["p_fused_cal"]
            self.rrf.push_context(self.ob.qib1, self.ob.micro_pressure, self.ob.spread)
            self.maybe_emit_csv(ts, extras)

    def close(self):
        try:
            self.csv_file.flush()
        finally:
            self.csv_file.close()
            print(f"CSV file saved to {self.cfg.csv_path}")


# --------------------------------- Main ------------------------------------- #

async def main_async():
    cfg = EngineConfig()
    eng = Engine(cfg)

    print("\nStarting HYPE-PERP Price Predictor...")
    print(f"Symbol: {cfg.symbol} | Tick Size: {cfg.tick}")
    print(f"Output CSV: {cfg.csv_path}")

    try:
        await eng.run_hyperliquid_ws()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down.")
    finally:
        eng.close()

def main():
    if not HAS_WS:
        print("ERROR: The 'websockets' library is not installed.", file=sys.stderr)
        print("Please run: pip install websockets", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
