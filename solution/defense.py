"""
Data Siege defense.

Strategy
--------
* Exactly one metered call per event (the cheapest one that carries the
  signal). Full coverage costs ~1.5 credits/event on average, which fits the
  private budget (200 events -> ~300 of 320 credits), so no event is skipped.
* Deterministic faults (contract violations, missing upstream edges, orphaned
  outputs) are alerted on directly — they have no clean-variance overlap.
* Continuous metrics are turned into z-scores. sigma comes from ctx.baseline
  (each bound is published as mean +/- 3 sigma of clean data). The center
  starts at the baseline-implied mean and is then tracked online with a
  robust median of values we did NOT alert on, so a slightly shifted clean
  distribution doesn't miscalibrate the thresholds.
* Thresholds are deliberately below 3 sigma: with ~25% faulty events, one
  missed fault costs ~0.5/n_faulty while one false alarm costs ~0.3/n_clean,
  i.e. a miss is ~5x more expensive than a false alarm. A second rule
  combines several moderately elevated signals on the same event, which
  catches subtle faults no single threshold would.
"""
from api import Verdict


# Clean-stream centers for the metrics whose baseline is one-sided (only a
# max is published). Estimated from clean practice measurements; the online
# median replaces them once enough clean history has been seen.
_ONE_SIDED_PRIOR_MEAN = {
    "null_rate": 0.0045,
    "staleness": 4.1,
    "freshness": 5.1,
    "lineage_ms": 4150.0,
    "centroid_shift": 0.020,
    "doc_age": 26.4,
}

# z thresholds (single-signal) per metric, in the fault direction.
_Z_ALERT = {
    "row_count": 2.0,
    "null_rate": 2.4,
    "mean_amount": 2.0,
    "std_amount": 3.0,
    "staleness": 2.0,
    "freshness": 2.3,
    "lineage_ms": 2.5,
    "centroid_shift": 2.0,
    "doc_age": 2.0,
}

# Combined rule: this many signals each at/above _Z_SOFT -> alert.
_Z_SOFT = 1.6
_MIN_HISTORY = 8
_HISTORY_CAP = 60

# feature_drift returns |serve - train| / train_std. Clean serving means sit
# within ~0.17 of that unit (half-normal, 3 sigma ~= 0.5); the published
# 0.41 bound is drawn on the abs value. Any skew fault we care about is far
# above this.
_FEATURE_SHIFT_ALERT = 0.5

_EXPECTED_UPSTREAM = {"raw.orders", "raw.customers"}


def register(ctx):
    b = ctx.baseline
    params = {}

    def two_sided(name, lo, hi):
        params[name] = ((lo + hi) / 2.0, (hi - lo) / 6.0)

    def one_sided(name, hi):
        mean = _ONE_SIDED_PRIOR_MEAN[name]
        params[name] = (mean, max((hi - mean) / 3.0, 1e-9))

    two_sided("row_count", b.get("row_count_min", 435.47), b.get("row_count_max", 561.29))
    two_sided("mean_amount", b.get("mean_amount_min", 72.76), b.get("mean_amount_max", 90.61))
    one_sided("null_rate", b.get("null_rate_max", 0.0109))
    one_sided("staleness", b.get("staleness_min_max", 8.418))
    one_sided("freshness", b.get("freshness_delay_max_min", 11.11))
    one_sided("lineage_ms", b.get("lineage_duration_ms_max", 5134.98))
    one_sided("centroid_shift", b.get("embedding_centroid_shift_max", 0.0435))
    one_sided("doc_age", b.get("corpus_avg_doc_age_days_max", 49.80))
    # std_amount has no published bound; clean batches sit around 14.8 +/- 1.2.
    params["std_amount"] = (14.8, 1.2)

    ctx.state["params"] = params
    ctx.state["history"] = {k: [] for k in params}

    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------

def _median(xs):
    # statistics.median pulls in modules the sandbox's import allowlist blocks.
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _z(ctx, name, value):
    """Signed z-score of value against the clean distribution for `name`."""
    mean, sigma = ctx.state["params"][name]
    hist = ctx.state["history"][name]
    if len(hist) >= _MIN_HISTORY:
        mean = _median(hist)
    return (value - mean) / sigma


def _remember(ctx, values):
    """Feed values from an event we judged clean into the online history."""
    for name, value in values.items():
        hist = ctx.state["history"][name]
        hist.append(value)
        if len(hist) > _HISTORY_CAP:
            del hist[0]


def _judge(ctx, signals, pillar, hard_reasons=()):
    """signals: {name: (value, z_in_fault_direction)}. Returns a Verdict."""
    reasons = list(hard_reasons)
    soft = []
    for name, (_, z) in signals.items():
        if z >= _Z_ALERT[name]:
            reasons.append(f"{name} z={z:.2f}")
        elif z >= _Z_SOFT:
            soft.append(f"{name} z={z:.2f}")
    if not reasons and len(soft) >= 2:
        reasons.append("combined: " + ", ".join(soft))

    if reasons:
        top = max([z for _, z in signals.values()] + [0.0])
        conf = 1.0 if hard_reasons else min(0.99, 0.5 + 0.1 * top)
        return Verdict(alert=True, confidence=conf, reason="; ".join(reasons), pillar=pillar)

    _remember(ctx, {name: v for name, (v, _) in signals.items()})
    return Verdict(alert=False, confidence=0.8, reason="within clean variance", pillar=pillar)


def _ok(resp):
    return isinstance(resp, dict) and "error" not in resp


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

def check_data_batch(payload, ctx):
    p = ctx.tools.batch_profile(payload["batch_id"])
    if not _ok(p):
        return Verdict(alert=False, confidence=0.0, reason=f"tool error: {p}", pillar="checks")

    rc = p["row_count"]
    nr = (p.get("null_rate") or {}).get("customer_id", 0.0)
    mean_amt = p["mean_amount"]
    std_amt = p["std_amount"]
    stale = p["staleness_min"]

    z_rc = _z(ctx, "row_count", rc)
    z_mean = _z(ctx, "mean_amount", mean_amt)
    z_std = _z(ctx, "std_amount", std_amt)
    signals = {
        "row_count": (rc, abs(z_rc)),            # spikes and drops
        "null_rate": (nr, _z(ctx, "null_rate", nr)),
        "mean_amount": (mean_amt, abs(z_mean)),  # shift in either direction
        "std_amount": (std_amt, abs(z_std)),
        "staleness": (stale, _z(ctx, "staleness", stale)),
    }
    return _judge(ctx, signals, "checks")


def check_contract_checkpoint(payload, ctx):
    d = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if not _ok(d):
        return Verdict(alert=False, confidence=0.0, reason=f"tool error: {d}", pillar="contracts")

    hard = [f"violation: {v}" for v in d.get("violations") or []]
    delay = d["freshness_delay_min"]
    declared = (payload.get("declared_sla") or {}).get("freshness_min")
    if declared is not None and delay > declared:
        hard.append(f"freshness {delay} > declared SLA {declared}")

    signals = {"freshness": (delay, _z(ctx, "freshness", delay))}
    if hard:
        return Verdict(alert=True, confidence=1.0, reason="; ".join(hard), pillar="contracts")
    return _judge(ctx, signals, "contracts")


def check_lineage_run(payload, ctx):
    g = ctx.tools.lineage_graph_slice(payload["run_id"])
    if not _ok(g):
        return Verdict(alert=False, confidence=0.0, reason=f"tool error: {g}", pillar="lineage")

    hard = []
    upstream = set(g.get("actual_upstream") or [])
    declared_inputs = {i.get("name") for i in payload.get("inputs") or []}
    missing = (_EXPECTED_UPSTREAM | declared_inputs) - upstream
    if missing:
        hard.append(f"missing upstream: {sorted(missing)}")

    downstream = g.get("actual_downstream_count", 0)
    if downstream < max(1, len(payload.get("outputs") or [])):
        hard.append(f"orphaned output (downstream_count={downstream})")

    dur = g["duration_ms"]
    signals = {"lineage_ms": (dur, _z(ctx, "lineage_ms", dur))}
    if hard:
        return Verdict(alert=True, confidence=1.0, reason="; ".join(hard), pillar="lineage")
    return _judge(ctx, signals, "lineage")


def check_feature_materialization(payload, ctx):
    f = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if not _ok(f):
        return Verdict(alert=False, confidence=0.0, reason=f"tool error: {f}", pillar="ai_infra")

    shift = f.get("mean_shift_sigma")
    if shift is None:
        shift = abs(f["serve_mean"] - f["train_mean"]) / max(f["train_std"], 1e-9)
    if shift >= _FEATURE_SHIFT_ALERT:
        return Verdict(alert=True, confidence=min(0.99, 0.6 + shift / 5),
                       reason=f"train/serve skew {shift:.3f} sigma", pillar="ai_infra")
    return Verdict(alert=False, confidence=0.8, reason=f"skew {shift:.3f} sigma", pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    e = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if not _ok(e):
        return Verdict(alert=False, confidence=0.0, reason=f"tool error: {e}", pillar="ai_infra")

    cs = e["centroid_shift"]
    age = e["avg_doc_age_days"]
    signals = {
        "centroid_shift": (cs, _z(ctx, "centroid_shift", cs)),
        "doc_age": (age, _z(ctx, "doc_age", age)),
    }
    return _judge(ctx, signals, "ai_infra")
