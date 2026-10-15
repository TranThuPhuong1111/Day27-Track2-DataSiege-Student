# Reflection (≤1 page)

**Results (final thresholds):** practice 48.97 (TPR 1.00, FPR 0.034) · public 46.45 (TPR 1.00, FPR 0.058, 9% cost overage on public's 220-credit budget) · **private 33.23 (TPR 0.685, FPR 0.034, no overage)**.

A first version with thresholds at 2.2–2.5σ scored 32.51 on private (TPR 0.667, FPR 0.027). Lowering the noisy metrics to 2.0σ gained ~1 caught fault for ~1 extra false alarm (+0.7 points), and slightly lowered practice/public, where nothing was being missed.

**Which fault types were hardest to catch, and why?**

- *Easy, deterministic:* contract `schema_hash_mismatch` / `type_violation`, lineage `missing_upstream` (actual upstream is missing `raw.customers`) and `orphan_output` (`actual_downstream_count == 0`). These have no overlap with clean data, so I alert on them directly. Freshness above the declared SLA is the same kind of rule.
- *Easy, but only once you look at the right scale:* feature skew. The published bound (0.41σ) is drawn on |serve − train| / train_std. Clean serving means stay under ~0.5 on that scale, and even the "subtle" skews were at 1.8–2.9, so a 0.5 threshold separates them cleanly.
- *Hardest:* subtle `data_batch` faults (mean_amount shift, small volume or null-rate changes, staleness) and subtle embedding faults (centroid shift ~0.035–0.045, doc age ~45–50 days). These land at 2–3σ, right in the tail of clean variance. In public, my borderline data_batch alerts were right only about 1 time in 5. On private, the TPR of 0.685 means about a third of faults sat even closer to normal than my thresholds. A single-metric threshold can't separate a 2σ fault from a 2σ clean batch, and each event exposes only 1–5 numbers, so there is little else to combine.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

- Cost was never the binding constraint on private: one call per event used ~300 of 320 credits, and calling the same tool twice returns the same deterministic values, so extra credits buy no extra information. The real tradeoff is **threshold placement**, not call budget.
- The scoring makes a miss cost ~5× more than a false alarm (0.5/n_faulty vs 0.3/n_clean with ~25% faulty), so the break-even precision for an alert is only ~17%. That is why thresholds sit at 2.0–2.5σ instead of the baseline's 3σ. Pushing the noisy metrics (mean_amount, row_count, staleness, centroid shift, doc age) from 2.2–2.5σ down to 2.0σ helped only marginally. So most of the ~1/3 of private faults I still miss are not just under my thresholds. They look clean on the single metric I read, and lowering thresholds further would mostly add false alarms.
- I would replace the "max single z / two soft signals" rule with a proper per-event likelihood-ratio score (sum of one-sided log-LRs across a batch's metrics), alerting when the posterior odds exceed ~0.2. That is the Bayes-optimal cutoff for this scoring function.
- The online median I use for the clean center helped calibration (public's clean mean_amount sat ~1.3 above the baseline). I would extend it to estimate σ online as well (robust MAD) rather than trusting the baseline σ alone.
