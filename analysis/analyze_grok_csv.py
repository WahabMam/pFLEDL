#!/usr/bin/env python3
"""Grok-condition analysis for FedGrok CSV logs.

FROZEN ANALYSIS CONVENTIONS -- chosen 2026-07-21, BEFORE running or looking at any
FedSOLAR (intervention) results on non-IID MNIST or any CIFAR-scale config. These
constants (MEM_FRACTION, GEN_FRACTION, the plateau/flatness/contraction thresholds)
replace the old fixed 95%/90% absolute-accuracy bars, which broke down whenever a
config's achievable ceiling (e.g. MNIST c100 at 10 samples/client, or any CIFAR/
TinyImageNet run) sits below 90% for reasons unrelated to grokking.

Do NOT retune these per-dataset, per-partition-scheme, or per-algorithm after seeing
which configs they favor -- that reintroduces the exact after-the-look, results-
favorable bias this rewrite exists to remove. If a convention needs to change, change
it once here, re-run it on ALL existing baseline data, and document why in the task
log BEFORE generating any new intervention (FedSOLAR-vs-baseline) comparison.
"""
import argparse
import csv
import json
from pathlib import Path

# --- frozen conventions -----------------------------------------------------
MEM_FRACTION = 0.95            # memorization = train_acc reaches 95% of its OWN plateau
GEN_FRACTION = 0.5             # generalization = test_acc reaches the midpoint between
                                # its own starting value and its own plateau (same point
                                # a sigmoid fit's t0 would target -- keeps primary and
                                # validator metrics conceptually aligned)
PLATEAU_WINDOW_ROUNDS = 300    # round-span averaged for ceiling/floor (was a fixed
                                # 6-eval-point window, silently assuming eg=50 for every
                                # run in this project's history until 2026-07-26, when the
                                # v08 addendum first varied eval_gap per arm and this
                                # assumption broke: at eg=5 a 6-point window covers only
                                # 30 rounds of plateau, dominated by eval noise, causing
                                # spurious "censored" verdicts on runs that had plainly
                                # already converged. Fixed to a round-span, with the point
                                # count derived from each CSV's own inferred eval_gap, so
                                # the flatness/plateau reading no longer silently depends
                                # on which eval_gap a run happened to use.
SLOPE_WINDOW_ROUNDS = 600      # round-span used to check "has this flattened" (was 12
                                # eval points, same eg=50 assumption as above)
FLAT_SLOPE_EPS = 0.005         # |slope| below this (per 1000 rounds) counts as flattened
TRAIN_PLATEAU_MIN = 0.5        # below this, the client never memorized at all (collapse,
                                # not "no grok")
PLATEAU_ABOVE_CHANCE = 2.0     # final_test_acc must be >= this many times chance level
PLATEAU_ABOVE_INITIAL = 0.05   # ... and >= this many points above its own starting test_acc
NORM_CONTRACTION_THRESHOLD = 0.7  # weight_norm must shrink to <=70% of its value at
                                   # memorization -- the mechanistic gate (Omnigrok)


def first_sustained(rows, key, threshold, patience):
    for start in range(len(rows) - patience + 1):
        window = rows[start : start + patience]
        if all(float(row[key]) >= threshold for row in window):
            return int(float(rows[start]["round"]))
    return None


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def infer_eval_gap(rows):
    """Median round-delta between consecutive evals -- robust to an odd first/last row."""
    rounds = sorted({int(float(r["round"])) for r in rows})
    if len(rounds) < 2:
        return 50  # fall back to the project's historical default
    deltas = [b - a for a, b in zip(rounds, rounds[1:])]
    return median(deltas)


def linear_slope_per_1000(rows, key, window):
    window = min(window, len(rows))
    sub = rows[-window:]
    xs = [int(float(r["round"])) for r in sub]
    ys = [float(r[key]) for r in sub]
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope * 1000.0


def classify_run(rows, num_classes, patience):
    """Primary metric: plateau-relative thresholds + three guardrails + censoring.

    Gate order matters: flatness (can we trust the plateau reading at all) is
    checked first, then whether training/test plateaus are meaningful, then
    timing + the norm-contraction mechanism gate.
    """
    eval_gap = infer_eval_gap(rows)
    plateau_window = max(2, round(PLATEAU_WINDOW_ROUNDS / eval_gap))
    slope_window = max(2, round(SLOPE_WINDOW_ROUNDS / eval_gap))

    train_ceil = median([float(r["train_acc"]) for r in rows[-plateau_window:]])
    test_ceil = median([float(r["test_acc"]) for r in rows[-plateau_window:]])
    test_floor = float(rows[0]["test_acc"])

    train_slope = linear_slope_per_1000(rows, "train_acc", slope_window)
    test_slope = linear_slope_per_1000(rows, "test_acc", slope_window)
    train_flat = abs(train_slope) < FLAT_SLOPE_EPS
    test_flat = abs(test_slope) < FLAT_SLOPE_EPS

    base = {
        "eval_gap": eval_gap,
        "plateau_window_evals": plateau_window,
        "slope_window_evals": slope_window,
        "train_ceil": train_ceil,
        "test_ceil": test_ceil,
        "test_floor": test_floor,
        "train_slope_per_1000": train_slope,
        "test_slope_per_1000": test_slope,
    }

    if not (train_flat and test_flat):
        return {
            **base,
            "verdict": "censored",
            "reason": "train and/or test curve still trending at the last eval; "
                      "run more rounds before trusting a plateau reading",
        }

    if train_ceil < TRAIN_PLATEAU_MIN:
        return {**base, "verdict": "failed_to_memorize"}

    chance = 1.0 / num_classes
    if test_ceil < chance * PLATEAU_ABOVE_CHANCE or (test_ceil - test_floor) < PLATEAU_ABOVE_INITIAL:
        return {**base, "verdict": "collapse_no_generalization", "chance": chance}

    mem_bar = MEM_FRACTION * train_ceil
    gen_bar = test_floor + GEN_FRACTION * (test_ceil - test_floor)
    t_mem = first_sustained(rows, "train_acc", mem_bar, patience)
    t_gen = first_sustained(rows, "test_acc", gen_bar, patience)
    base.update({"mem_bar": mem_bar, "gen_bar": gen_bar})

    if t_mem is None or t_gen is None:
        return {
            **base,
            "verdict": "censored",
            "reason": "plateau looked flat but sustained-crossing was not found; inconclusive",
        }

    by_round = {int(float(r["round"])): r for r in rows}
    norm_at_mem = float(by_round[t_mem]["weight_norm"])
    norm_at_gen = float(by_round[t_gen]["weight_norm"])
    norm_final = float(rows[-1]["weight_norm"])
    contraction_ratio = (norm_at_gen / norm_at_mem) if norm_at_mem else None
    mechanism_fired = contraction_ratio is not None and contraction_ratio <= NORM_CONTRACTION_THRESHOLD

    delay = t_gen - t_mem
    delay_normalized = (delay / t_mem) if t_mem > 0 else None
    positive_delay = delay > 0

    if mechanism_fired and positive_delay:
        verdict = "real_grok"
    elif not mechanism_fired and positive_delay:
        verdict = "smooth_colearning_not_grok"
    elif mechanism_fired and not positive_delay:
        verdict = "grok_trivial_delay"
    else:
        verdict = "ambiguous"

    return {
        **base,
        "verdict": verdict,
        "t_mem": t_mem,
        "t_gen": t_gen,
        "delay": delay,
        "delay_normalized": delay_normalized,
        "norm_at_mem": norm_at_mem,
        "norm_at_gen": norm_at_gen,
        "norm_final": norm_final,
        "contraction_ratio": contraction_ratio,
        "mechanism_fired": mechanism_fired,
    }


def sigmoid_fit(rows):
    """Secondary validator (demoted from primary): fits floor/ceil/t0/k to test_acc(t).
    t0 should land close to primary's t_gen if the transition is real and clean;
    a low R^2 means the curve is too noisy/steppy to trust this fit at all.
    """
    try:
        import numpy as np
        from scipy.optimize import curve_fit

        xs = np.array([int(float(r["round"])) for r in rows], dtype=float)
        ys = np.array([float(r["test_acc"]) for r in rows], dtype=float)

        def sigmoid(x, floor, ceil, t0, k):
            return floor + (ceil - floor) / (1 + np.exp(-k * (x - t0)))

        p0 = [ys.min(), ys.max(), xs[len(xs) // 2], 4.0 / (xs[-1] - xs[0] + 1e-9)]
        bounds = ([-0.5, -0.5, xs[0] - 1, 0.0], [1.5, 1.5, xs[-1] + 1, 1.0])
        popt, _ = curve_fit(sigmoid, xs, ys, p0=p0, bounds=bounds, maxfev=20000)
        floor, ceil, t0, k = (float(v) for v in popt)
        pred = sigmoid(xs, *popt)
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
        return {"fit_ok": True, "floor": floor, "ceil": ceil, "t0": t0, "k": k, "r2": r2}
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostic, never fatal
        return {"fit_ok": False, "error": str(exc)}


def summarize(path, patience=3, num_classes=10):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no data rows in {path}")

    # legacy absolute-threshold metrics, kept for continuity/comparison only --
    # NOT the primary verdict anymore (see module docstring).
    t_mem95 = first_sustained(rows, "train_acc", 0.95, patience)
    t_mem99 = first_sustained(rows, "train_acc", 0.99, patience)
    t_gen90 = first_sustained(rows, "test_acc", 0.90, patience)
    by_round = {int(float(row["round"])): row for row in rows}

    return {
        "path": str(path),
        "evaluations": len(rows),
        "last_round": int(float(rows[-1]["round"])),
        "num_classes": num_classes,
        "primary": classify_run(rows, num_classes, patience),
        "sigmoid_validator": sigmoid_fit(rows),
        "legacy_absolute": {
            "t_mem95": t_mem95,
            "t_mem99": t_mem99,
            "t_gen90": t_gen90,
            "delay_from_mem95": None if t_mem95 is None or t_gen90 is None else t_gen90 - t_mem95,
            "delay_from_mem99": None if t_mem99 is None or t_gen90 is None else t_gen90 - t_mem99,
            "test_acc_at_mem95": None if t_mem95 is None else float(by_round[t_mem95]["test_acc"]),
            "norm_at_mem95": None if t_mem95 is None else float(by_round[t_mem95]["weight_norm"]),
            "norm_at_gen90": None if t_gen90 is None else float(by_round[t_gen90]["weight_norm"]),
        },
        "best_test_acc": max(float(row["test_acc"]) for row in rows),
        "final_test_acc": float(rows[-1]["test_acc"]),
        "final_train_acc": float(rows[-1]["train_acc"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10,
                        help="chance-level denominator for the plateau-above-chance gate "
                             "(10 for MNIST/CIFAR10, 100 for CIFAR100, 200 for TinyImageNet)")
    args = parser.parse_args()
    for path in args.csv:
        print(json.dumps(summarize(path, args.patience, args.num_classes), sort_keys=True))


if __name__ == "__main__":
    main()
