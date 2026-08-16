#!/usr/bin/env python3
"""Compare safety-eval metrics.json from base vs trained model runs."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Same keys as eval/parse_result_to_csv.py safety_metrics_to_extract
METRIC_SPECS: List[Tuple[str, str, str, bool]] = [
    # (benchmark, metric_key, display_name, lower_is_better)
    ("harmbench_precompute", "micro ASR (lower)", "HarmBench precompute ASR", True),
    ("harmbench", "micro ASR (lower)", "HarmBench ASR", True),
    ("xstest", "inverted_rta_all_safe", "XSTest inverted RTA safe", False),
    ("xstest", "rta_all_contrast", "XSTest RTA contrast", False),
    ("xstest", "overall_accuracy", "XSTest overall acc", False),
    ("xstest", "refusal_prf_f1", "XSTest refusal F1", False),
    ("or_bench:hard-1k", "compliance_rate_mean", "OR-bench hard compliance", False),
    ("or_bench:toxic", "refusal_rate_mean", "OR-bench toxic refusal", False),
    ("wildguardtest", "adversarial harm (lower)", "WG-test adv harm", True),
    ("wildguardtest", "vanilla harm (lower)", "WG-test vanilla harm", True),
    ("wildjailbreak:benign", "macro ASR", "WJB-benign macro ASR", True),
    ("wildjailbreak:harmful", "macro ASR", "WJB-harmful macro ASR", True),
    ("do_anything_now", "macro ASR", "DAN macro ASR", True),
]


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def get_metric(metrics: Dict[str, Any], benchmark: str, key: str) -> Any:
    if benchmark not in metrics:
        return None
    data = metrics[benchmark]
    if key == "refusal_prf_f1":
        return data.get("refusal_prf", {}).get("f1")
    return data.get(key)


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def improved(base: float, trained: float, lower_is_better: bool) -> str:
    if lower_is_better:
        return "better" if trained < base else ("worse" if trained > base else "same")
    return "better" if trained > base else ("worse" if trained < base else "same")


def compare(base_metrics: Dict[str, Any], trained_metrics: Dict[str, Any]) -> None:
    print(f"{'Metric':<28} {'Base':>10} {'Trained':>10} {'Delta':>10} {'vs base':>8}")
    print("-" * 70)
    for benchmark, key, name, lower_is_better in METRIC_SPECS:
        b = get_metric(base_metrics, benchmark, key)
        t = get_metric(trained_metrics, benchmark, key)
        if b is None and t is None:
            continue
        delta = None
        verdict = "—"
        if isinstance(b, (int, float)) and isinstance(t, (int, float)):
            delta = t - b
            verdict = improved(b, t, lower_is_better)
        delta_str = f"{delta:+.4f}" if delta is not None else "—"
        print(f"{name:<28} {fmt(b):>10} {fmt(t):>10} {delta_str:>10} {verdict:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base vs trained safety eval metrics")
    parser.add_argument("base_metrics", type=Path, help="Path to base model metrics.json")
    parser.add_argument("trained_metrics", type=Path, help="Path to trained model metrics.json")
    args = parser.parse_args()

    base = load_metrics(args.base_metrics)
    trained = load_metrics(args.trained_metrics)

    print("Safety comparison (trained vs base)")
    print(f"  Base:    {args.base_metrics}")
    print(f"  Trained: {args.trained_metrics}")
    print()
    print("For ASR / harm metrics: lower is safer. For accuracy/refusal: higher is better.")
    print()
    compare(base, trained)


if __name__ == "__main__":
    main()
