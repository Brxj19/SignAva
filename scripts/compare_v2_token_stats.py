import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_v2.utils import OUTPUT_V2_DIR


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing stats file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _value(stats: dict[str, Any], key: str) -> float:
    return float(stats.get(key, 0.0))


def compare_token_stats() -> dict[str, Any]:
    eval_dir = OUTPUT_V2_DIR / "evaluation"
    real_train = _load(eval_dir / "real_token_stats_train.json")
    real_test = _load(eval_dir / "real_token_stats_test.json")
    generated = _load(eval_dir / "generated_token_stats.json")

    report = {
        "real_train": {
            "average_unique_tokens": _value(real_train, "average_unique_tokens"),
            "average_transition_count": _value(real_train, "average_transition_count"),
            "average_repeated_token_ratio": _value(real_train, "average_repeated_token_ratio"),
        },
        "real_test": {
            "average_unique_tokens": _value(real_test, "average_unique_tokens"),
            "average_transition_count": _value(real_test, "average_transition_count"),
            "average_repeated_token_ratio": _value(real_test, "average_repeated_token_ratio"),
        },
        "generated": {
            "average_unique_tokens": _value(generated, "average_unique_tokens"),
            "average_transition_count": _value(generated, "average_transition_count"),
            "average_repeated_token_ratio": _value(generated, "average_repeated_token_ratio"),
        },
    }
    real_reference_unique = report["real_test"]["average_unique_tokens"] or report["real_train"]["average_unique_tokens"]
    real_reference_transitions = report["real_test"]["average_transition_count"] or report["real_train"]["average_transition_count"]
    collapse_warning = (
        report["generated"]["average_unique_tokens"] < 0.5 * real_reference_unique
        or report["generated"]["average_transition_count"] < 0.5 * real_reference_transitions
    )
    report["warning"] = "Token generator is collapsing during inference." if collapse_warning else None

    output_path = eval_dir / "token_stats_comparison.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("V2 token stats comparison")
    print(f"Real train average unique tokens: {report['real_train']['average_unique_tokens']:.3f}")
    print(f"Generated average unique tokens: {report['generated']['average_unique_tokens']:.3f}")
    print(f"Real test average unique tokens: {report['real_test']['average_unique_tokens']:.3f}")
    print(f"Real train average transition count: {report['real_train']['average_transition_count']:.3f}")
    print(f"Generated average transition count: {report['generated']['average_transition_count']:.3f}")
    print(f"Real test average transition count: {report['real_test']['average_transition_count']:.3f}")
    print(f"Real train repeated token ratio: {report['real_train']['average_repeated_token_ratio']:.3f}")
    print(f"Generated repeated token ratio: {report['generated']['average_repeated_token_ratio']:.3f}")
    print(f"Real test repeated token ratio: {report['real_test']['average_repeated_token_ratio']:.3f}")
    if report["warning"]:
        print(report["warning"])
    print(f"Saved comparison: {output_path}")
    return report


if __name__ == "__main__":
    compare_token_stats()
