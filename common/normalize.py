"""Result normalization: performance per dollar, per watt, energy per task."""


def perf_per_dollar(metric_value: float, system_cost_usd: float) -> float:
    """Compute performance-per-dollar (higher is better)."""
    if system_cost_usd <= 0:
        return 0.0
    return metric_value / system_cost_usd


def perf_per_watt(metric_value: float, avg_power_watts: float) -> float:
    """Compute performance-per-watt (higher is better)."""
    if avg_power_watts <= 0:
        return 0.0
    return metric_value / avg_power_watts


def energy_per_task(avg_power_watts: float, task_duration_seconds: float) -> float:
    """Compute energy per task in Joules."""
    return avg_power_watts * task_duration_seconds


def normalize_results(
    result_a: dict,
    result_b: dict,
    metric_key: str,
    cost_a: float,
    cost_b: float,
) -> dict:
    """Compare two systems on a given metric with normalization.

    Args:
        result_a: BenchmarkResult dict for system A.
        result_b: BenchmarkResult dict for system B.
        metric_key: Key to extract from results (e.g. 'tokens_per_sec').
        cost_a: System A cost in USD.
        cost_b: System B cost in USD.

    Returns:
        Comparison dict with raw and normalized values.
    """
    # Compute average metric across all result entries
    vals_a = [r[metric_key] for r in result_a["results"] if metric_key in r]
    vals_b = [r[metric_key] for r in result_b["results"] if metric_key in r]

    avg_a = sum(vals_a) / len(vals_a) if vals_a else 0.0
    avg_b = sum(vals_b) / len(vals_b) if vals_b else 0.0

    power_a = result_a.get("power", {}).get("avg_watts", 0.0)
    power_b = result_b.get("power", {}).get("avg_watts", 0.0)

    return {
        "metric": metric_key,
        "system_a": {
            "name": result_a.get("system", "A"),
            "raw": avg_a,
            "perf_per_dollar": perf_per_dollar(avg_a, cost_a),
            "perf_per_watt": perf_per_watt(avg_a, power_a),
        },
        "system_b": {
            "name": result_b.get("system", "B"),
            "raw": avg_b,
            "perf_per_dollar": perf_per_dollar(avg_b, cost_b),
            "perf_per_watt": perf_per_watt(avg_b, power_b),
        },
    }
