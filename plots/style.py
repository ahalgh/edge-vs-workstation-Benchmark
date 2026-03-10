"""Shared plot styling for consistent benchmark visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns

# System colors
COLORS = {
    "jetson_thor": "#2ecc71",           # green for edge
    "blackwell_rtx5090": "#3498db",     # blue for workstation
}

# Fallback colors for unknown systems
DEFAULT_COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]


def apply_style():
    """Apply consistent plot style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def get_system_color(system_name: str, idx: int = 0) -> str:
    """Get color for a system name."""
    return COLORS.get(system_name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
