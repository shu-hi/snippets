import matplotlib as plt
from scipy import stats
import numpy as np

def series_hist(series, ax, name):
    an, bins, patches = ax.hist(
        series.dropna(), bins=20, density=True, alpha=0.7, label=name
    )
    for boundary in bins:
        ax.text(
            boundary,
            -500,
            f"{boundary:.1f}",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
            color="gray",
        )
    x = np.linspace(
        series.mean() - 3 * series.std(),
        series.mean() + 3 * series.std(),
        500,
    )
    y = stats.norm.pdf(x, series.mean(), series.std(ddof=1))
    ax.plot(
        x,
        y,
        color="C4",
    )
    ax.legend()