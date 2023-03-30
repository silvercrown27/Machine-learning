import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)


def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("rocket", n_colors=X[period].nunique())
    ax = sns.lineplot(x=freq, y=y, hue=period, data=X,ci=False,
                      ax=ax, palette=palette, legend=False)
    ax.set_title(f"Seasonal Plot ({period}/{freq}")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name, xy=(1, y_), xytext=(6, 0), color=line.get_color(), size=14,
            xycoords=ax.get_yaxis_transform(), textcoords="offset points", va="center"
        )
    return ax

def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    frequencies, spectrum = periodogram(
        ts, fs=fs, detrend=detrend, window="boxcar", scaling='spectrum'
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(frequencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels([
        "Annual (1)", "Semiannual (2)", "Quarterly (4)", "Bimonthly (6)",
        "Monthly (12)", "Biweekly (26)", "Weekly (52)", "Semiweekly (104)"
    ], rotation=30,)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax