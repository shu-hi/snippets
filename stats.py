# 頻度論（頻度主義）

# t検定（1標本、2標本、対応のある）
# カイ二乗検定（独立性、適合度）
# Z検定（比率、平均）
# ANOVA（分散分析）
# Mann-Whitney U検定（ノンパラメトリック）
# Wilcoxon符号付順位検定（ノンパラメトリック）


# ベイジアン

# ベイジアンt検定
# ベイジアン回帰分析
# ベイジアンA/Bテスト
# ベイジアンモデル比較
# ベイジアン順序検定

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import yfinance as yf


# 1標本t検定
def t_1(df_column, expected_mean, both_side=False):
    # p_value=t_1(df["target"],float,True)
    t_double = 2 if both_side else 1

    standard_error = df_column.std(ddof=1) / np.sqrt(len(df_column))
    t = (df_column.mean() - expected_mean) / standard_error
    ci_low = df_column.mean() - stats.t.ppf(0.975, len(df_column) - 1) * standard_error
    ci_high = df_column.mean() + stats.t.ppf(0.975, len(df_column) - 1) * standard_error
    p_value = stats.t.sf(abs(t), len(df_column) - 1) * t_double

    plt.figure(figsize=(25, 15))
    fig, ax = plt.subplots(figsize=(25, 15))
    series_hist(df_column, ax)

    ax.axvline(expected_mean, color="C1", lw=2)
    ax.axvline(ci_low, color="C2", lw=2)
    ax.axvline(ci_high, color="C2", lw=2)
    plt.savefig("t_1")

    return standard_error, ci_low, ci_high, p_value


# 2標本t検定(非等分散)welch
def unpaired_t(df_1_column, df_2_column, both_side=False, name_1="", name_2=""):
    t_double = 2 if both_side else 1
    s2_1 = df_1_column.std(ddof=1) ** 2
    s2_2 = df_2_column.std(ddof=1) ** 2
    t = (df_1_column.mean() - df_2_column.mean()) / np.sqrt(
        s2_1 / len(df_1_column) + s2_2 / len(df_2_column)
    )
    freeliness = (s2_1 / len(df_1_column) + s2_2 / len(df_2_column)) ** 2 / (
        s2_1**2 / (len(df_1_column) ** 2 * (len(df_1_column) - 1))
        + s2_2**2 / (len(df_2_column) ** 2 * (len(df_2_column) - 1))
    )

    standard_error = np.sqrt(s2_1 / len(df_1_column) + s2_2 / len(df_2_column))
    ci_low = (df_1_column.mean() - df_2_column.mean()) - stats.t.ppf(
        0.975, freeliness
    ) * standard_error
    ci_high = (df_1_column.mean() - df_2_column.mean()) + stats.t.ppf(
        0.975, freeliness
    ) * standard_error
    p_value = stats.t.sf(abs(t), freeliness) * t_double

    plt.figure(figsize=(25, 15))
    fig, ax = plt.subplots(figsize=(25, 15))
    series_hist(df_1_column, ax, name_1)
    series_hist(df_2_column, ax, name_2)
    ax.axvline(df_1_column.mean(), color="C1", lw=2)
    ax.axvline(df_2_column.mean(), color="C1", lw=2)
    plt.legend()
    plt.savefig("unpaired_t")

    return standard_error, ci_low, ci_high, p_value


def paired_t(df_1_column, df_2_column, both_side=False):
    t_double = 2 if both_side else 1
    if len(df_1_column) != len(df_2_column):
        raise Exception("different series length")

    s2 = 0
    for i in range(len(df_1_column)):
        s2 += (df_1_column[i] - df_2_column[i]) ** 2 / (len(df_1_column) - 1)
    # t値の計算
    t = (df_1_column - df_2_column).mean() / np.sqrt(
        (df_1_column - df_2_column) / len(df_1_column)
    )
    # 標準誤差の計算
    standard_error = np.sqrt((df_1_column - df_2_column) / len(df_1_column))
    # 信頼区間の計算
    ci_low = (df_1_column - df_2_column).mean() - stats.t.ppf(
        0.975, (len(df_1_column) - 1)
    ) * standard_error
    ci_high = (df_1_column - df_2_column).mean() + stats.t.ppf(
        0.975, (len(df_1_column) - 1)
    ) * standard_error
    p_value = stats.t.sf(abs(t), (len(df_1_column) - 1)) * t_double

    return standard_error, ci_low, ci_high, p_value


# カイ二乗検定()
def chi_squared(df):
    observed = df.iloc[0].values
    expected = df.iloc[1].values

    if (
        observed.sum() < 99
        or observed.sum() > 101
        or expected.sum() < 99
        or expected.sum() > 101
    ):
        raise Exception("chi_sq input must be percentge")
    chi_sq = ((observed - expected) ** 2 / expected).sum()
    dfree = len(observed) - 1

    p_value = stats.chi2.sf(chi_sq, dfree)
    return chi_sq, dfree, p_value


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


def get_stock_history(symbol: str, start, end):
    ticker = yf.Ticker(symbol)
    return ticker.history(start=start, end=end)


df = get_stock_history("^N225", "2023-01-01", "2024-01-01")
df_all = get_stock_history("^N225", "2024-01-01", "2025-01-01")
expected_mean = df_all["Close"].mean()
standard_error, ci_low, ci_high, p_value = unpaired_t(
    df["Close"], df_all["Close"], False, "under_covid", "post_covid"
)
plt.figure(figsize=(20, 10))
plt.plot(df_all.index, df_all["Close"])
plt.savefig("line.png")
