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
# TODO var model
# TODO time series
# TODO ベイズ構造時系列モデル
# TODO　ポアソン回帰
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import yfinance as yf
# 正規分布(平均,分散)に従うX(μx,ρx^2),Y(μy,ρy^2),それぞれ独立について
# X+Yは正規分布X+Y(μx+μy,ρx^2+ρy^2)に従う
# 一般化すると線形結合aX+bYは正規分布aX+bY(aμx+bμy,(a^2)(ρx^2)+(b^2)(ρy^2))
# 正規分布(μ,ρ^2)に従うデータをn個ランダム抽出した標本は正規分布(nμ,nρ^2)に従う。
# 線形結合の性質から平均＝正規分布(nμ,nρ^2)/nは正規分布(μx,(ρx^2)/n)に従う
# ランダム抽出した標本数nの分散はカイ二乗分布(df=n-1)に従う

# 分布が既知の任意の統計量は変数変換で確率密度関数にできるらしい……


# 2*2のカイ二乗検定(=Z検定)二群の比率の検定
# 同一の集団と仮定して共通の母比率から分散を求め、カイ二乗分布に従うか検定している
# |t1|t2|
# |s1|s2|
def chi_squared(t1, t2, s1, s2):
    if t1 * s1 * t2 * s1 <= 0:
        raise Exception("all columns must be positive")
    chi_sq = (
        (t1 + t2 + s1 + s2)
        * ((t1 * s2 - s1 * t2) ** 2)
        / (t1 + t2)
        / (s1 + s2)
        / (t1 + s1)
        / (t2 + s2)
    )

    p_value = stats.chi2.sf(chi_sq, 1)
    return chi_sq, p_value


# 1標本t検定
# 正規分布からランダム抽出したものは正規分布に従う
# 平均の差も正規分布に、推定された分散もカイ二乗分布に従う
# 正規分布/(カイ二乗分布/自由度)**1/2はt分布に従うので検定できる
def t_1(df_column, expected_mean, both_side=False):
    # p_value=t_1(df["target"],float,True)
    t_double = 2 if both_side else 1

    standard_error = np.sqrt(df_column.var(ddof=1) / len(df_column))
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


# 対応ある2標本t検定
# 正規分布に従うならその平均の差も正規分布に従う
# 統合された(標本)標準誤差**1/2(=分散)はカイ二乗分布に従う
# 正規分布/(カイ二乗分布/自由度)**1/2はt分布に従うので検定できる
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


# 2標本t検定(非等分散)welch
# それぞれ正規分布に従うならその平均の差も正規分布に従う
# 等分散であれば統合された(標本)標準誤差**1/2(=分散)はカイ二乗分布に従う
# だが非等分散なので統合された(標本)標準誤差**1/2(=分散)を自由度で調整することでカイ二乗分布に近似
# 正規分布/(カイ二乗分布/自由度)**1/2はt分布に従うので検定できる
# なお非正規分布でも平均の差や分散は正規分布に収束するため使える
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


# グループごとの平方和の比の検定(分散分析)
# 一つの特徴量の平均値について、3カテゴリ以上の間で差があるか検定する
# category間の比/category内の比=F値は、F(グループ間の自由度,全体の自由度-グループ間の自由度)分布に従う
# ||target|category|value_1|value_2|value_3...
# |1|--s1-|-A--|...
# |2|--s2-|-B--|...
# |3|--s3-|-A--|...
# |4|--s4-|-C--|...
def annova(df, target, category):
    categorys = df[category].unique()
    N = len(df)
    k = len(categorys)
    grand_mean = df[target].mean()

    ss_between = 0
    ss_within = 0

    for g in categorys:
        x = df[df[category] == g][target]
        ss_between += len(x) * (x.mean() - grand_mean) ** 2
        ss_within += ((x - x.mean()) ** 2).sum()

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)

    F = ms_between / ms_within
    p_value = stats.f.sf(
        F, k - 1, N - k * 1
    )  # Nは全グループの自由度の和と考えると、グループ一種類毎に-1しなければならないため
    return F, p_value


# mann-whitneyのU検定
# 各標本についてもう一つの標本より順位の高い場合1を足していった和
# =ベルヌーイ変数の和なのでU値は標準化すると正規分布に収束する
# ノンパラだがそれぞれの分布の形が近しいことが条件
# 帰無仮説は一方がもう一方に勝つ勝率==0.5
def mann_whitney_u(df, target, category):
    categorys = df[category].unique()
    if len(categorys) != 2:
        raise ValueError("category must have exactly 2 groups")
    df["u_index"] = stats.rankdata(df[target], method="average")
    df_grouped = df.groupby(category)
    n1 = df_grouped.size()[0]
    n2 = df_grouped.size()[1]
    R1 = df_grouped["u_index"].sum()[0]
    R2 = df_grouped["u_index"].sum()[1]
    u_value = min(
        n1 * n2 + (n1 * (n1 + 1) / 2) - R1, n1 * n2 + (n2 * (n2 + 1) / 2) - R2
    )
    z_value = (u_value - (n1 * n2 / 2) - 0.5) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    p_value = 2 * stats.norm.sf(abs(z_value))
    plt.figure(figsize=(25, 15))
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.scatter(
        np.linspace(0, 0, n1),
        df.loc[df[category] == categorys[0]][target],
        c="red",
        label=f"{category}:{categorys[0]}",
    )
    plt.scatter(
        np.linspace(1, 1, n2),
        df.loc[df[category] == categorys[1]][target],
        c="blue",
        label=f"{category}:{categorys[1]}",
    )
    plt.title(f"p_value:{p_value} z_value:{z_value}")
    plt.legend()
    plt.savefig("mann_whitney_u")
    return u_value, z_value, p_value


# brunner-munzel検定
# 非等分散でも(分布の形が違っても)使える順位和検定(ノンパラ)
# 順位和と順位の分散を使ったBM統計量(構造はwelch-tと一緒)が、t分布に従う
def brunner_munzel(df, target, category):
    categorys = df[category].unique()
    if len(categorys) != 2:
        raise ValueError("category must have exactly 2 groups")
    df["u_index"] = stats.rankdata(df[target], method="average")
    df_1 = df.loc[df[category] == categorys[0]].copy()
    df_2 = df.loc[df[category] == categorys[1]].copy()
    R1 = df_1["u_index"].sum()
    R2 = df_2["u_index"].sum()
    df_1["inner_index"] = stats.rankdata(df_1[target], method="average")
    df_2["inner_index"] = stats.rankdata(df_2[target], method="average")
    df_1["sigma"] = (
        df_1["u_index"] - df_1["inner_index"] - R1 + (len(df_1) + 1) / 2
    ) ** 2
    df_2["sigma"] = (
        df_2["u_index"] - df_2["inner_index"] - R2 + (len(df_2) + 1) / 2
    ) ** 2
    sigma1 = df_1["sigma"].sum() / (len(df_1) - 1)
    sigma2 = df_2["sigma"].sum() / (len(df_2) - 1)
    BM = (
        len(df_1)
        * len(df_2)
        * (R1 - R2)
        / (len(df))
        / np.sqrt(len(df_1) * sigma1 + len(df_2) * sigma2)
    )
    dfree = ((len(df_1) * sigma1 + len(df_2) * sigma2) ** 2) / (
        (((len(df_1) * sigma1) ** 2) / (len(df_1) - 1))
        + (((len(df_2) * sigma2) ** 2) / (len(df_2) - 1))
    )
    p_value = stats.t.sf(abs(BM), dfree) * 2

    plt.figure(figsize=(25, 15))
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.scatter(
        np.linspace(0, 0, len(df_1)),
        df_1[target],
        c="red",
        label=f"{category}:{categorys[0]}",
    )
    plt.scatter(
        np.linspace(1, 1, len(df_2)),
        df_2[target],
        c="blue",
        label=f"{category}:{categorys[1]}",
    )
    plt.title(f"p_value:{p_value} bm_value:{BM}")
    plt.legend()
    plt.savefig("mann_whitney_u")
    return BM, p_value


# 単回帰
# y=ax+b,(sn,tn)について残差はsigma(asn+b-tn)**2、これの最小化を目指す
# 展開してa**2*sigma(sn)**2+2ab*sigma(sn)+n*b**2-2a*sigma(tn*sn)-2b*sigma(tn)+sigma(tn**2)
# aについて微分を取って2a*sigma(sn**2)+2b*sigma(sn)-2*sigma(sn*tn)=0
# bについて微分を取って2a*sigma(sn)+2nb-2*sigma(tn)=0
# 連立するとa=(-sigma(sn)*sigma(tn)+n*sigma(tn*sn))/(n*sigma(sn**2)-(sigma(sn))**2)
# b=-(a*sigma(sn)-sigma(tn))/n
def simple_reg(df, target):
    df_copy = df.copy()
    if len(df.columns) != 2:
        raise ValueError("simple reg is single parameter regression")
    parameters = {}
    plt.figure(figsize=(25, 15))
    for column in df_copy.columns:
        if column != target:
            df_copy[column + "_sq"] = df_copy[column] ** 2
            df_copy[column + "_mult"] = df_copy[column] * df_copy[target]
            param = (
                -df_copy[column].sum() * df_copy[target].sum()
                + len(df_copy) * df_copy[column + "_mult"].sum()
            ) / (
                len(df_copy) * (df_copy[column + "_sq"].sum())
                - (df_copy[column].sum()) ** 2
            )
            parameters[column] = {"param": param}
            seg = (
                -1
                * (param * (df_copy[column].sum()) - df_copy[target].sum())
                / len(df_copy)
            )
            plt.scatter(df_copy[column], df_copy[target], c="red")
            plt.scatter(df_copy[column], (df_copy[column] * param) + seg, c="blue")
    plt.savefig("simple_reg")


# 重回帰
# 実測値をベクトルY、説明変数を行列x、各係数をベクトルwとし、モデルをy=wx(y:vector,w:vector,x:vector matrix)として
# sigma(Y-y)**2の最小化を目指す。
# (Y-y)^T*(Y-y)=(Y-xw)^T*(Y-xw)=(Y^T-(xw)^T)*(Y-xw)=Y^T*Y-2Y^T*xw+w^T*x^T*w*xと変形してwについて微分し、0になるのは
# w=(x^T*x)**-1*x^T*yのとき。
# ただし(x^T*x)**-1が計算できないとき解が定まらない
def multi_reg(df, target):
    cols = [c for c in df.columns if c != target]
    X_vector = np.hstack([np.ones((len(df), 1)), df[cols].to_numpy()])
    beta = np.linalg.pinv(X_vector) @ df[target].to_numpy()


def hist(df):
    plt.figure(figsize=(20, 10))
    n_cols = len(df.columns)
    n_rows = (n_cols + 3) // 4  # 1行に4つ並べる
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        ax = axes[i]
        if df[col].dtype == "object":
            counts = df[col].value_counts()
            counts.plot(kind="bar", ax=ax, alpha=0.7)
            ax.set_title(f"Value counts of {col}")
        else:
            # if df[col].nunique() > 2 and np.issubdtype(df[col].dtype, np.number):
            #    df = func.shrink_outlier(df, col)
            an, bins, patches = ax.hist(df[col].dropna(), bins=20, alpha=0.7)
            # ビンの境界をx軸上にテキストで表示
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
            ax.set_title(f"{col}")
    # plt.tight_layout()
    plt.legend()
    plt.savefig("hist")


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


if __name__ == "__main__":
    df = get_stock_history("^N225", "2023-01-01", "2024-01-01")
    df_all = get_stock_history("^N225", "2024-01-01", "2025-01-01")
    simple_reg(df.loc[:, ["Close", "Open"]], "Close")
    # expected_mean = df_all["Close"].mean()
    # standard_error, ci_low, ci_high, p_value = unpaired_t(
    #    df["Close"], df_all["Close"], False, "under_covid", "post_covid"
    # )
    # plt.figure(figsize=(20, 10))
    # plt.plot(df.index, df["Close"])
    # plt.savefig("line.png")
    # df["is_post_june"] = (df.index >= "2023-06-01").astype(int)
    # mann_whitney_u(df, "Close", "is_post_june")
