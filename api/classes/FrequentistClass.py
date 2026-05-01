import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from scipy import stats
from . import statsHelper as sH
class Frequentist:
    def __init__(self,df_1,df_2,target):
        self.df_1=df_1.copy()
        self.df_2=df_2.copy()
        self.target=target
    # 1標本t検定
    # 正規分布からランダム抽出したものは正規分布に従う
    # 平均の差も正規分布に、推定された分散もカイ二乗分布に従う
    # 正規分布/(カイ二乗分布/自由度)**1/2はt分布に従うので検定できる
    def t_1(self, expected_mean, both_side=False):
        # p_value=t_1(df_1["target"],float,True)
        t_double = 2 if both_side else 1
    
        standard_error = np.sqrt(self.df_1[self.target].var(ddof=1) / len(self.df_1[self.target]))
        t = (self.df_1[self.target].mean() - expected_mean) / standard_error
        ci_low = self.df_1[self.target].mean() - stats.t.ppf(0.975, len(self.df_1[self.target]) - 1) * standard_error
        ci_high = self.df_1[self.target].mean() + stats.t.ppf(0.975, len(self.df_1[self.target]) - 1) * standard_error
        p_value = stats.t.sf(abs(t), len(self.df_1[self.target]) - 1) * t_double
    
        plt.figure(figsize=(25, 15))
        fig, ax = plt.subplots(figsize=(25, 15))
        sH.series_hist(self.df_1[self.target], ax,'')
    
        ax.axvline(expected_mean, color="C1", lw=2)
        ax.axvline(ci_low, color="C2", lw=2)
        ax.axvline(ci_high, color="C2", lw=2)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    
        return standard_error, ci_low, ci_high, p_value,img_base64
    # 対応ある2標本t検定
    # 正規分布に従うならその平均の差も正規分布に従う
    # 統合された(標本)標準誤差**1/2(=分散)はカイ二乗分布に従う
    # 正規分布/(カイ二乗分布/自由度)**1/2はt分布に従うので検定できる
    def paired_t(self, both_side=False):
        t_double = 2 if both_side else 1
        if len(self.df_1[self.target]) != len(self.df_2[self.target]):
            raise Exception("different series length")
    
        s2 = 0
        for i in range(len(self.df_1[self.target])):
            s2 += (self.df_1[self.target][i] - self.df_2[self.target][i]) ** 2 / (len(self.df_1[self.target]) - 1)
        # t値の計算
        t = (self.df_1[self.target] - self.df_2[self.target]).mean() / np.sqrt(
            (self.df_1[self.target] - self.df_2[self.target]) / len(self.df_1[self.target])
        )
        # 標準誤差の計算
        standard_error = np.sqrt((self.df_1[self.target] - self.df_2[self.target]) / len(self.df_1[self.target]))
        # 信頼区間の計算
        ci_low = (self.df_1[self.target] - self.df_2[self.target]).mean() - stats.t.ppf(
            0.975, (len(self.df_1[self.target]) - 1)
        ) * standard_error
        ci_high = (self.df_1[self.target] - self.df_2[self.target]).mean() + stats.t.ppf(
            0.975, (len(self.df_1[self.target]) - 1)
        ) * standard_error
        p_value = stats.t.sf(abs(t), (len(self.df_1[self.target]) - 1)) * t_double
    
        return standard_error, ci_low, ci_high, p_value

    # 2標本t検定(非等分散)welch
    # それぞれ正規分布に従うならその平均の差も正規分布に従う
    # 等分散であれば統合された(標本)標準誤差**1/2(=分散)はカイ二乗分布に従う
    # だが非等分散なので統合された(標本)標準誤差**1/2(=分散)を自由度で調整することでカイ二乗分布に近似
    # 正規分布/(カイ二乗分布/自由度)**1/2はt分布に従うので検定できる
    # なお非正規分布でも平均の差や分散は正規分布に収束するため使える
    def unpaired_t(self, both_side=False, name_1="1", name_2="2"):#name_(1|2)は表示用のカラム名
        t_double = 2 if both_side else 1
        s2_1 = self.df_1[self.target].std(ddof=1) ** 2
        s2_2 = self.df_2[self.target].std(ddof=1) ** 2
        t = (self.df_1[self.target].mean() - self.df_2[self.target].mean()) / np.sqrt(
            s2_1 / len(self.df_1[self.target]) + s2_2 / len(self.df_2[self.target])
        )
        freeliness = (s2_1 / len(self.df_1[self.target]) + s2_2 / len(self.df_2[self.target])) ** 2 / (
            s2_1**2 / (len(self.df_1[self.target]) ** 2 * (len(self.df_1[self.target]) - 1))
            + s2_2**2 / (len(self.df_2[self.target]) ** 2 * (len(self.df_2[self.target]) - 1))
        )
    
        standard_error = np.sqrt(s2_1 / len(self.df_1[self.target]) + s2_2 / len(self.df_2[self.target]))
        ci_low = (self.df_1[self.target].mean() - self.df_2[self.target].mean()) - stats.t.ppf(
            0.975, freeliness
        ) * standard_error
        ci_high = (self.df_1[self.target].mean() - self.df_2[self.target].mean()) + stats.t.ppf(
            0.975, freeliness
        ) * standard_error
        p_value = stats.t.sf(abs(t), freeliness) * t_double
    
        plt.figure(figsize=(25, 15))
        fig, ax = plt.subplots(figsize=(25, 15))
        sH.series_hist(self.df_1[self.target], ax, name_1)
        sH.series_hist(self.df_2[self.target], ax, name_2)
        ax.axvline(self.df_1[self.target].mean(), color="C1", lw=2)
        ax.axvline(self.df_2[self.target].mean(), color="C1", lw=2)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    
        return standard_error, ci_low, ci_high, p_value,img_base64