from abc import ABC, abstractmethod
import pandas as pd
import matplotlib as plt
import io
cache = {}
class BasicEDA:

    def __init__(self,df):
        self.df=df.copy()
    def _hist(self):
        plt.figure(figsize=(20, 10))
        n_cols = len(self.df.columns)
        n_rows = (n_cols + 3) // 4  # 1行に4つ並べる
        fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        for i, col in enumerate(self.df.columns):
            ax = axes[i]
            if self.df[col].dtype == "object":
                counts = self.df[col].value_counts()
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
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)