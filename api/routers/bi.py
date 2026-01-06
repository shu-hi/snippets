from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from fastapi.concurrency import run_in_threadpool
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
import io
import statsmodels.formula.api as smf
import numpy as np
import base64
from lifelines import CoxPHFitter
import logging
import func

logging.basicConfig(level=logging.INFO)
router = APIRouter()


class ExeData(BaseModel):
    sql: str
    params: List[str]


class DidData(BaseModel):
    sql: str
    params: List[str]
    del_outlier: bool


@router.post("/api/execute")
async def execute(data: ExeData):
    return "temporary disaled for bar"
    result = await run_in_threadpool(func.db_pd, data.sql, data.params)
    logging.info(result)
    if (result["status"] == "ok") or (result["status"] == "fallback"):
        if result["data"] == "noresult":
            return result
        df = result["data"].replace([np.inf, -np.inf], np.nan).fillna(0)
        result["data"] = df.to_dict(orient="records")
    else:
        result["data"] = "error"
    return result


@router.post("/api/head")
async def head(data: ExeData):
    return "temporary disaled for bar"
    result = await run_in_threadpool(func.db_pd, data.sql, data.params)

    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"].replace([np.inf, -np.inf], np.nan).fillna(0)
        result["data"] = df.head(100).to_dict(orient="records")
    return result


@router.get("/api/pref")
async def pref(data: str):
    result = await run_in_threadpool(func.convert_pref, data)
    return result


@router.post("/api/plot")
async def plot(data: ExeData):
    result = await run_in_threadpool(func.db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"]
        plt.figure(figsize=(20, 10))
        for i, plots in enumerate(df.columns):
            if i == 0:
                continue
            else:
                plt.plot(df[df.columns[0]], df[plots], label=plots, marker="o")
        plt.ylabel("plots")
        plt.xlabel(df.columns[0])
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")


@router.post("/api/hist")
async def hist(data: ExeData):
    result = await run_in_threadpool(func.db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"]

        plt.figure(figsize=(20, 10))
        n_cols = len(df.columns)
        n_rows = (n_cols + 2) // 3  # 1行に3つ並べる

        fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(df.columns):
            ax = axes[i]
            if df[col].dtype == "object":
                counts = df[col].value_counts()
                counts.plot(kind="bar", ax=ax, alpha=0.7)
                ax.set_title(f"Value counts of {col}")
            else:
                if df[col].nunique() > 2 and np.issubdtype(df[col].dtype, np.number):
                    df = func.del_outlier(df, col)
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
        plt.tight_layout()
        plt.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")


@router.post("/api/did")
async def did(data: DidData):
    result = await run_in_threadpool(func.db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"]
        if data.del_outlier:
            df = func.del_outlier(df, "target")
        model = smf.ols("target ~ treated * C(period_flag)", data=df).fit()
        summary_df = pd.DataFrame(
            {
                "Variable": model.params.index,
                "Coef": model.params.values,
                "StdErr": model.bse.values,
                "t": model.tvalues.values,
                "P>|t|": model.pvalues.values,
                "CI_lower": model.conf_int()[0].values,
                "CI_upper": model.conf_int()[1].values,
                "outlier": "deleted" if data.del_outlier else "included",
            }
        )
        buf = func.visualize_did(model)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        # JSON化しやすい形に変換
        result["data"] = {
            "summary": summary_df.to_dict(orient="records"),
            "plot": img_base64,
        }
    return result


@router.post("/api/lifelines")
async def lifelines(data: ExeData):
    result = await run_in_threadpool(func.db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"]
        df = func.del_outlier(df, "duration")
        cph = CoxPHFitter()
        cph.fit(df, duration_col="duration", event_col="event")
        cph.print_summary()
        summary = cph.summary.reset_index()
        summary = summary.rename(columns={"covariate": "variable"})
        summary["exp(coef)"] = summary["exp(coef)"].round(3)
        summary["p"] = summary["p"].round(4)
        result["data"] = summary[["variable", "coef", "exp(coef)", "p"]].to_dict(
            orient="records"
        )
    return result


@router.post("/api/Linreg")
async def Linreg(data: ExeData):
    result = await run_in_threadpool(func.db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"]
        df = func.del_outlier(df, "target")
        df.dropna(inplace=True)
        df.index = np.arange(len(df.index))
        formula = "target ~ " + " + ".join(
            [col for col in df.columns if col != "target"]
        )
        model = smf.ols(formula, data=df).fit()
        summary_df = pd.DataFrame(
            {
                "Variable": model.params.index,
                "Coef": model.params.values,
                "rsquared": model.rsquared,
                "t": model.tvalues,
                "P>|t|": model.pvalues,
            }
        )

        df["predicted"] = model.predict(df)
        plt.figure(figsize=(20, 10))
        plt.plot(df.index, df["target"], label="Actual", marker="o")
        plt.plot(
            df.index,
            df["predicted"],
            label="Predicted(Regression Line)",
            linestyle="--",
        )
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        # JSON化しやすい形に変換
        result["data"] = {
            "summary": summary_df.to_dict(orient="records"),
            "plot": img_base64,
        }
    return result
