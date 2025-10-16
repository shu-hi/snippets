from fastapi import FastAPI
import sys
import os
from pydantic import BaseModel
from typing import List
from fastapi.concurrency import run_in_threadpool
import json
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
import io
import statsmodels.formula.api as smf
import numpy as np
import base64
from lifelines import CoxPHFitter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

main = FastAPI()

FILE_PATH = "table_data.json"


class TableData(BaseModel):
    name: str
    columns: str


class DellData(BaseModel):
    name: str


class ExeData(BaseModel):
    sql: str
    params: List[str]


class DidData(BaseModel):
    sql: str
    params: List[str]
    del_outlier: bool


@main.get("/api/data")
async def get_data():
    response = {"message": "Hello-World!"}
    return response


@main.post("/api/add_table")
async def add_table(data: TableData):
    # 1. 既存データを読み込む
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            try:
                table_list = json.load(f)
            except json.JSONDecodeError:
                table_list = []
    else:
        table_list = []

    # 2. name が既にあるかチェック
    if any(item["name"] == data.name for item in table_list):
        return {"status": "ng", "data": table_list}
    max_serial = max([item["serial"] for item in table_list], default=0)
    new_serial = max_serial + 1

    # 3. 新しいデータを追加
    new_entry = {"name": data.name, "columns": data.columns, "serial": new_serial}
    table_list.append(new_entry)

    # 4. JSON に書き込み
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(table_list, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "data": table_list}


@main.get("/api/get_table")
async def get_table():
    # 1. 既存データを読み込む
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            try:
                table_list = json.load(f)
            except json.JSONDecodeError:
                table_list = []
    else:
        table_list = []

    return {"status": "ok", "data": table_list}


@main.delete("/api/del_table")
async def del_table(data: DellData):
    # 1. 既存データを読み込む
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            try:
                table_list = json.load(f)
            except json.JSONDecodeError:
                table_list = []
    else:
        table_list = []

    new_table = [item for item in table_list if not (item["name"] == data.name)]

    if len(new_table) == len(table_list):
        # 削除対象が見つからなかった
        return {
            "status": "ng",
            "message": "該当データが見つかりません",
            "data": table_list,
        }

    # 3. ファイルに上書き保存
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(new_table, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "data": new_table}


@main.post("/api/execute")
async def execute(data: ExeData):
    result = await run_in_threadpool(db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"].replace([np.inf, -np.inf], np.nan).fillna(0)
        result["data"] = df.to_dict(orient="records")
    return result


@main.post("/api/head")
async def head(data: ExeData):
    result = await run_in_threadpool(db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"].replace([np.inf, -np.inf], np.nan).fillna(0)
        result["data"] = df.head(100).to_dict(orient="records")
    return result


@main.get("/api/pref")
async def pref(data: str):
    result = await run_in_threadpool(convert_pref, data)
    return result


@main.post("/api/plot")
async def plot(data: ExeData):
    result = await run_in_threadpool(db_pd, data.sql, data.params)
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


@main.post("/api/hist")
async def hist(data: ExeData):
    result = await run_in_threadpool(db_pd, data.sql, data.params)
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
                df = del_outlier(df, col)
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


@main.post("/api/did")
async def did(data: DidData):
    result = await run_in_threadpool(db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"]
        if data.del_outlier:
            df = del_outlier(df, "target")
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
        buf = visualize_did(model)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        # JSON化しやすい形に変換
        result["data"] = {
            "summary": summary_df.to_dict(orient="records"),
            "plot": img_base64,
        }
    return result


@main.post("/api/lifelines")
async def lifelines(data: ExeData):
    result = await run_in_threadpool(db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"]
        df = del_outlier(df, "duration")
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


@main.post("/api/Linreg")
async def Linreg(data: ExeData):
    result = await run_in_threadpool(db_pd, data.sql, data.params)
    if result["status"] == "ok" or result["status"] == "falback":
        df = result["data"]
        df = del_outlier(df, "target")
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


def tsv2_connection():
    return mysql.connector.connect(
        host="10.0.1.46",
        port=3306,
        user="ai_mmart",
        password="S76zt2zMPH",
        database="dami2",
    )


def tsv2_engine():
    return create_engine(
        "mysql+mysqlconnector://ai_mmart:S76zt2zMPH@10.0.1.46:3306/dami2"
    )


def db_connect(sql, params):
    """
    classic sql executer
    """
    try:
        conn6 = tsv2_connection()
        cursor6 = conn6.cursor(dictionary=True, buffered=True)
        cursor6.execute(sql, params)
        rserials = cursor6.fetchall()
        cursor6.close()
        conn6.close()
        print(rserials, flush=True)
    except Exception as e:
        print("rserial処理エラー:", e, flush=True)
    finally:
        # コネクションとカーソルを閉じる
        if cursor6:
            cursor6.close()
        if conn6:
            conn6.close()


def db_pd(sql, params):
    """
    sqlとparamを受けてstatus,data,errorを返す
    paramの形によってalchemyだったりmysqlだったり
    @params string sql
    @params list/tuple params(list-->mysql,tuple-->alchemy)

    @return res{"status": "", "data": "", "error": ""}
    """
    res = {"status": "error", "data": "", "error": ""}
    con = None
    try:
        con = tsv2_engine()
        res["data"] = pd.read_sql(sql=sql, con=con, params=params)
        res["status"] = "ok"
    except Exception as e:
        try:
            con = tsv2_connection()
            res["data"] = pd.read_sql(sql=sql, con=con, params=params)
            res["status"] = "fallback"
            res["error"] = str(e)
        except Exception as e2:
            res["status"] = "ng"
            res["error"] = str(e2)

    finally:
        if con:
            try:
                con.close()
            except Exception:
                pass
    return res


def convert_pref(input):
    """
    都道府県-郵便番号上三桁-県コード変換関数
    どれか入れるとレコードで返す
    郵便番号は例外が多いため、正確でないことに注意
    """
    pref_data = [
        {
            "name": "北海道",
            "code": "01",
            "zip": ["00", "04", "05", "06", "07", "08", "09"],
            "area": "北海道",
        },
        {"name": "青森県", "code": "02", "zip": ["03"], "area": "北東北"},
        {"name": "岩手県", "code": "03", "zip": ["02"], "area": "北東北"},
        {"name": "宮城県", "code": "04", "zip": ["98"], "area": "南東北"},
        {"name": "秋田県", "code": "05", "zip": ["01"], "area": "北東北"},
        {"name": "山形県", "code": "06", "zip": ["99"], "area": "南東北"},
        {"name": "福島県", "code": "07", "zip": ["96", "97"], "area": "南東北"},
        {"name": "茨城県", "code": "08", "zip": ["30", "31"], "area": "関東"},
        {"name": "栃木県", "code": "09", "zip": ["32"], "area": "関東"},
        {"name": "群馬県", "code": "10", "zip": ["37"], "area": "関東"},
        {
            "name": "埼玉県",
            "code": "11",
            "zip": ["33", "34", "35", "36"],
            "area": "関東",
        },
        {
            "name": "千葉県",
            "code": "12",
            "zip": ["26", "27", "28", "29"],
            "area": "関東",
        },
        {
            "name": "東京都",
            "code": "13",
            "zip": ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
            "area": "関東",
        },
        {
            "name": "神奈川県",
            "code": "14",
            "zip": ["21", "22", "23", "24", "25"],
            "area": "関東",
        },
        {"name": "新潟県", "code": "15", "zip": ["94", "95"], "area": "信越"},
        {"name": "富山県", "code": "16", "zip": ["93"], "area": "北陸"},
        {"name": "石川県", "code": "17", "zip": ["92"], "area": "北陸"},
        {"name": "福井県", "code": "18", "zip": ["91"], "area": "北陸"},
        {"name": "山梨県", "code": "19", "zip": ["40"], "area": "関東"},
        {"name": "長野県", "code": "20", "zip": ["38", "39"], "area": "信越"},
        {"name": "岐阜県", "code": "21", "zip": ["50"], "area": "中部"},
        {"name": "静岡県", "code": "22", "zip": ["41", "42", "43"], "area": "中部"},
        {
            "name": "愛知県",
            "code": "23",
            "zip": ["44", "45", "46", "47", "48", "49"],
            "area": "中部",
        },
        {"name": "三重県", "code": "24", "zip": ["51"], "area": "中部"},
        {"name": "滋賀県", "code": "25", "zip": ["52"], "area": "関西"},
        {"name": "京都府", "code": "26", "zip": ["60", "61", "62"], "area": "関西"},
        {
            "name": "大阪府",
            "code": "27",
            "zip": ["53", "54", "55", "56", "57", "58", "59"],
            "area": "関西",
        },
        {"name": "兵庫県", "code": "28", "zip": ["65", "66", "67"], "area": "関西"},
        {"name": "奈良県", "code": "29", "zip": ["63"], "area": "関西"},
        {"name": "和歌山県", "code": "30", "zip": ["64"], "area": "関西"},
        {"name": "鳥取県", "code": "31", "zip": ["68"], "area": "中国"},
        {"name": "島根県", "code": "32", "zip": ["69"], "area": "中国"},
        {"name": "岡山県", "code": "33", "zip": ["70", "71"], "area": "中国"},
        {"name": "広島県", "code": "34", "zip": ["72", "73"], "area": "中国"},
        {"name": "山口県", "code": "35", "zip": ["74", "75"], "area": "中国"},
        {"name": "徳島県", "code": "36", "zip": ["77"], "area": "四国"},
        {"name": "香川県", "code": "37", "zip": ["76"], "area": "四国"},
        {"name": "愛媛県", "code": "38", "zip": ["79"], "area": "四国"},
        {"name": "高知県", "code": "39", "zip": ["78"], "area": "四国"},
        {
            "name": "福岡県",
            "code": "40",
            "zip": ["80", "81", "82", "83"],
            "area": "九州",
        },
        {"name": "佐賀県", "code": "41", "zip": ["84"], "area": "九州"},
        {"name": "長崎県", "code": "42", "zip": ["85"], "area": "九州"},
        {"name": "熊本県", "code": "43", "zip": ["86"], "area": "九州"},
        {"name": "大分県", "code": "44", "zip": ["87"], "area": "九州"},
        {"name": "宮崎県", "code": "45", "zip": ["88"], "area": "九州"},
        {"name": "鹿児島県", "code": "46", "zip": ["89"], "area": "九州"},
        {"name": "沖縄県", "code": "47", "zip": ["90"], "area": "沖縄"},
    ]

    res = {"data": [], "status": "", "error": ""}
    if input:
        if isinstance(input, str) and len(input) == 2 and input.isdigit():
            for data in pref_data:
                if data["code"] == input:
                    res["data"] = data
                    break
        elif isinstance(input, str) and len(input) == 3 and input.isdigit():
            for data in pref_data:
                if input[0:2] in data["zip"]:
                    res["data"] = data
                    break
        else:
            for data in pref_data:
                if data["name"] == input:
                    res["data"] = data
                    break
        if len(res["data"]) > 0:
            res["status"] = "ok"
        else:
            res["status"] = "ng"
            res["error"] = "not found"
    else:
        res["status"] = "ng"
        res["error"] = "no input"
    return res


def visualize_did(model):
    """
    前・中・後について可視化する
    """
    # 回帰係数
    params = model.params
    pvalues = model.pvalues

    intercept = params["Intercept"]
    treated = params["treated"]
    period1_ctrl = params.get("C(period_flag)[T.1]", 0)
    period2_ctrl = params.get("C(period_flag)[T.2]", 0)
    period1_trt_inter = params.get("treated:C(period_flag)[T.1]", 0)
    period2_trt_inter = params.get("treated:C(period_flag)[T.2]", 0)

    # p値
    p_intercept = pvalues["Intercept"]
    p_treated = pvalues["treated"]
    p_period1_ctrl = pvalues.get("C(period_flag)[T.1]", 1)
    p_period2_ctrl = pvalues.get("C(period_flag)[T.2]", 1)
    p_period1_trt = pvalues.get("treated:C(period_flag)[T.1]", 1)
    p_period2_trt = pvalues.get("treated:C(period_flag)[T.2]", 1)
    periods = ["pre", "treatment", "post"]
    ctrl = [intercept, intercept + period1_ctrl, intercept + period2_ctrl]
    treat_effect = [treated, treated + period1_trt_inter, treated + period2_trt_inter]
    treat = [
        intercept + treated,
        intercept + treated + period1_trt_inter,
        intercept + treated + period2_trt_inter,
    ]
    ctrl_p = [p_intercept, p_period1_ctrl, p_period2_ctrl]
    treat_p = [p_treated, p_period1_trt, p_period2_trt]

    # グラフ作成
    x = np.arange(len(periods))
    width = 0.5
    fig, ax = plt.subplots(figsize=(8, 6))

    # control 部分
    ax.bar(x, ctrl, width, color="#a6bddb", label="control")

    # treatment は control の上に積む
    ax.bar(
        x, treat_effect, width, bottom=ctrl, color="#2b8cbe", label="treatment (extra)"
    )

    # 棒の上に有意性を表示
    for i, (c, p) in enumerate(zip(ctrl, ctrl_p)):
        if p < 0.05:
            ax.text(
                x[i],
                c + 0.1,
                "*",
                ha="center",
                va="bottom",
                color="black",
                fontsize=12,
            )
    for i, (t, p) in enumerate(zip(treat, treat_p)):
        if p < 0.05:
            ax.text(
                x[i],
                t + 0.1,
                "*",
                ha="center",
                va="bottom",
                color="black",
                fontsize=12,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_ylabel("predicted (Coef)")
    ax.set_title("DID: control/treatment")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    buffer = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def del_outlier(df, column):
    """
    外れ値を除いたものを返す
    @params df dataframe
    @params column str
    """
    return df[np.abs((df[column] - df[column].mean()) / (df[column].std())) < 2]
