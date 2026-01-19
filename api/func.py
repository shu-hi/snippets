import os
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector
from sqlalchemy import create_engine
from dotenv import load_dotenv
import psycopg2

load_dotenv()  # func.pyもそうだが、envの読み込みは要工夫
if os.getenv("ENV") != "DEV" and os.getenv("ENV") != "TSV2":
    load_dotenv("/etc/secrets/.env")


def connection():
    if os.getenv("ENV") == "TSV2":
        return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            connection_timeout=3,
        )
    else:
        return psycopg2.connect(  # ipv6でしか直アクセスはできないので、pooler経由になる。https://supabase.com/docs/guides/database/connecting-to-postgres#connection-pooler 参照。なおuserも変わるので注意
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 5432)),  # PostgreSQLのデフォルトポートは5432
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            connect_timeout=3,
        )


def engine():
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    if os.getenv("ENV") == "TSV2":
        db_port = os.getenv("DB_PORT", "3306")
        url = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}?connect_timeout=3"
    else:
        db_port = os.getenv("DB_PORT", "3306")
        url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}?connect_timeout=3"
    return create_engine(url)


def db_connect(sql, params):
    """
    classic sql executer
    """
    try:
        conn6 = connection()
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
        if type(params) is list:
            con = connection()

        elif type(params) is tuple or params is None or not params:
            con = engine()
        else:
            raise Exception("invalid params" + str(type(params)))
        res["data"] = pd.read_sql(sql=sql, con=con, params=params)
        res["status"] = "ok"

    except Exception as e:
        res["error"] = str(e)
    return res


def pg_exec(sql, params):
    """
    postgress用のクエリ実行関数
    """
    res = {"status": "error", "data": "", "error": ""}
    con = None
    try:
        print(params, flush=True)
        print(type(params), flush=True)
        if type(params) is list:
            con = connection()
            cursor = con.cursor()
        elif type(params) is tuple or params is None or not params:
            con = engine()
            cursor = con.connect()
        else:
            raise Exception("invalid params" + str(type(params)))
        if isinstance(con, mysql.connector.connection.MySQLConnection) or isinstance(
            con, psycopg2.extensions.connection
        ):
            cursor.execute(sql, params)
            con.commit()
            res["data"] = "noresult"
            res["status"] = "ok"
        else:  # SQLAlchemyの場合
            with con.connect() as conn:
                conn.execute(sql, params)
            res["data"] = "noresult"
            res["status"] = "ok"
    except Exception as e:
        res["error"] = str(e)
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


def shrink_outlier(df, column, z=2):
    """
    外れ値を削除せず、±zσの範囲に丸める
    """
    mean = df[column].mean()
    std = df[column].std()

    lower = mean - z * std
    upper = mean + z * std

    df[column] = df[column].clip(lower, upper)
    return df


def get_estimated_per_pref(pref_name):
    """https://www.ipss.go.jp/pp-shicyoson/j/shicyoson18/1kouhyo/gaiyo_a.pdf
    page 18から手動で持ってきた"""
    df = pd.read_csv("estimated_population_per_city.csv")
    return df.loc[df["name"] == pref_name]


def get_estimated_per_city(city_name):
    """https://www.ipss.go.jp/pp-shicyoson/j/shicyoson18/1kouhyo/gaiyo_a.pdf
    page 67から手動で持ってきた"""
    df = pd.read_csv("estimated_population_per_city.csv")
    return df.loc[df["name"] == city_name]


def get_envs():
    CS_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY", "fb")
    CX = os.getenv("CUSTOM_SEARCH_ENGINE_ID", "fb")
    GROQ = os.getenv("GROQ_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")
    ESTAT_API_KEY = os.getenv("ESTAT_API_KEY")
    REAL_ESTATE_API = os.getenv("REAL_ESTATE_API")
    TOKEN_GENERATE_KEY = os.getenv("TOKEN_GENERATE_KEY")
    return {
        "cs_api_key": CS_API_KEY,
        "cx": CX,
        "groq": GROQ,
        "hf": HF_API_KEY,
        "estat": ESTAT_API_KEY,
        "estate": REAL_ESTATE_API,
        "token_generate_key": TOKEN_GENERATE_KEY,
    }
