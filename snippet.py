import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import math
import re

df = pd.read_csv("/home/ubuntu/pandas-snippet/train.csv")


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


df = pd.read_csv("/home/ubuntu/pandas-snippet/train.csv")
print("read from csv\n")
print(df)
print("\n-----------------------------------\n")
df = db_pd("SELECT * FROM order_dat_m WHERE order_serial=%s", [4175800])
print("read from db mysql\n")
print(df["data"])
print("\n")
print(df["status"])
print("\n")
print(df["error"])
print("\n-----------------------------------\n")
df = db_pd(
    "SELECT  order_serial,juchubi,tanka,shouhin,kingaku,urite_shamei,ken,gyoushu,chumon,suryou FROM order_dat_m WHERE ken=%s AND order_time>DATE_ADD(CURRENT_DATE, INTERVAL - 30 DAY)",
    ("東京都",),
)
print("read from db sqlalchemy\n")
print(df["data"])
print("\n")
print(df["status"])
print("\n")
print(df["error"])
print("\n-----------------------------------\n")
df = df["data"]
print(df.mean(numeric_only=True))
df_filtered = df[
    np.abs((df["kingaku"] - df["kingaku"].mean()) / (df["kingaku"].std())) < 2
]  # 金額について外れ値除外
df1 = df_filtered.loc[df["chumon"] == "通"]  # 特定のデータ抜粋
df2 = df_filtered.loc[df["chumon"] == "ONEクリック"]
plt.figure()
plt.hist(df1["kingaku"].tolist(), bins=20, color="skyblue", alpha=0.5)
plt.hist(df2["kingaku"].tolist(), bins=20, color="red", alpha=0.5)
plt.savefig("histogram.png", dpi=300, bbox_inches="tight")
# 配列について
test = [21]
test.append(22)
print(test)
test.extend([23, 24, 25])
print(test)
test.insert(2, 22.5)
print(test)
ans = []
for i, num in enumerate(test):
    if num % 1 > 0:
        print("{}th in test has digits".format(i))
    ans += [math.floor(num)]
print(ans)
print(ans.count(22))
ans.pop()
print(ans)
ans.pop(2)
print(ans)
tdict = {"insert": 0, "home": 1, "pgup": 2, "delete": 3, "end": 4, "pgdn": 5}
print(tdict.keys())
print(tdict.values())
for key in tdict.keys():
    if re.match("pg", key):
        print("{} is related with page".format(key))
