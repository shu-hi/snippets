import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import math
import re
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


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
print("\n" + df["status"] + "\n" + df["error"])
print("\n-----------------------------------\n")
df = db_pd(
    "SELECT  order_serial,juchubi,tanka,shouhin,kingaku,urite_shamei,ken,gyoushu,chumon,suryou FROM order_dat_m WHERE ken=%s AND order_time>DATE_ADD(CURRENT_DATE, INTERVAL - 3 DAY) order by juchubi",
    ("東京都",),
)
print("read from db sqlalchemy\n")
print(df["data"])
print("\n" + df["status"] + "\n" + df["error"])
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


df["index"] = np.arange(len(df.index))
X = df.loc[:, ["index"]]
y = df.loc[:, ["kingaku"]]
model = LinearRegression()
model.fit(X, y)
df["predicted_sales"] = model.predict(X)
plt.figure(figsize=(20, 10))
plt.plot(df.index, y, label="Actual Sales", marker="o")
plt.plot(
    df.index,
    df["predicted_sales"],
    label="Predicted Sales (Regression Line)",
    linestyle="--",
)
plt.xlabel("Time")
plt.ylabel("Sales")
plt.title("Linear Regression: Sales over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("linreg.png")
print(f"傾き（coefficient）: {model.coef_[0]}")
print(f"切片（intercept）: {model.intercept_}")

df3 = db_pd(
    "SELECT  order_serial,max(juchubi)AS juchubi,tanka,shouhin,SUM(kingaku)AS kingaku,urite_shamei,ken,gyoushu,chumon,suryou,max(order_time)as ord FROM order_dat_m WHERE ken=%s AND order_time>DATE_ADD(CURRENT_DATE, INTERVAL - 360 DAY) group by juchubi order by ord",
    ("東京都",),
)
df3 = df3["data"]

df3["lag_1"] = df3["kingaku"].shift(1)  # try to predict value of next day
df3.set_index("juchubi", inplace=True)  # set juchubi as index
df3.index = pd.to_datetime(df3.index)
X = df3.loc[:, ["lag_1"]]
X.dropna(inplace=True)  # drop n/a of x
y = df3.loc[:, "kingaku"]  # target
y, X = y.align(X, join="inner")  # inner:drop y where there's no X
model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

plt.figure(figsize=(20, 6))

# 実際の sales（ターゲット）
plt.plot(y.index, y, label="Actual Sales", marker="o")

# 予測した sales（モデル予測）
plt.plot(y_pred.index, y_pred, label="Predicted Sales (Lag Model)", linestyle="--")

plt.xlabel("Date" if isinstance(y.index[0], pd.Timestamp) else "Time")
plt.ylabel("Sales")
plt.title("Sales Prediction using Lag Feature (lag_1)")

plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lagreg.png")  # まあ当然当たらん

df4 = db_pd(
    """SELECT
    DATE(order_time) AS ord,
    SUM(kingaku) AS sales,
    CASE
        WHEN order_time BETWEEN '2023-09-02' AND '2024-09-01' THEN 2024
        WHEN order_time BETWEEN '2024-09-02' AND '2025-09-01' THEN 2025
    END AS year
FROM order_dat_m
WHERE ken = %s
  AND order_time BETWEEN '2023-09-02' AND '2025-09-01'
GROUP BY ord, year
ORDER BY ord, year;
""",
    ("東京都",),
)
df4 = df4["data"]
df4["ord"] = pd.to_datetime(df4["ord"], errors="coerce")
df4["month_day"] = df4["ord"].dt.strftime("%m-%d")

df_pivot = df4.pivot(index="month_day", columns="year", values="sales").reset_index()

# プロット
plt.figure(figsize=(14, 6))

for col in df_pivot.columns[1:]:  # year列だけループ
    plt.plot(df_pivot["month_day"], df_pivot[col], marker="o", label=f"{col}年度")

plt.title("Daily Sales Comparison by Month-Day")
plt.xlabel("Month-Day")
plt.ylabel("Sales (Kingaku)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("daily_sales_comparison.png")
# 以下DID、2-3月になんか試作したつもりでやる
df4["treatment"] = 0
df4.loc[
    (df4["year"] == 2025) & (df4["month_day"].between("02-01", "03-31")), "treatment"
] = 1

# postフラグ（今年の年を1、前年を0）
df4["post"] = (df4["year"] == 2025).astype(int)

# DID交互作用
df4["did"] = df4["treatment"] * df4["post"]

# DID回帰
model = smf.ols("sales ~ treatment + post + did", data=df4).fit()
print(model.summary())


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

fruits = pd.DataFrame(
    {"Apples": [35, 41], "Bananas": [21, 34]},
    index=["2017 Sales", "2018 Sales"],
)
print(fruits.iloc[:1])
print(fruits.loc[[0, 1], ["Apples", "Bananas"]])
# locはsliceすると尻も含むので注意
# top_oceania_wines = reviews.loc[(reviews.country.isin(["Australia","New Zealand"]))&(reviews.points>=95)]
# centered_price = reviews.price-reviews.price.mean()
# bargain_idx = (reviews.points / reviews.price).idxmax()
# bargain_wine = reviews.loc[bargain_idx, 'title']
# def stars(row):
#     if row.country == 'Canada':
#         return 3
#     elif row.points >= 95:
#         return 3
#     elif row.points >= 85:
#         return 2
#     else:
#         return 1

# star_ratings = reviews.apply(stars, axis='columns')
# price_extremes = reviews.groupby('variety').price.agg([min, max])
# best_rating_per_price = reviews.groupby('price').points.max().sort_index()
# country_variety_counts = reviews.groupby(['country','variety']).size().sort_values(ascending=False)
# combined_products = pd.concat([gaming_products, movie_products])
# powerlifting_combined = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
# n_missing_prices = reviews.price.isnull().sum()
# reviews_per_region = reviews.region_1.fillna("Unknown").value_counts().sort_values(ascending=False)


# store_sales['day']=store_sales['date'].dt.day

# may_data = store_sales[store_sales['date'].dt.month == 5]
# june_data = store_sales[store_sales['date'].dt.month == 6]

# # groupby して日ごとのfamilyごとの平均売上を比較（または合計でも可）
# may_grouped = may_data.groupby(['family', 'day'])['sales'].sum().reset_index()
# june_grouped = june_data.groupby(['family', 'day'])['sales'].sum().reset_index()

# # 比較のためにマージ（横並びで比較）
# comparison = pd.merge(
#     may_grouped,
#     june_grouped,
#     on=['family', 'day'],
#     how='outer',
#     suffixes=('_may', '_june')
# )

# # 欠損値を0で埋める（比較しやすく）
# comparison.sales_may.fillna(0, inplace=True)
# comparison.sales_june.fillna(0, inplace=True)

# # 結果表示
# print(comparison.head())
# plt.figure(figsize=(12,6))
