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


def del_outlier(df, column):
    """
    外れ値を除いたものを返す
    @params df dataframe
    @params column str
    """
    return df[np.abs((df[column] - df[column].mean()) / (df[column].std())) < 2]


def convert_pref(input):
    """
    都道府県-郵便番号上三桁-県コード変換関数
    どれか入れるとレコードで返す
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


if __name__ == "__main__":
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

    df_pivot = df4.pivot(
        index="month_day", columns="year", values="sales"
    ).reset_index()

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
        (df4["year"] == 2025) & (df4["month_day"].between("02-01", "03-31")),
        "treatment",
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

    # # groupby して日ごとのfamilyごとの平均売上を比較（または合計でも可） groupby(column_name).agg(new_name=('old_name','mean'),new_name_2=('old_name_2','sum'))みたいなことができる
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
# SELECT
#   DAY(STR_TO_DATE(juchubi, '%Y/%m/%d')) AS juchu_date,
# SUM(CASE WHEN order_time >= '2025-06-01' AND order_time < '2025-07-01' THEN kingaku ELSE 0 END) AS total_june,
#   SUM(CASE WHEN order_time >= '2025-07-01' AND order_time < '2025-08-01' THEN kingaku ELSE 0 END) AS total_july,
#   SUM(CASE WHEN order_time >= '2025-08-01' AND order_time < '2025-09-01' THEN kingaku ELSE 0 END) AS total_augst
# from dami2.order_dat_m WHERE order_time >= '2025-06-01' AND order_time < '2025-09-01'
# GROUP BY DAY(STR_TO_DATE(juchubi, '%Y/%m/%d'))
# ORDER BY juchu_date;こうやってクエリで処理したほうが楽かも
