import pandas as pd

df_order = pd.read_excel(
    "sample.xlsx", sheet_name="注文データ"
)  # xlsxからdf読む　csvの時はread_csv
print(df_order.head())
df_stock = pd.read_excel("sample.xlsx", sheet_name="在庫データ")
df_customer = pd.read_excel("sample.xlsx", sheet_name="顧客データ")
df_item = pd.read_excel("sample.xlsx", sheet_name="アイテムデータ")
print(df_item["itemcate"].unique())  # 種類確認
df_item_book = df_item.loc[
    df_item["itemcate"] == "本"
]  # 条件に合うカラムのみ抜き出して新df作成
print(df_item_book.head())
df_item["item_num"] = df_item["item"].str.extract(
    r"(\d+)"
)  # 既存カラムから新規カラム作成
print(df_item[["item", "item_num"]].head())
df_electronics_customer = pd.merge(  # join
    df_item.loc[df_item["itemcate"] == "家電"][
        ["item", "itemcate"]
    ],  # itemcateが家電のレコードでitemとitemcateを抜粋
    df_order,  # join先
    left_on="item",
    right_on="orderitem",
    how="left",
)
print(df_electronics_customer.head())
