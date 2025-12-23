import requests
from fastapi import APIRouter, Request
import func
import logging
import numpy as np
import jageocoder
import geopandas as gpd
import pandas as pd
import folium
from fastapi.responses import HTMLResponse
import math
import datetime

today = datetime.date.today()
this_year = int(today.year)
jageocoder.init(url="https://jageocoder.info-proto.com/jsonrpc")


router = APIRouter()

envs = func.get_envs()
ESTATE = envs["estate"]
ESTAT = envs["estat"]
logging.basicConfig(level=logging.INFO)


@router.get("/api/q2tile/{code}")
async def estat_data(code: str):
    target = {}
    address_data = jageocoder.search(code.replace(" ", ""))
    if (
        address_data["candidates"][0]["fullname"][1] in DESIGNATED_CITY
    ):  # 政令指定都市の場合、区をWardに代入
        target["pref"] = address_data["candidates"][0]["fullname"][0]
        target["ward"] = address_data["candidates"][0]["fullname"][2]
        target["city"] = address_data["candidates"][0]["fullname"][3]
    else:
        target["pref"] = address_data["candidates"][0]["fullname"][0]
        target["ward"] = address_data["candidates"][0]["fullname"][1]
        target["city"] = address_data["candidates"][0]["fullname"][2]

    # タイル情報取得
    zoom_level = 15  # 14:約2.45キロ 15:約1.22キロ
    y = address_data["candidates"][0]["y"]
    x = address_data["candidates"][0]["x"]
    first_column, population = get_population(x, y, zoom_level)
    population_rate = get_population_rate(target)
    return {
        "x": x,
        "y": y,
        "zoom_level": zoom_level,
        "population": {
            "years": population.index.tolist(),
            "ages": population.columns.tolist(),
            "data": population.values.tolist(),
            "sum": first_column.values.tolist(),
        },
        "population_rate": {
            "year": population_rate.columns.tolist(),
            "data": population_rate.values.tolist(),
        },
        "pref": address_data["candidates"][0]["fullname"][0],
        "city": target["city"],
    }


@router.get("/api/tile2map")
async def tile2map(
    request: Request,
    x: float,
    y: float,
    zoom_level: int,
):
    tile = latlon2tile(x, y, zoom_level)

    m = folium.Map(
        location=[
            y,
            x,
        ],
        zoom_start=15,
        width=800,
    )
    folium.Marker(
        location=[
            (y),
            (x),
        ]
    ).add_to(m)
    folium.Polygon(
        locations=[
            tile2latlon(tile[1], tile[0], zoom_level),
            tile2latlon(tile[1] + 1, tile[0], zoom_level),
            tile2latlon(tile[1] + 1, tile[0] + 1, zoom_level),
            tile2latlon(tile[1], tile[0] + 1, zoom_level),
        ],
        color="blue",
        weight=1,
        fill=True,
        fill_opacity=0.5,
    ).add_to(m)

    map_html = m._repr_html_()
    return HTMLResponse(content=map_html)


def latlon2tile(lon, lat, z):
    """
    緯度経度をタイル座標に変換
    """
    x = int((lon / 180 + 1) * 2**z / 2)  # x座標
    y = int(
        ((-np.log(np.tan((45 + lat / 2) * np.pi / 180)) + np.pi) * 2**z / (2 * np.pi))
    )  # y座標
    return [y, x]


def tile2latlon(xtile, ytile, z):
    n = 2**z
    lon = xtile / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
    return lat, lon


DESIGNATED_CITY = [
    "札幌市",
    "仙台市",
    "千葉市",
    "横浜市",
    "川崎市",
    "名古屋市",
    "京都市",
    "大阪市",
    "堺市",
    "神戸市",
    "岡山市",
    "広島市",
    "北九州市",
    "福岡市",
    "熊本市",
    "鹿児島市",
]


def api2df(url, params):
    """
    不動産APIの情報をデータフレームに変換
    """
    response = requests.get(
        url, headers={"Ocp-Apim-Subscription-Key": ESTATE}, params=params
    )
    user_data = response.json()
    return user_data
    if "message" in user_data:
        return user_data
    elif "data" in user_data:
        return pd.DataFrame(user_data["data"])
    elif "features" in user_data:
        if len(user_data["features"]) > 0:
            return gpd.GeoDataFrame.from_features(
                user_data["features"], crs="EPSG:4326"
            )
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()


def round5(x):
    if x >= 0:
        return math.floor(x / 5) * 5
    else:
        return 0


def get_population(x, y, zoom_level):
    tile = latlon2tile(x, y, zoom_level)
    url = "https://www.reinfolib.mlit.go.jp/ex-api/external/XKT013"
    params = {"response_format": "geojson", "z": zoom_level, "x": tile[1], "y": tile[0]}
    response = requests.get(
        url, headers={"Ocp-Apim-Subscription-Key": ESTATE}, params=params
    )
    user_data = response.json()
    # return user_data
    p_list = []
    for properties in user_data["features"]:
        property = properties["properties"]
        # propertiesの中の各キーを1行の辞書としてリストに追加
        p_list.append(property)
    df = pd.DataFrame(p_list)
    long = df.filter(regex=r"^PT\d{2}_\d{4}$").melt(var_name="key", value_name="value")

    long[["age", "year"]] = long["key"].str.extract(r"PT(\d{2})_(\d{4})").astype(int)

    population = long.groupby(["year", "age"])["value"].sum().unstack("age")
    first_column = population.iloc[:, 0]
    population = population.iloc[:, 1:]
    population.columns = [
        f"{5 * a - 5}-{5 * a - 1}歳" for a in population.columns if a > 0
    ]
    return first_column, population


def get_population_rate(target):
    pref = func.get_estimated_per_pref(target["pref"])
    city = func.get_estimated_per_city(target["city"])
    df = pd.concat([pref, city], ignore_index=True)
    return df
