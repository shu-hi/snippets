import requests
from fastapi import APIRouter
import func
import logging
import numpy as np
import jageocoder
import geopandas as gpd
import pandas as pd

jageocoder.init(url="https://jageocoder.info-proto.com/jsonrpc")

target = dict(address="東京都足立区綾瀬")
router = APIRouter()

envs = func.get_envs()
ESTATE = envs["estate"]
logging.basicConfig(level=logging.INFO)


@router.get("/api/real_estate/{code}")
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
    target["lat_lon"] = [
        address_data["candidates"][0]["y"],
        address_data["candidates"][0]["x"],
    ]

    tile = latlon2tile(target["lat_lon"][1], target["lat_lon"][0], zoom_level)
    # return tile
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
    summary = {}
    for i in range(2025, 2070, 5):
        for y in range(1, 20, 1):
            if i not in summary:
                summary[i] = {}
            summary[i][str(5 * y - 5) + "-" + str(5 * y - 1) + "歳"] = df[
                "PT" + f"{y:02}" + "_" + str(i)
            ].sum()
    return {
        "summary": summary,
        "latlon": user_data["features"][0]["geometry"]["coordinates"],
    }


def latlon2tile(lon, lat, z):
    """
    緯度経度をタイル座標に変換
    """
    x = int((lon / 180 + 1) * 2**z / 2)  # x座標
    y = int(
        ((-np.log(np.tan((45 + lat / 2) * np.pi / 180)) + np.pi) * 2**z / (2 * np.pi))
    )  # y座標
    return [y, x]


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


def get_9tiles_data(url, params):
    """
    該当タイルの周りまで取得
    """
    gdf_list = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            new_params = params.copy()
            new_params["x"] = params["x"] + i
            new_params["y"] = params["y"] + j
            gdf = api2df(url, new_params)
            if len(gdf) > 0:
                gdf_list.append(gdf)
    if len(gdf_list) > 0:
        return pd.concat(gdf_list)
    else:
        return pd.DataFrame()


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
