from fastapi import APIRouter
from pydantic import BaseModel
import json
import os

router = APIRouter()

FILE_PATH = "table_data.json"


class TableData(BaseModel):
    name: str
    columns: str


class DellData(BaseModel):
    name: str


@router.post("/api/add_table")
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


@router.get("/api/get_table")
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


@router.delete("/api/del_table")
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
