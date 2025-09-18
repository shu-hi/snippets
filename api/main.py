from fastapi import FastAPI
import sys
import os
from pydantic import BaseModel
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

main = FastAPI()

FILE_PATH = "table_data.json"


class TableData(BaseModel):
    name: str
    index: str


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

    # 3. 新しいデータを追加
    new_entry = {"name": data.name, "index": data.index}
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
