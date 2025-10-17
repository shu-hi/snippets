import requests
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

router = APIRouter()

API_KEY = "AIzaSyDViYNLgQPIQjG8Pqzj76nVnBYb4wj-pOM"
CX = "259ab40b8dcf64b08"


@router.get("/api/search/{query}")
async def search(query: str):
    url = (
        "https://www.googleapis.com/customsearch/v1"
        + "?cx="
        + CX
        + "&q="
        + query
        + "&key="
        + API_KEY
    )
    response = await run_in_threadpool(requests.get, url)
    data = response.json()
    # items がなければ空リスト返す
    items = data.get("items", [])

    # 抽出したいフィールドだけリストにまとめる
    results = []
    for item in items:
        results.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )
    return results
