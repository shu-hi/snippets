import requests
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
import func
import logging

router = APIRouter()

envs = func.get_envs()
CX = envs["cx"]
CS_API_KEY = envs["cs_api_key"]
GROQ = envs["groq"]
HF = envs["hf"]
logging.basicConfig(level=logging.INFO)


@router.get("/api/search/{query}")
async def search(query: str):
    url = (
        "https://www.googleapis.com/customsearch/v1"
        + "?cx="
        + CX
        + "&q="
        + query
        + "&key="
        + CS_API_KEY
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
                "snippet": item.get("snippet"),
                "url": item.get("formattedUrl"),
            }
        )
    return results
