import requests
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from groq import AsyncGroq
from groq import Groq
from fastapi.responses import StreamingResponse
import func
import logging

router = APIRouter()

envs = func.get_envs()
CX = envs["cx"]
API_KEY = envs["api_key"]
GROQ = envs["groq"]
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
                "snippet": item.get("snippet"),
                "url": item.get("formattedUrl"),
            }
        )
    return results


@router.get("/api/chat/{query}")
async def chat(query: str):
    client = Groq(
        api_key=GROQ,
    )

    chat_completion = await run_in_threadpool(
        client.chat.completions.create,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content


@router.get("/api/streamchat/{query}")
async def streamchat(query: str):
    client = AsyncGroq(api_key=GROQ)

    async def generate():
        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            stream=True,
        )
        async for chunk_data in stream:
            content = chunk_data.choices[0].delta.content
            # logging.info(f"Chunk received: {content}")
            if content:
                yield content
            else:
                yield ""

    return StreamingResponse(generate(), media_type="text/event-stream")
