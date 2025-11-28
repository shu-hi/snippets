from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from groq import AsyncGroq
from groq import Groq
from fastapi.responses import StreamingResponse
import func
import logging
import numpy as np
import pandas as pd
import faiss
import httpx
import os
from io import BytesIO
import json

router = APIRouter()

envs = func.get_envs()
GROQ = envs["groq"]
HF = envs["hf"]
logging.basicConfig(level=logging.INFO)


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


@router.get("/api/sentence_embedding/{query}")
async def sentence_embedding(query: str):
    processed_sentence = await sentence_embed([query])
    return processed_sentence.tolist()


@router.post("/api/make_faiss")
async def make_faiss(csv: UploadFile = File(...)):
    file_path = await save_csv(csv)
    doc = await load_csv(file_path)
    embeddings = []
    batch_size = 16  # バッチサイズ
    for i in range(0, len(doc), batch_size):
        batch_docs = doc[i : i + batch_size]
        embedding_batch = await sentence_embed(batch_docs)
        embeddings.append(embedding_batch)
    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]  # ベクトル次元を取得
    faiss_index = faiss.IndexFlatL2(dim)  # FAISSインデックス作成
    faiss_index.add(embeddings)  # 埋め込みをFAISSに追加
    faiss.write_index(faiss_index, "documents.index")
    np.save("documents.npy", embeddings)
    return {"message": "FAISS index created and saved successfully."}


async def save_csv(csv_file: UploadFile):
    file_location = os.path.join("./", "documents.csv")
    with open(file_location, "wb") as f:
        content = await csv_file.read()  # ファイルの内容をメモリに読み込む
        f.write(content)
    return file_location


async def load_csv(file_path: str):
    # file_path を使ってファイルを読み込む
    try:
        with open(file_path, "rb") as f:  # バイナリモードでファイルを開く
            content = f.read()  # ファイルの内容を読み込む
    except FileNotFoundError:
        raise ValueError(f"ファイル {file_path} が見つかりません。")

    # ファイルが空かどうかをチェック
    if not content:
        raise ValueError(f"{file_path} は空のファイルです。")

    # content (bytes) を BytesIO オブジェクトに変換
    content_io = BytesIO(content)

    # pandasでCSVを読み込む
    try:
        df = pd.read_csv(content_io)
    except pd.errors.EmptyDataError:
        raise ValueError(f"{file_path} は空のCSVファイルです。")

    # '内容' 列が存在するか確認
    if "内容" not in df.columns:
        raise ValueError(f"{file_path} に '内容' 列が含まれていません。")

    # '内容' 列のデータをリストに変換
    documents = df["内容"].dropna().tolist()
    return documents


async def sentence_embed(query: list):
    api_url = "https://router.huggingface.co/hf-inference/models/intfloat/multilingual-e5-large/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {HF}"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            api_url, headers=headers, json={"inputs": query}, timeout=300.0
        )

    if response.status_code == 200:
        # APIレスポンスから埋め込みを取得
        embeddings = np.array(response.json())
        return embeddings
    else:
        raise Exception(
            f"Error fetching embeddings from Hugging Face API: {response.text}envs:{json.dumps(envs)}"
        )


@router.get("/api/rag_doc/{query}")
async def rag_doc(query: str):
    embed_query = await sentence_embed(query)
    embed_query = np.expand_dims(embed_query, axis=0)
    faiss_index = faiss.read_index("documents.index")
    csv = await load_csv("documents.csv")
    D, I = faiss_index.search(embed_query, 3)
    ragans = "参考情報："
    for idx, score in zip(I[0], D[0]):
        if score < 0.1:
            continue
        ragans += csv[idx] + "/"
    return (
        ragans
        if ragans != "参考情報："
        else "参考情報が見つかりませんでした。もともと持っている知識で回答してください。"
    )


@router.get("/api/rag/{query}")
async def rag(query: str):
    embed_query = await sentence_embed(query)
    embed_query = np.expand_dims(embed_query, axis=0)
    faiss_index = faiss.read_index("documents.index")
    csv = await load_csv("documents.csv")
    D, I = faiss_index.search(embed_query, 3)
    ragans = "参考情報："
    for idx, score in zip(I[0], D[0]):
        if score < 0.1:
            continue
        ragans += csv[idx] + "/"

    if ragans != "参考情報：":
        ragdoc = ragans
    else:
        ragdoc = (
            "参考情報が見つかりませんでした。もともと持っている知識で回答してください。"
        )
    client = AsyncGroq(api_key=GROQ)

    async def generate():
        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"{query}に対して{ragdoc}をもとに答えてください",
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
