from fastapi import FastAPI
import sys
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import playground, bi, myfinance, search, rag, real_estate, bar

load_dotenv()  # .envファイルの読み込み
if os.getenv("ENV") != "DEV":
    load_dotenv("/etc/secrets/.env")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

main = FastAPI()
main.add_middleware(
    CORSMiddleware,
    allow_origins=["https://react-nu-pink.vercel.app"],  # or ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


main.include_router(playground.router)
main.include_router(bi.router)
main.include_router(myfinance.router)
main.include_router(search.router)
main.include_router(rag.router)
main.include_router(real_estate.router)
main.include_router(bar.router)


@main.get("/api/data")
async def get_data():
    response = {"message": ("env:" + os.getenv("ENV"))}
    return response
