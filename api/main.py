from fastapi import FastAPI
import sys
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from routers import playground, bi

load_dotenv()  # .envファイルの読み込み
if os.getenv("TSV2_DB_HOST") != "10.0.146":
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


@main.get("/api/data")
async def get_data():
    response = {"message": ("TSV2_DB_PORT:" + os.getenv("TSV2_DB_PORT"))}
    return response
