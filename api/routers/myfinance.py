from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
import yfinance as yf

router = APIRouter()


def get_stock_info(symbol: str):
    ticker = yf.Ticker(symbol)
    return ticker.info  # この中でHTTPリクエストが走る


@router.get("/api/stock-info/{symbol}")
async def stock_info(symbol: str):
    data = await run_in_threadpool(get_stock_info, symbol)
    return data


@router.get("/api/stock/hi")
def hi():
    return {"message": "hi"}
