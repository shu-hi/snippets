from fastapi import APIRouter, UploadFile
import logging
import classes.statsClass as sC
import os
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
router = APIRouter()


# Pydantic models
class StatsRequest(BaseModel):
    csv_path:str|None
    gss_path:str|None
    query:str


@router.post("/api/stats/getData")
async def store(request: StatsRequest):
    try:
        db=sC.UnifiedData()
        if request.csv_path:
            db.load_csv(request.csv_path)
        if request.gss_path:
            db.load_sheet(request.gss_path)
        db.attach_postgres()
        return {'status':'ok','data':db.query(request.query)}
    except Exception as e:
        return {'status':'ng','data':[],'error':repr(e)}

@router.post("/api/stats/test")
async def test():
    return 'hello world'