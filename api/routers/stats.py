from fastapi import APIRouter, UploadFile
import logging
import classes.dbClass as sC
import os
from pydantic import BaseModel
import traceback
# Setup logging
logging.basicConfig(level=logging.INFO)
router = APIRouter()


# Pydantic models
class StatsRequest(BaseModel):
    csv_path:list[str]
    gss_path:list[str]
    query:str


@router.post("/api/stats/getData")
async def store(request: StatsRequest):
    try:
        db=sC.UnifiedData()
        for i,path in enumerate(request.csv_path):
            if(len(path)>2):
                db.load_csv(i,path)
        for i,path in enumerate(request.gss_path):
            if(len(path)>2):
                db.load_sheet(i,path)
        db.attach_postgres()
        return {'status':'ok','data':db.query(request.query)}
    except Exception as e:
        return {'status':'ng','data':[],'error':traceback.format_exc()}

@router.post("/api/stats/test")
async def test():
    return 'hello world'