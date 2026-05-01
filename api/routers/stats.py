from fastapi import APIRouter, UploadFile
import logging
import classes.dbClass as sC
import classes.FrequentistClass as FC
import json
from pydantic import BaseModel
import pandas as pd
import traceback
# Setup logging
logging.basicConfig(level=logging.INFO)
router = APIRouter()


# Pydantic models
class InitRequest(BaseModel):
    csv_path:list[str]
    gss_path:list[str]
    query:str
class StatsRequest(BaseModel):
    type:str
    target:str
    expectation:float
    both_side:bool
    query_1:str
    query_2:str

db=sC.UnifiedData()
db.connect()
db.attach_postgres()

@router.post("/api/stats/init")
async def init(request: InitRequest):
    try:
        
        for i,path in enumerate(request.csv_path):
            if(len(path)>2):
                db.load_csv(i,path)
        for i,path in enumerate(request.gss_path):
            if(len(path)>2):
                db.load_sheet(i,path)
        
        json_str = db.query('show all tables;').to_json(orient="records", date_format="iso")#fastapiのjson変換が怪しいので
        data = json.loads(json_str)
        return {'status':'ok','data':data}
    except Exception as e:
        return {'status':'ng','data':[],'error':traceback.format_exc()}
@router.post("/api/stats/getData")
async def getData(request: InitRequest):
    try:
        json_str = db.query(request.query).to_json(orient="records", date_format="iso")#fastapiのjson変換が怪しいので
        data = json.loads(json_str)
        return {'status':'ok','data':data}
    except Exception as e:
        return {'status':'ng','data':[],'error':traceback.format_exc()}


@router.post("/api/stats/getStats")
async def getStats(request:StatsRequest):
    try:
        df_1=db.query(request.query_1)
        df_2=db.query(request.query_2)
        result=statsMain(request,df_1,df_2)
        return {'status':'ok','data':result['data'],'blob':result['blob']}
    except Exception as e:
        return {'status':'ng','data':[],'error':traceback.format_exc()}


@router.post("/api/stats/test")
async def test():
    return 'hello world'



def statsMain(request:StatsRequest,df_1:pd.DataFrame,df_2:pd.DataFrame):
    match request.type:
        case 't_1':
            StatsManager=FC.Frequentist(df_1,df_2,request.target)
            standard_error, ci_low, ci_high, p_value,img_base64= StatsManager.t_1(request.expectation,request.both_side)
            return {'data':{'standard_error':standard_error, 'ci_low': ci_low, 'ci_high':ci_high, 'p_value':p_value},'blob':img_base64}
        case 'paired_t':
            StatsManager=FC.Frequentist(df_1,df_2,request.target)
        case 'unpaired_t':
            StatsManager=FC.Frequentist(df_1,df_2,request.target)
            standard_error, ci_low, ci_high, p_value,img_base64= StatsManager.unpaired_t(request.both_side)
            return {'data':{'standard_error':standard_error, 'ci_low': ci_low, 'ci_high':ci_high, 'p_value':p_value},'blob':img_base64}