from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
import logging
import func
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from jose import jwt, exceptions

# Setup logging
logging.basicConfig(level=logging.INFO)
router = APIRouter()

# OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Secret key to encode/decode JWT (use a secure key in production)
SECRET_KEY = "mysecretkey"  # Replace this with a more secure secret in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 180  # Token expiry time


# Pydantic models
class LoginRequest(BaseModel):
    id: str
    password: str


class HealthCheckRequest(BaseModel):
    date: str


class ShiftAddRequest(BaseModel):
    date: str
    user_serial: str
    start_datetime: str
    end_datetime: str


class HealthCheck(BaseModel):
    attr_1: bool
    attr_2: bool
    attr_3: bool
    attr_4: bool
    attr_5: bool
    attr_6: bool
    attr_7: bool
    attr_8: bool
    attr_9: bool
    attr_10: bool
    attr_11: bool


class HealthCheckAddRequest(BaseModel):
    date: str
    healthCheck: HealthCheck


# Function to create JWT token
def create_access_token(
    data: dict,
    expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})

    # Create token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except exceptions.ExpiredSignatureError:
        raise HTTPException(
            status_code=401, detail="トークンの有効期限が切れています。"
        )
    except exceptions.JWTClaimsError as e:
        raise HTTPException(status_code=401, detail=f"クレームの検証エラー: {e}")
    except exceptions.JWTError as e:
        raise HTTPException(
            status_code=401, detail=f"JWT検証エラー (署名不正など): {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"予期せぬエラー: {e}")


# Endpoint for login
@router.post("/api/bar/login")
async def login(request: LoginRequest):
    id = request.id
    password = request.password
    try:
        result = await run_in_threadpool(
            func.db_pd,
            "select * from public.bar_users where quit_date is null and user_id=%s and user_pass=%s and del_flg=false",
            (id, password),
        )
        result["data"] = result["data"].to_dict(orient="records")

        # If user exists and credentials are correct
        if len(result["data"]) == 1:
            user = result["data"][0]  # Assuming the user data is returned here
            # Create JWT access token
            access_token = create_access_token(
                data={"sub": str(user["serial"])}
            )  # subは文字列じゃないと、decodeの時エラーが出る
            result["data"] = {
                "access_token": access_token,
                "job_class": user["job_class"],
            }
            result["status"] = "ok"
        else:
            result["status"] = "ng"
            result["error"] = "Invalid credentials"
    except Exception as e:
        result["status"] = "ng"
        result["error"] = str(e)

    return result


@router.get("/api/bar/test")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"msg": "You have access!", "user": current_user}


@router.post("/api/bar/get_health_check")
async def get_health_check(
    request: HealthCheckRequest, current_user: dict = Depends(get_current_user)
):
    result = {"status": "ng", "data": {}, "error": "error"}
    try:
        result = await run_in_threadpool(
            func.db_pd,
            "select * from public.check_list where date=%s and del_flg=false",
            (request.date,),
        )
        result["data"] = result["data"].to_dict(orient="records")
    except Exception as e:
        result["status"] = "ng"
        result["error"] = str(e)

    return result


@router.post("/api/bar/set_health_check")
async def set_health_check(
    request: HealthCheckAddRequest, current_user: dict = Depends(get_current_user)
):
    result = {"status": "ng", "data": "", "error": "error"}

    try:
        if current_user is None:
            raise Exception("token expired")
        result = await run_in_threadpool(
            func.pg_exec,
            """insert into public.check_list (user_serial,date,attr_1,attr_2,attr_3,attr_4,attr_5,attr_6,attr_7,attr_8,attr_9,attr_10,attr_11) 
            values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) 
            on conflict(date) 
            do update set 
            user_serial=excluded.user_serial,
            attr_1=excluded.attr_1,
            attr_2=excluded.attr_2,
            attr_3=excluded.attr_3,
            attr_4=excluded.attr_4,
            attr_5=excluded.attr_5,
            attr_6=excluded.attr_6,
            attr_7=excluded.attr_7,
            attr_8=excluded.attr_8,
            attr_9=excluded.attr_9,
            attr_10=excluded.attr_10,
            attr_11=excluded.attr_11""",
            [
                current_user["sub"],
                request.date,
                request.healthCheck.attr_1,
                request.healthCheck.attr_2,
                request.healthCheck.attr_3,
                request.healthCheck.attr_4,
                request.healthCheck.attr_5,
                request.healthCheck.attr_6,
                request.healthCheck.attr_7,
                request.healthCheck.attr_8,
                request.healthCheck.attr_9,
                request.healthCheck.attr_10,
                request.healthCheck.attr_11,
            ],
        )
    except Exception as e:
        result["status"] = "ng"
        result["error"] = str(e)

    return result


@router.post("/api/bar/get_shift")
async def get_shift(
    request: HealthCheckRequest, current_user: dict = Depends(get_current_user)
):
    result = {"status": "ng", "data": {}, "error": "error"}
    try:
        if current_user is None:
            raise Exception("token expired")
        result = await run_in_threadpool(
            func.db_pd,
            """select shift.date as date,shift.start_datetime as start_datetime,shift.end_datetime as end_datetime,user.first_name as first_name from public.shift_table as shift join bar_users as user on shift.user_serial=user.serial where shift.del_flg=false and user.del_flg=false and user.quit_date is null""",
            (),
        )
        result["data"] = result["data"].to_dict(orient="records")
    except Exception as e:
        result["status"] = "ng"
        result["error"] = str(e)

    return result


@router.post("/api/bar/set_shift")
async def set_shiftk(
    request: ShiftAddRequest, current_user: dict = Depends(get_current_user)
):
    result = {"status": "ng", "data": "", "error": "error"}

    try:
        if current_user is None:
            raise Exception("token expired")
        result = await run_in_threadpool(
            func.pg_exec,
            """insert into public.shit_table(date,user_serial,start_datetime,end_datetime,reg_user_serial)values(%s,%s,%s,%s,%s)""",
            [
                request.date,
                request.user_serial,
                request.start_datetime,
                request.end_datetime,
                current_user["sub"],
            ],
        )
    except Exception as e:
        result["status"] = "ng"
        result["error"] = str(e)

    return result


@router.post("/api/bar/get_available")
async def get_available(
    request: HealthCheckRequest, current_user: dict = Depends(get_current_user)
):
    result = {"status": "ng", "data": {}, "error": "error"}
    try:
        if current_user is None:
            raise Exception("token expired")
        result = await run_in_threadpool(
            func.db_pd,
            """select serial,first_name from bar_users where del_flg=false and quit_date is null""",
            (),
        )
        result["data"] = result["data"].to_dict(orient="records")
    except Exception as e:
        result["status"] = "ng"
        result["error"] = str(e)

    return result
