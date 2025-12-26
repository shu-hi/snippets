from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
import logging
import func
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from jose import JWTError, jwt

# Setup logging
logging.basicConfig(level=logging.INFO)
router = APIRouter()

# OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Secret key to encode/decode JWT (use a secure key in production)
SECRET_KEY = "mysecretkey"  # Replace this with a more secure secret in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Token expiry time


# Pydantic models
class LoginRequest(BaseModel):
    id: str
    password: str


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
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


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
            access_token = create_access_token(data={"sub": user["serial"]})
            result["data"] = {"access_token": access_token, "token_type": "bearer"}
            result["status"] = "ok"
        else:
            result["status"] = "ng"
            result["err"] = "Invalid credentials"
    except Exception as e:
        result["status"] = "ng"
        result["err"] = str(e)

    return result


@router.get("/api/bar/test")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"msg": "You have access!", "user": current_user}
