from fastapi import APIRouter, UploadFile
import logging
import classes.holo_class as hC
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
router = APIRouter()


# Pydantic models
class StoreRequest(BaseModel):
    mediatype: str
    video: UploadFile | None
    right: UploadFile | None
    down: UploadFile | None
    left: UploadFile | None
    up: UploadFile | None


class ShowRequest(BaseModel):
    password: str


@router.post("/api/holo/store")
async def store(request: StoreRequest):
    holo_factory = hC.holoFactory()
    holo = holo_factory.create(request)
    result = holo.store()
    return result


@router.post("/api/holo/show")
async def show(request: ShowRequest):
    password = request.password
