from fastapi import FastAPI

main = FastAPI()


@main.get("/api/data")
async def get_data():
    response = {"message": "Hello World!"}
    return response
