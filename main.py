from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "AXEL ROOT OK"}

@app.get("/health")
async def health():
    return {"status": "AXEL HEALTH OK"}

@app.get("/ping")
async def ping():
    return {"status": "AXEL PING OK"}
