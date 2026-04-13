import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://detect.roboflow.com")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "axels-workspace-lm5wm")
ROBOFLOW_WORKFLOW_ID = os.environ.get("ROBOFLOW_WORKFLOW_ID", "find-windows-2")

if not ROBOFLOW_API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY ontbreekt in de environment variables.")

app = FastAPI(title="Gevel Calculator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY,
)

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Geen bestand ontvangen.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload een geldig image-bestand.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Leeg bestand ontvangen.")

    print("BESTAND ONTVANGEN")
    print("filename:", file.filename)
    print("content_type:", file.content_type)
    print("size:", len(image_bytes))

    try:
        result: Any = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={"image": image_bytes},
            use_cache=True,
        )
        print("ROBOFLOW RESULT:", result)
        return JSONResponse(content=result)
    except Exception as exc:
        print("ROBOFLOW ERROR:", str(exc))
        raise HTTPException(status_code=500, detail=f"Roboflow fout: {str(exc)}") from exc
