import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient

app = FastAPI()

# CORS (nodig voor CodePen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Roboflow client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ.get("ROBOFLOW_API_KEY")  # zet deze in Render
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    
    # 🔥 HIER komt de fix
    image_bytes = await file.read()

    print(f"Image size: {len(image_bytes)} bytes")  # debug

    # 🔥 HIER gebruik je die in je workflow
    result = client.run_workflow(
        workspace_name="axels-workspace-lm5wm",
        workflow_id="find-windows-2",
        images={
            "image": image_bytes
        },
        use_cache=True
    )

    print(result)  # debug

    return result
