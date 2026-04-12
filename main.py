import os
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient

app = FastAPI()

# Allow requests from your frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your specific domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
# PRO TIP: Add ROBOFLOW_API_KEY in Render's "Environment" tab
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE_NAME = "axels-workspace-lm5wm"
WORKFLOW_ID = "home-net-surface-area-measurement-1776016948034"

# Initialize the Roboflow client
# We use the hosted API so Render doesn't need a GPU
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

@app.get("/")
def home():
    return {"status": "Wall Measurement API is running"}

@app.post("/measure-wall")
async def measure_wall(file: UploadFile = file(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Roboflow API Key not configured on server.")

    try:
        # 1. Read the uploaded file into memory
        image_bytes = await file.read()

        # 2. Run the Roboflow Workflow
        # This sends the image to Roboflow, runs the segmentation, 
        # and performs the area calculation using the 400mm ruler.
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": image_bytes}
        )

        # 3. Extract outputs defined in your workflow
        # 'result' is a list (one per image). We take the first one.
        workflow_output = result[0]
        
        net_area = workflow_output.get("net_wall_area_m2", 0)
        annotated_image_b64 = workflow_output.get("annotated_image")

        return {
            "success": True,
            "net_area_m2": round(net_area, 3),
            "annotated_image": annotated_image_b64, # This is the base64 string for the UI
            "unit": "square meters"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
