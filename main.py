from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "API draait"}

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    marker: UploadFile = File(...)
):
    return {
        "reference_detected": True,
        "scale_cm_per_pixel": 0.21,
        "marker": {
            "x": 120,
            "y": 340,
            "width": 28,
            "height": 190
        },
        "windows": [
            {
                "x": 420,
                "y": 180,
                "width": 360,
                "height": 520,
                "width_cm": 75.6,
                "height_cm": 109.2,
                "area_cm2": 8255.52,
                "area_m2": 0.826,
                "confidence": 0.94
            }
        ],
        "warning": None
    }
