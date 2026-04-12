from __future__ import annotations

import os
import tempfile
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient

app = FastAPI(title="Gevel Calculator API - Roboflow")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROBOFLOW_API_URL = "https://detect.roboflow.com"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "VtnRHY1p1oVlLMgwj6zA")
ROBOFLOW_WORKSPACE = "axels-workspace-lm5wm"
ROBOFLOW_WORKFLOW_ID = "find-fluorescent-rulers-and-windows"
MARKER_REAL_HEIGHT_CM = 40.0

client = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY,
)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def extract_predictions(result: Any) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        nonlocal predictions

        if isinstance(node, list):
            for item in node:
                walk(item)
            return

        if not isinstance(node, dict):
            return

        for key in ("predictions", "detections"):
            maybe = node.get(key)
            if isinstance(maybe, list):
                for item in maybe:
                    if isinstance(item, dict):
                        predictions.append(item)

        for value in node.values():
            if isinstance(value, (dict, list)):
                walk(value)

    walk(result)
    return predictions


def prediction_to_box(pred: dict[str, Any]) -> dict[str, Any]:
    x_center = float(pred.get("x", 0))
    y_center = float(pred.get("y", 0))
    width = float(pred.get("width", 0))
    height = float(pred.get("height", 0))

    x = safe_int(x_center - width / 2)
    y = safe_int(y_center - height / 2)
    w = safe_int(width)
    h = safe_int(height)

    return {
        "x": x,
        "y": y,
        "width": max(0, w),
        "height": max(0, h),
        "confidence": round(float(pred.get("confidence", 0) or 0), 4),
        "label": str(pred.get("class") or pred.get("label") or "").strip(),
    }


def is_window_label(label: str) -> bool:
    label = label.lower()
    return any(word in label for word in ["window", "raam", "windows"])


def is_marker_label(label: str) -> bool:
    label = label.lower()
    return any(word in label for word in ["ruler", "marker", "scale", "lat"])


def add_measurements(windows: list[dict[str, Any]], marker: dict[str, Any] | None):
    if not marker or marker["height"] <= 0:
        return windows, None

    px_per_cm = marker["height"] / MARKER_REAL_HEIGHT_CM
    if px_per_cm <= 0:
        return windows, None

    measured = []
    for w in windows:
        width_cm = round(w["width"] / px_per_cm, 1)
        height_cm = round(w["height"] / px_per_cm, 1)
        area_m2 = round((width_cm * height_cm) / 10000, 2)

        item = dict(w)
        item["width_cm"] = width_cm
        item["height_cm"] = height_cm
        item["area_m2"] = area_m2
        measured.append(item)

    return measured, px_per_cm


@app.get("/")
async def root():
    return {"status": "ok", "message": "API draait"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: Request):
    temp_path: str | None = None

    try:
        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" not in content_type:
            return JSONResponse(
                {
                    "error": f"Verkeerd content-type: {content_type}. Verwacht multipart/form-data."
                },
                status_code=400,
            )

        form = await request.form()

        upload = form.get("image") or form.get("file") or form.get("photo")

        if upload is None:
            return JSONResponse(
                {
                    "error": "Geen afbeelding ontvangen. Verwachte veldnaam: image, file of photo."
                },
                status_code=400,
            )

        filename = getattr(upload, "filename", "upload.jpg") or "upload.jpg"
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"

        file_bytes = await upload.read()
        if not file_bytes:
            return JSONResponse(
                {"error": "Bestand is leeg."},
                status_code=400,
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        result = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={"image": temp_path},
            use_cache=True,
        )

        predictions = extract_predictions(result)

        windows = []
        marker_candidates = []

        for pred in predictions:
            box = prediction_to_box(pred)
            label = box["label"]

            if box["width"] <= 0 or box["height"] <= 0:
                continue

            if is_window_label(label):
                windows.append(box)
            elif is_marker_label(label):
                marker_candidates.append(box)

        marker = None
        if marker_candidates:
            marker = sorted(
                marker_candidates,
                key=lambda m: (m.get("confidence", 0), m["height"]),
                reverse=True,
            )[0]

        windows = sorted(windows, key=lambda w: (w["x"], w["y"]))
        windows, px_per_cm = add_measurements(windows, marker)

        response = {
            "reference_detected": marker is not None,
            "marker": marker,
            "windows": windows,
            "pixels_per_cm": round(px_per_cm, 4) if px_per_cm else None,
            "raw_prediction_count": len(predictions),
            "warning": None,
        }

        if marker is None:
            response["warning"] = "Marker niet gevonden. Afmetingen in cm zijn niet berekend."
        elif not windows:
            response["warning"] = "Marker gevonden, maar geen ramen gedetecteerd."

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse(
            {
                "error": str(e),
                "reference_detected": False,
                "marker": None,
                "windows": [],
            },
            status_code=500,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
