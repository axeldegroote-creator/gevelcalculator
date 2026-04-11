from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Gevel Calculator API - Window Detection Only")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_upload_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded image bytes into an OpenCV BGR image.
    Raises ValueError if the image cannot be decoded.
    """
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Afbeelding kon niet gelezen worden.")
    return img


def resize_for_processing(image: np.ndarray, max_dim: int = 1600) -> tuple[np.ndarray, float]:
    """
    Resize image for faster processing while preserving aspect ratio.
    Returns resized image and scale factor relative to original.
    scale < 1 means image was reduced.
    """
    h, w = image.shape[:2]
    largest = max(h, w)

    if largest <= max_dim:
        return image.copy(), 1.0

    scale = max_dim / float(largest)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def order_and_clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> dict[str, int]:
    """
    Make sure the rectangle stays inside image bounds.
    """
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}


def iou(a: dict[str, int], b: dict[str, int]) -> float:
    """
    Intersection over Union for axis-aligned boxes.
    """
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]

    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def merge_boxes(boxes: list[dict[str, int]], iou_threshold: float = 0.35) -> list[dict[str, int]]:
    """
    Simple NMS-like merging of overlapping rectangles.
    Larger rectangles are preferred.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b["width"] * b["height"], reverse=True)
    kept: list[dict[str, int]] = []

    for box in boxes:
        should_add = True
        for existing in kept:
            if iou(box, existing) > iou_threshold:
                should_add = False
                break
        if should_add:
            kept.append(box)

    return kept


def detect_windows(image_bgr: np.ndarray) -> list[dict[str, int]]:
    """
    Detect likely windows in a facade image using classic computer vision:
    - contrast enhancement
    - edge detection
    - contour extraction
    - geometric filtering
    """
    original_h, original_w = image_bgr.shape[:2]
    proc_img, scale = resize_for_processing(image_bgr, max_dim=1600)
    proc_h, proc_w = proc_img.shape[:2]

    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)

    # Improve contrast locally.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Denoise while preserving edges.
    blur = cv2.bilateralFilter(gray_eq, d=7, sigmaColor=60, sigmaSpace=60)

    # Edge map.
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    # Connect fragmented window borders.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    edges = cv2.dilate(edges, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, int]] = []

    image_area = proc_w * proc_h
    min_area = image_area * 0.003      # ignore tiny regions
    max_area = image_area * 0.45       # ignore giant facade chunks

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 40 or h < 40:
            continue

        aspect_ratio = w / float(h)
        rect_area = w * h
        fill_ratio = area / float(rect_area) if rect_area > 0 else 0.0

        # Windows are often rectangular, medium fill, not too thin.
        if aspect_ratio < 0.35 or aspect_ratio > 3.5:
            continue

        if fill_ratio < 0.35:
            continue

        # Prefer shapes that look somewhat rectangular.
        rectangularity_bonus = 0
        if 4 <= len(approx) <= 8:
            rectangularity_bonus = 1

        # Exclude regions touching too much of the image border.
        border_touch = (
            x <= 3 or y <= 3 or (x + w) >= (proc_w - 3) or (y + h) >= (proc_h - 3)
        )
        if border_touch and rect_area > image_area * 0.08:
            continue

        # Basic score to rank candidates.
        score = 0.0
        score += min(fill_ratio, 1.0) * 2.0
        score += rectangularity_bonus * 1.0

        # Moderate preference for common window proportions.
        ideal_ratios = [0.6, 0.75, 1.0, 1.3, 1.6]
        ratio_score = max(0.0, 1.0 - min(abs(aspect_ratio - r) for r in ideal_ratios))
        score += ratio_score

        # Reject weak candidates.
        if score < 1.2:
            continue

        candidates.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "_score": float(score),
            }
        )

    # Sort by score then keep best non-overlapping boxes.
    candidates = sorted(candidates, key=lambda b: b["_score"], reverse=True)

    stripped = [
        {"x": c["x"], "y": c["y"], "width": c["width"], "height": c["height"]}
        for c in candidates
    ]
    merged = merge_boxes(stripped, iou_threshold=0.3)

    # Scale boxes back to original image size.
    if scale != 1.0:
        scaled_back: list[dict[str, int]] = []
        inv = 1.0 / scale
        for b in merged:
            x = int(round(b["x"] * inv))
            y = int(round(b["y"] * inv))
            w = int(round(b["width"] * inv))
            h = int(round(b["height"] * inv))
            scaled_back.append(order_and_clip_box(x, y, w, h, original_w, original_h))
        merged = scaled_back
    else:
        merged = [
            order_and_clip_box(b["x"], b["y"], b["width"], b["height"], original_w, original_h)
            for b in merged
        ]

    # Sort left-to-right, then top-to-bottom for stable frontend display.
    merged = sorted(merged, key=lambda b: (round(b["y"] / 80), b["x"]))

    return merged


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Window detection API actief."}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    """
    Accepts one uploaded facade image and returns detected windows.
    """
    try:
        raw = await image.read()
        img = read_upload_image(raw)
        img_h, img_w = img.shape[:2]

        windows = detect_windows(img)

        response: dict[str, Any] = {
            "success": True,
            "image_width": img_w,
            "image_height": img_h,
            "reference_detected": False,
            "marker": None,
            "windows": windows,
            "window_count": len(windows),
        }

        if len(windows) == 0:
            response["warning"] = "Geen ramen gedetecteerd."

        return JSONResponse(content=response)

    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(exc)},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Interne fout: {str(exc)}"},
        )
