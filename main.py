from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Gevel Calculator API - Facade Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MARKER_REAL_HEIGHT_CM = 40.0


def read_upload_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Afbeelding kon niet gelezen worden.")
    return img


def resize_for_processing(image: np.ndarray, max_dim: int = 1600) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def clamp_int(value: float) -> int:
    return int(round(value))


def safe_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def estimate_marker_hsv_range(marker_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Leidt een HSV-range af uit de geüploade markerfoto.
    We nemen de meest verzadigde/geel-groene pixels als referentie.
    """
    hsv = cv2.cvtColor(marker_img, cv2.COLOR_BGR2HSV)

    # Fluokleuren zijn meestal hoge saturatie + hoge helderheid
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    strong_mask = (sat > 80) & (val > 80)
    pixels = hsv[strong_mask]

    if len(pixels) < 50:
        # fallback-range voor fluogeel / geelgroen
        lower = np.array([18, 70, 70], dtype=np.uint8)
        upper = np.array([48, 255, 255], dtype=np.uint8)
        return lower, upper

    # Focus op dominante hue
    hues = pixels[:, 0]
    median_h = int(np.median(hues))

    lower_h = max(0, median_h - 12)
    upper_h = min(179, median_h + 12)

    lower = np.array([lower_h, 60, 60], dtype=np.uint8)
    upper = np.array([upper_h, 255, 255], dtype=np.uint8)
    return lower, upper


def detect_marker(image_bgr: np.ndarray, marker_img_bgr: np.ndarray | None) -> dict[str, Any] | None:
    """
    Detecteert een smalle verticale fluogele marker in de geüploade foto.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    if marker_img_bgr is not None:
        lower, upper = estimate_marker_hsv_range(marker_img_bgr)
    else:
        lower = np.array([18, 70, 70], dtype=np.uint8)
        upper = np.array([48, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    img_h, img_w = image_bgr.shape[:2]
    image_area = img_h * img_w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.0002:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0 or w <= 0:
            continue

        aspect = h / max(w, 1)
        fill_ratio = area / max(w * h, 1)

        # Marker moet eerder smal en verticaal zijn
        score = 0.0
        score += min(area / image_area * 3000, 8.0)
        score += min(aspect, 12.0) * 1.5
        score += fill_ratio * 3.0

        # lichte bonus als hij niet volledig bovenaan of volledig onderaan staat
        cy = y + h / 2
        center_bonus = 1.0 - abs(cy - img_h / 2) / (img_h / 2)
        score += max(center_bonus, 0) * 1.5

        if aspect < 2.5:
            score -= 6.0

        if score > best_score:
            best_score = score
            best = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area_px": float(area),
                "aspect_ratio": round(float(aspect), 2),
                "score": round(float(score), 2),
                "mask": mask,
            }

    return best


def detect_brick_facade(image_bgr: np.ndarray, marker_box: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Ruwe detectie van de gevel.
    Doel: een bruikbare eerste MVP, niet exact.
    """
    img_h, img_w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Baksteen zit vaak in rood/oranje/bruin gebied, maar niet altijd.
    # Daarom combineren we kleur + textuur/edges.
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    brick_color_mask = (
        (((h >= 3) & (h <= 25)) | ((h >= 170) & (h <= 179))) &
        (s >= 40) &
        (v >= 40) &
        (v <= 230)
    )

    # Lokale textuur via Laplacian/Canny
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    texture_mask = edges > 0

    # Gecombineerde facade-kans
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    combined[brick_color_mask | texture_mask] = 255

    # Marker liefst ook mee opnemen als deel van de gevel
    if marker_box is not None:
        mx, my, mw, mh = marker_box["x"], marker_box["y"], marker_box["width"], marker_box["height"]
        pad_x = int(max(20, mw * 4))
        pad_y = int(max(20, mh * 2))
        x1 = max(0, mx - pad_x)
        y1 = max(0, my - pad_y)
        x2 = min(img_w, mx + mw + pad_x)
        y2 = min(img_h, my + mh + pad_y)
        combined[y1:y2, x1:x2] = 255

    # Sluit gaten zodat baksteenpatronen samensmelten tot één regio
    kernel_close1 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close1, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close2, iterations=1)

    # Verwijder kleine stukken
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    best = None
    best_score = -1.0
    image_area = img_h * img_w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.03:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = max(w * h, 1)
        fill_ratio = area / rect_area

        # vermoedelijke geveloppervlakte: groot, redelijk rechthoekig
        score = 0.0
        score += min(area / image_area * 30, 18.0)
        score += fill_ratio * 5.0

        # Bonus als marker binnen of dicht bij de contour zit
        if marker_box is not None:
            mx = marker_box["x"] + marker_box["width"] / 2
            my = marker_box["y"] + marker_box["height"] / 2
            if x <= mx <= x + w and y <= my <= y + h:
                score += 6.0

        # Straf wanneer de contour bijna de hele foto is
        if area > image_area * 0.92:
            score -= 6.0

        # gevel is meestal hoger dan 20% van beeldhoogte en breder dan 20% van beeldbreedte
        if w < img_w * 0.2 or h < img_h * 0.2:
            score -= 5.0

        if score > best_score:
            best_score = score
            best = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area_px": float(area),
                "fill_ratio": round(float(fill_ratio), 3),
                "score": round(float(score), 2),
            }

    if best is None:
        return None

    # Maak de box iets strakker via projecties binnen ROI
    x, y, w, h = best["x"], best["y"], best["width"], best["height"]
    roi = combined[y:y + h, x:x + w]

    col_sum = np.sum(roi > 0, axis=0)
    row_sum = np.sum(roi > 0, axis=1)

    col_thresh = max(5, int(h * 0.12))
    row_thresh = max(5, int(w * 0.12))

    valid_cols = np.where(col_sum > col_thresh)[0]
    valid_rows = np.where(row_sum > row_thresh)[0]

    if len(valid_cols) > 0 and len(valid_rows) > 0:
        x1 = x + int(valid_cols[0])
        x2 = x + int(valid_cols[-1])
        y1 = y + int(valid_rows[0])
        y2 = y + int(valid_rows[-1])

        new_x = x1
        new_y = y1
        new_w = max(1, x2 - x1 + 1)
        new_h = max(1, y2 - y1 + 1)

        new_x, new_y, new_w, new_h = safe_box(new_x, new_y, new_w, new_h, img_w, img_h)

        best["x"] = new_x
        best["y"] = new_y
        best["width"] = new_w
        best["height"] = new_h

    return best


def build_response(
    marker: dict[str, Any] | None,
    facade: dict[str, Any] | None,
    scale_x: float,
    scale_y: float,
    original_shape: tuple[int, int, int],
    processing_scale: float,
) -> dict[str, Any]:
    img_h, img_w = original_shape[:2]

    result: dict[str, Any] = {
        "reference_detected": marker is not None,
        "marker": None,
        "facade": None,
        "facade_width_cm": None,
        "facade_height_cm": None,
        "facade_area_m2": None,
        "windows": [],  # compatibel houden met je huidige frontend
        "warning": None,
        "debug": {
            "image_width_px": img_w,
            "image_height_px": img_h,
            "processing_scale": round(float(processing_scale), 4),
        },
    }

    if marker is not None:
        result["marker"] = {
            "x": clamp_int(marker["x"] * scale_x),
            "y": clamp_int(marker["y"] * scale_y),
            "width": clamp_int(marker["width"] * scale_x),
            "height": clamp_int(marker["height"] * scale_y),
            "height_cm_real": MARKER_REAL_HEIGHT_CM,
        }

    if facade is not None:
        fx = clamp_int(facade["x"] * scale_x)
        fy = clamp_int(facade["y"] * scale_y)
        fw = clamp_int(facade["width"] * scale_x)
        fh = clamp_int(facade["height"] * scale_y)

        result["facade"] = {
            "x": fx,
            "y": fy,
            "width": fw,
            "height": fh,
        }

        if marker is not None and marker["height"] > 0:
            cm_per_px = MARKER_REAL_HEIGHT_CM / float(marker["height"])
            facade_width_cm = facade["width"] * cm_per_px
            facade_height_cm = facade["height"] * cm_per_px
            facade_area_m2 = (facade_width_cm * facade_height_cm) / 10000.0

            result["facade_width_cm"] = round(float(facade_width_cm), 1)
            result["facade_height_cm"] = round(float(facade_height_cm), 1)
            result["facade_area_m2"] = round(float(facade_area_m2), 2)
        else:
            result["warning"] = "Marker niet betrouwbaar genoeg gedetecteerd om schaal te berekenen."

    if marker is None:
        result["warning"] = "Marker niet gedetecteerd."
    elif facade is None:
        result["warning"] = "Gevel niet gedetecteerd."

    return result


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Gevel Calculator API draait."}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    marker: UploadFile = File(...),
) -> JSONResponse:
    try:
        image_bytes = await image.read()
        marker_bytes = await marker.read()

        original_img = read_upload_image(image_bytes)
        marker_img = read_upload_image(marker_bytes)

        proc_img, proc_scale = resize_for_processing(original_img, max_dim=1600)
        proc_marker_img, _ = resize_for_processing(marker_img, max_dim=600)

        detected_marker = detect_marker(proc_img, proc_marker_img)
        detected_facade = detect_brick_facade(proc_img, detected_marker)

        # van processing-image terug naar originele pixelcoördinaten
        inv_scale = 1.0 / proc_scale
        result = build_response(
            marker=detected_marker,
            facade=detected_facade,
            scale_x=inv_scale,
            scale_y=inv_scale,
            original_shape=original_img.shape,
            processing_scale=proc_scale,
        )

        return JSONResponse(content=result)

    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "reference_detected": False, "windows": []},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Interne fout: {str(exc)}",
                "reference_detected": False,
                "windows": [],
            },
        )
