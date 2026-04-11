from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Gevel Calculator API - Window Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_upload_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Afbeelding kon niet gelezen worden.")
    return img


def resize_for_processing(image: np.ndarray, max_dim: int = 1600) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    largest = max(h, w)

    if largest <= max_dim:
        return image.copy(), 1.0

    scale = max_dim / float(largest)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> dict[str, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}


def iou(a: dict[str, int], b: dict[str, int]) -> float:
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

    if inter_area <= 0:
        return 0.0

    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def merge_boxes(boxes: list[dict[str, int]], iou_threshold: float = 0.25) -> list[dict[str, int]]:
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b["width"] * b["height"], reverse=True)
    kept: list[dict[str, int]] = []

    for box in boxes:
        keep = True
        for existing in kept:
            if iou(box, existing) > iou_threshold:
                keep = False
                break
        if keep:
            kept.append(box)

    return kept


def expand_box(x: int, y: int, w: int, h: int, pad_x: float, pad_y: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    px = int(round(w * pad_x))
    py = int(round(h * pad_y))

    nx = max(0, x - px)
    ny = max(0, y - py)
    nx2 = min(img_w, x + w + px)
    ny2 = min(img_h, y + h + py)

    return nx, ny, nx2 - nx, ny2 - ny


def get_edge_density(edge_img: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    roi = edge_img[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi > 0))


def get_ring_strength(edge_img: np.ndarray, x: int, y: int, w: int, h: int, thickness: int = 6) -> float:
    roi = edge_img[y:y + h, x:x + w]
    if roi.size == 0 or w <= 2 * thickness or h <= 2 * thickness:
        return 0.0

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:thickness, :] = 1
    mask[-thickness:, :] = 1
    mask[:, :thickness] = 1
    mask[:, -thickness:] = 1

    values = roi[mask == 1]
    if values.size == 0:
        return 0.0

    return float(np.mean(values > 0))


def detect_windows(image_bgr: np.ndarray) -> list[dict[str, int]]:
    original_h, original_w = image_bgr.shape[:2]
    proc_img, scale = resize_for_processing(image_bgr, max_dim=1600)
    proc_h, proc_w = proc_img.shape[:2]

    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # 1) Edge map: belangrijk voor kozijnen / dagkanten / raamkaders
    edges = cv2.Canny(blur, 50, 150)

    edges_closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )

    # 2) Donkere regio's: goed voor glas/openingen
    dark_mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )

    image_area = proc_w * proc_h

    candidates: list[dict[str, Any]] = []

    def add_candidate(x: int, y: int, w: int, h: int, source: str) -> None:
        if w < 35 or h < 35:
            return

        rect_area = w * h
        if rect_area < image_area * 0.003:
            return
        if rect_area > image_area * 0.35:
            return

        aspect_ratio = w / float(h)

        # Laat zowel smalle staande ramen als bredere ramen toe
        if aspect_ratio < 0.22 or aspect_ratio > 2.8:
            return

        # Deuren eruit duwen:
        # heel hoog én bijna tot op de onderrand = waarschijnlijk deur
        bottom_ratio = (y + h) / float(proc_h)
        height_ratio = h / float(proc_h)
        if bottom_ratio > 0.94 and height_ratio > 0.45:
            return

        # ROI-analyse
        roi_gray = gray_eq[y:y + h, x:x + w]
        if roi_gray.size == 0:
            return

        mean_intensity = float(np.mean(roi_gray))
        std_intensity = float(np.std(roi_gray))

        edge_density = get_edge_density(edges_closed, x, y, w, h)
        ring_strength = get_ring_strength(edges_closed, x, y, w, h, thickness=max(4, min(w, h) // 18))

        # Binnen extra box kijken voor omlijsting/structuur
        ex, ey, ew, eh = expand_box(x, y, w, h, 0.08, 0.08, proc_w, proc_h)
        expanded_edge_density = get_edge_density(edges_closed, ex, ey, ew, eh)

        score = 0.0

        # Ramen hebben meestal duidelijke randen
        score += edge_density * 4.0
        score += ring_strength * 5.0
        score += expanded_edge_density * 2.0

        # Contrast binnenin helpt
        score += min(std_intensity / 18.0, 2.5)

        # Donkerte kan helpen, maar is niet verplicht
        if mean_intensity < 160:
            score += (160.0 - mean_intensity) / 40.0

        # Typische raamverhoudingen
        preferred_ratios = [0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0]
        ratio_distance = min(abs(aspect_ratio - r) for r in preferred_ratios)
        score += max(0.0, 1.3 - ratio_distance)

        # Boxen die tegen de rand hangen zijn meestal geen ramen
        border_touch = (
            x <= 2 or y <= 2 or x + w >= proc_w - 2 or y + h >= proc_h - 2
        )
        if border_touch:
            score -= 1.0

        # Nog een penalty voor "deurachtig"
        if bottom_ratio > 0.88 and height_ratio > 0.35 and aspect_ratio < 0.9:
            score -= 1.2

        # Hele lichte, weinig contrastrijke vlakken zijn meestal geen raam
        if mean_intensity > 215 and std_intensity < 18:
            score -= 1.5

        if score < 2.2:
            return

        candidates.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "_score": float(score),
                "_source": source,
            }
        )

    # Proposal set A: contouren uit edge map, met interne hiërarchie behouden
    contours_edges, hierarchy_edges = cv2.findContours(
        edges_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy_edges is not None:
        hierarchy_edges = hierarchy_edges[0]

    for idx, cnt in enumerate(contours_edges):
        area = cv2.contourArea(cnt)
        if area < image_area * 0.0015:
            continue
        if area > image_area * 0.30:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        approx_points = len(approx)

        # Interne contouren zijn vaak interessanter dan de buitenste gevelcontour
        parent = hierarchy_edges[idx][3] if hierarchy_edges is not None else -1
        child = hierarchy_edges[idx][2] if hierarchy_edges is not None else -1

        base_bonus = 0.0
        if parent != -1:
            base_bonus += 0.6
        if child != -1:
            base_bonus += 0.3
        if 4 <= approx_points <= 10:
            base_bonus += 0.6

        x2, y2, w2, h2 = expand_box(x, y, w, h, 0.04, 0.04, proc_w, proc_h)
        before_count = len(candidates)
        add_candidate(x2, y2, w2, h2, "edges")
        if len(candidates) > before_count:
            candidates[-1]["_score"] += base_bonus

    # Proposal set B: donkere regio's als extra ingang
    contours_dark, hierarchy_dark = cv2.findContours(
        dark_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy_dark is not None:
        hierarchy_dark = hierarchy_dark[0]

    for idx, cnt in enumerate(contours_dark):
        area = cv2.contourArea(cnt)
        if area < image_area * 0.0015:
            continue
        if area > image_area * 0.25:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        rect_area = w * h
        fill_ratio = area / float(rect_area) if rect_area > 0 else 0.0
        if fill_ratio < 0.18:
            continue

        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        approx_points = len(approx)

        base_bonus = 0.0
        if 4 <= approx_points <= 10:
            base_bonus += 0.5
        if fill_ratio > 0.45:
            base_bonus += 0.4

        x2, y2, w2, h2 = expand_box(x, y, w, h, 0.08, 0.08, proc_w, proc_h)
        before_count = len(candidates)
        add_candidate(x2, y2, w2, h2, "dark")
        if len(candidates) > before_count:
            candidates[-1]["_score"] += base_bonus

    if not candidates:
        return []

    # Beste eerst
    candidates = sorted(candidates, key=lambda c: c["_score"], reverse=True)

    raw_boxes = [
        {
            "x": c["x"],
            "y": c["y"],
            "width": c["width"],
            "height": c["height"],
        }
        for c in candidates
    ]

    merged = merge_boxes(raw_boxes, iou_threshold=0.22)

    final_boxes: list[dict[str, int]] = []
    inv = 1.0 / scale

    for b in merged:
        x = int(round(b["x"] * inv))
        y = int(round(b["y"] * inv))
        w = int(round(b["width"] * inv))
        h = int(round(b["height"] * inv))
        final_boxes.append(clip_box(x, y, w, h, original_w, original_h))

    # Kleine/ruisachtige boxen na rescale nog eens filteren
    cleaned: list[dict[str, int]] = []
    for b in final_boxes:
        area = b["width"] * b["height"]
        if area < (original_w * original_h) * 0.003:
            continue
        cleaned.append(b)

    # Sorteer boven-naar-beneden en links-naar-rechts
    cleaned = sorted(cleaned, key=lambda b: (round(b["y"] / 100), b["x"]))

    return cleaned


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Window detection API actief."}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    try:
        raw = await image.read()
        img = read_upload_image(raw)
        img_h, img_w = img.shape[:2]

        windows = detect_windows(img)

        response: dict[str, Any] = {
            "success": True,
            "image_width": img_w,
            "image_height": img_h,
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
