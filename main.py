from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Gevel Calculator API - Opening Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Basics
# =========================

def read_upload_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Afbeelding kon niet gelezen worden.")
    return img


def resize_for_processing(image: np.ndarray, max_dim: int = 1800) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    largest = max(h, w)

    if largest <= max_dim:
        return image.copy(), 1.0

    scale = max_dim / float(largest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
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


def non_max_suppression(
    boxes: list[dict[str, Any]],
    iou_threshold: float = 0.28,
) -> list[dict[str, Any]]:
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)
    kept: list[dict[str, Any]] = []

    for box in boxes:
        keep = True
        for existing in kept:
            if iou(box, existing) > iou_threshold:
                keep = False
                break
        if keep:
            kept.append(box)

    return kept


def expand_box(
    x: int,
    y: int,
    w: int,
    h: int,
    pad_x: float,
    pad_y: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    px = int(round(w * pad_x))
    py = int(round(h * pad_y))

    nx = max(0, x - px)
    ny = max(0, y - py)
    nx2 = min(img_w, x + w + px)
    ny2 = min(img_h, y + h + py)

    return nx, ny, nx2 - nx, ny2 - ny


# =========================
# Feature helpers
# =========================

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


def get_rectangularity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    if area <= 0:
        return 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area <= 0:
        return 0.0
    return float(area / rect_area)


def get_inner_outer_stats(gray: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[float, float, float]:
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0 or w < 12 or h < 12:
        return 0.0, 0.0, 0.0

    t = max(4, min(w, h) // 10)

    mask_border = np.zeros((h, w), dtype=np.uint8)
    mask_border[:t, :] = 1
    mask_border[-t:, :] = 1
    mask_border[:, :t] = 1
    mask_border[:, -t:] = 1

    mask_inner = np.ones((h, w), dtype=np.uint8)
    mask_inner[:t, :] = 0
    mask_inner[-t:, :] = 0
    mask_inner[:, :t] = 0
    mask_inner[:, -t:] = 0

    border_vals = roi[mask_border == 1]
    inner_vals = roi[mask_inner == 1]

    if border_vals.size == 0 or inner_vals.size == 0:
        return 0.0, 0.0, 0.0

    border_mean = float(np.mean(border_vals))
    inner_mean = float(np.mean(inner_vals))
    inner_std = float(np.std(inner_vals))
    return border_mean, inner_mean, inner_std


def count_long_lines(edge_roi: np.ndarray, w: int, h: int) -> tuple[int, int]:
    lines = cv2.HoughLinesP(
        edge_roi,
        1,
        np.pi / 180,
        threshold=max(16, min(w, h) // 4),
        minLineLength=max(16, min(w, h) // 3),
        maxLineGap=8,
    )

    if lines is None:
        return 0, 0

    vertical = 0
    horizontal = 0

    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx <= 5 and dy >= h * 0.30:
            vertical += 1
        elif dy <= 5 and dx >= w * 0.30:
            horizontal += 1

    return vertical, horizontal


# =========================
# Gevel/opening pipeline
# =========================

def estimate_facade_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Benadert grote, samenhangende gevelmassa.
    Niet perfect, maar nuttig om lucht/grond/omgeving te reduceren.
    """
    h, w = image_bgr.shape[:2]

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Egaliseren zodat baksteen/crepi/hout wat consistenter wordt
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # Zachte clustering op kleur + licht
    features = np.stack(
        [
            l_eq.reshape(-1).astype(np.float32),
            a.reshape(-1).astype(np.float32),
            b.reshape(-1).astype(np.float32),
        ],
        axis=1,
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )

    k = 4
    _ret, labels, centers = cv2.kmeans(
        features,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    label_img = labels.reshape(h, w)

    # Kies cluster die het meest "gevelachtig" is:
    # groot, centraal, en niet te donker
    best_label = 0
    best_score = -1e9

    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2.0
    cy = h / 2.0

    for i in range(k):
        mask = (label_img == i).astype(np.uint8)
        area = int(mask.sum())
        if area <= 0:
            continue

        ys, xs = np.where(mask > 0)
        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))

        dist_center = ((mean_x - cx) ** 2 + (mean_y - cy) ** 2) ** 0.5
        l_mean = float(np.mean(l_eq[mask > 0]))

        score = 0.0
        score += area * 1.0
        score -= dist_center * 120.0
        score += l_mean * 50.0

        if score > best_score:
            best_score = score
            best_label = i

    facade_mask = (label_img == best_label).astype(np.uint8) * 255

    # Morphologische opschoning
    facade_mask = cv2.morphologyEx(
        facade_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
        iterations=2,
    )
    facade_mask = cv2.morphologyEx(
        facade_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1,
    )

    # Enkel grootste component houden
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(facade_mask, connectivity=8)
    if num_labels <= 1:
        return facade_mask

    best_idx = 1
    best_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_idx = i

    facade_mask = np.where(labels_cc == best_idx, 255, 0).astype(np.uint8)
    return facade_mask


def detect_openings_from_facade(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    original_h, original_w = image_bgr.shape[:2]
    proc_img, scale = resize_for_processing(image_bgr, max_dim=1800)
    proc_h, proc_w = proc_img.shape[:2]
    image_area = proc_w * proc_h

    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    edges = cv2.Canny(gray_eq, 60, 160)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    facade_mask = estimate_facade_mask(proc_img)

    # "Afwijkingen" t.o.v. lokaal gevelgedrag
    local_mean = cv2.blur(gray_eq, (41, 41))
    diff = cv2.absdiff(gray_eq, local_mean)

    # Binnen de gevel zoeken
    diff_in_facade = cv2.bitwise_and(diff, diff, mask=facade_mask)

    _, dev_mask = cv2.threshold(diff_in_facade, 18, 255, cv2.THRESH_BINARY)

    # Ook rechte randen binnen de gevel meenemen
    edge_in_facade = cv2.bitwise_and(edges, edges, mask=facade_mask)

    structure = cv2.morphologyEx(
        edge_in_facade,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )

    opening_mask = cv2.bitwise_or(dev_mask, structure)

    opening_mask = cv2.morphologyEx(
        opening_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=2,
    )
    opening_mask = cv2.morphologyEx(
        opening_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    contours, _ = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.0015:
            continue
        if area > image_area * 0.28:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < 35 or h < 35:
            continue

        rect_area = w * h
        if rect_area <= 0:
            continue

        fill_ratio = area / float(rect_area)
        if fill_ratio < 0.20:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.22 or aspect_ratio > 3.2:
            continue

        border_touch = x <= 2 or y <= 2 or x + w >= proc_w - 2 or y + h >= proc_h - 2
        if border_touch:
            continue

        # Kandidaten moeten grotendeels in de gevel liggen
        roi_facade = facade_mask[y:y + h, x:x + w]
        facade_fraction = float(np.mean(roi_facade > 0)) if roi_facade.size > 0 else 0.0
        if facade_fraction < 0.55:
            continue

        rectangularity = get_rectangularity(cnt)

        roi_edges = edge_in_facade[y:y + h, x:x + w]
        edge_density = get_edge_density(edge_in_facade, x, y, w, h)
        ring_strength = get_ring_strength(
            edge_in_facade,
            x,
            y,
            w,
            h,
            thickness=max(4, min(w, h) // 16),
        )

        border_mean, inner_mean, inner_std = get_inner_outer_stats(gray_eq, x, y, w, h)
        vertical_lines, horizontal_lines = count_long_lines(roi_edges, w, h)

        bottom_ratio = (y + h) / float(proc_h)
        top_ratio = y / float(proc_h)
        height_ratio = h / float(proc_h)
        width_ratio = w / float(proc_w)

        score = 0.0

        # Sterke rechthoek / openingvorm
        score += rectangularity * 2.8
        score += fill_ratio * 1.6

        # Rechte randen / kozijngevoel
        score += edge_density * 4.0
        score += ring_strength * 5.0
        score += min(vertical_lines, 4) * 0.35
        score += min(horizontal_lines, 4) * 0.35

        # Binnenkant wijkt af van rand/gevel
        border_inner_diff = abs(border_mean - inner_mean)
        score += min(2.2, border_inner_diff / 20.0)
        score += min(1.8, inner_std / 20.0)

        # Verhoudingen die vaak bij ramen voorkomen
        preferred_ratios = [0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0]
        ratio_distance = min(abs(aspect_ratio - r) for r in preferred_ratios)
        score += max(0.0, 1.2 - ratio_distance)

        # Te grote vlakken zijn vaak hele zones van muur/schaduw
        if width_ratio > 0.55:
            score -= 1.2
        if height_ratio > 0.70:
            score -= 1.2

        # Zeer lage en hoge smalle opening = eerder deur
        door_like = False
        if bottom_ratio > 0.90 and height_ratio > 0.28 and aspect_ratio < 1.0:
            door_like = True

        # Heel hoog bovenaan tegen dakzone is verdacht
        if top_ratio < 0.03 and h > proc_h * 0.18:
            score -= 1.0

        if score < 2.6:
            continue

        cls = "window"
        if door_like:
            cls = "door"

        candidates.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "score": float(score),
                "class": cls,
            }
        )

    if not candidates:
        return []

    candidates = non_max_suppression(candidates, iou_threshold=0.24)

    # Rescale naar origineel
    inv = 1.0 / scale
    results: list[dict[str, Any]] = []

    for c in candidates:
        x = int(round(c["x"] * inv))
        y = int(round(c["y"] * inv))
        w = int(round(c["width"] * inv))
        h = int(round(c["height"] * inv))

        clipped = clip_box(x, y, w, h, original_w, original_h)
        clipped["class"] = c["class"]
        clipped["_score"] = round(float(c["score"]), 3)
        results.append(clipped)

    # nog eens ruis na rescale weg
    cleaned: list[dict[str, Any]] = []
    min_area = (original_w * original_h) * 0.0025

    for r in results:
        area = r["width"] * r["height"]
        if area < min_area:
            continue
        cleaned.append(r)

    # Sorteer logisch
    cleaned = sorted(cleaned, key=lambda b: (round(b["y"] / 120), b["x"]))
    return cleaned


def split_results(openings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows: list[dict[str, Any]] = []
    doors: list[dict[str, Any]] = []

    for o in openings:
        item = {
            "x": o["x"],
            "y": o["y"],
            "width": o["width"],
            "height": o["height"],
        }
        if o.get("class") == "door":
            doors.append(item)
        else:
            windows.append(item)

    return windows, doors


# =========================
# API
# =========================

@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Gevel opening detection API actief."}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    try:
        raw = await image.read()
        img = read_upload_image(raw)
        img_h, img_w = img.shape[:2]

        openings = detect_openings_from_facade(img)
        windows, doors = split_results(openings)

        response: dict[str, Any] = {
            "success": True,
            "image_width": img_w,
            "image_height": img_h,
            "openings": openings,
            "opening_count": len(openings),
            "windows": windows,
            "window_count": len(windows),
            "doors": doors,
            "door_count": len(doors),
        }

        if len(openings) == 0:
            response["warning"] = "Geen duidelijke openingen gedetecteerd."

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
