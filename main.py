import base64
import math
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


MARKER_REAL_HEIGHT_CM = 40.0

app = FastAPI(title="Gevelcalculator API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later beperken tot je echte frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Kon afbeelding niet decoderen.")
    return img


def encode_bgr_to_base64(img: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("Kon afbeelding niet encoderen.")
    return base64.b64encode(buffer).decode("utf-8")


def resize_if_needed(img: np.ndarray, max_width: int = 1400) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= max_width:
        return img.copy(), 1.0
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def yellow_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([18, 70, 70], dtype=np.uint8)
    upper1 = np.array([42, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def find_vertical_yellow_candidates(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[int, int, int, int]] = []

    img_h, img_w = mask.shape[:2]
    min_area = max(300, int((img_h * img_w) * 0.0002))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue

        aspect = h / max(w, 1)
        if aspect < 2.0:
            continue

        if h < 30:
            continue

        candidates.append((x, y, w, h))

    return candidates


def extract_marker_template(marker_bgr: np.ndarray) -> Dict[str, Any]:
    mask = yellow_mask(marker_bgr)
    candidates = find_vertical_yellow_candidates(mask)

    if not candidates:
        raise ValueError("Geen bruikbare gele marker gevonden in markerfoto.")

    # Neem de grootste kandidaat
    best = max(candidates, key=lambda r: r[2] * r[3])
    x, y, w, h = best

    roi_mask = mask[y:y + h, x:x + w].copy()
    roi_bgr = marker_bgr[y:y + h, x:x + w].copy()

    # Normaliseer template naar vaste grootte voor vergelijking
    template_mask = cv2.resize(roi_mask, (40, 240), interpolation=cv2.INTER_AREA)

    # Zoek zwarte patronen in de markerfoto
    black_features = detect_black_marker_features(roi_bgr)

    return {
        "bbox": best,
        "template_mask": template_mask,
        "aspect_ratio": h / max(w, 1),
        "black_features": black_features,
    }


def detect_black_marker_features(marker_roi_bgr: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(marker_roi_bgr, cv2.COLOR_BGR2GRAY)
    _, black = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)

    h, w = black.shape[:2]
    contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corner_hits = 0
    middle_line_found = False

    corner_regions = [
        (0, 0, int(w * 0.35), int(h * 0.2)),                 # linksboven
        (int(w * 0.65), 0, w, int(h * 0.2)),                 # rechtsboven
        (0, int(h * 0.8), int(w * 0.35), h),                 # linksonder
        (int(w * 0.65), int(h * 0.8), w, h),                 # rechtsonder
    ]

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < 8:
            continue

        # Middenlijn: horizontaal element in middenzone
        if cw > w * 0.45 and ch < h * 0.12:
            cy = y + ch / 2
            if h * 0.35 <= cy <= h * 0.65:
                middle_line_found = True

        # Hoekmarkeringen
        cx = x + cw / 2
        cy = y + ch / 2
        for rx1, ry1, rx2, ry2 in corner_regions:
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                corner_hits += 1
                break

    return {
        "corner_hits": corner_hits,
        "middle_line_found": middle_line_found
    }


def compare_candidate_to_template(
    candidate_mask: np.ndarray,
    template_mask: np.ndarray
) -> float:
    resized_candidate = cv2.resize(
        candidate_mask,
        (template_mask.shape[1], template_mask.shape[0]),
        interpolation=cv2.INTER_AREA
    )
    diff = cv2.absdiff(resized_candidate, template_mask)
    similarity = 1.0 - (float(np.mean(diff)) / 255.0)
    return max(0.0, min(1.0, similarity))


def score_black_features(scene_roi_bgr: np.ndarray) -> float:
    features = detect_black_marker_features(scene_roi_bgr)
    score = 0.0

    score += min(features["corner_hits"], 4) / 4.0 * 0.6
    score += 0.4 if features["middle_line_found"] else 0.0

    return min(score, 1.0)


def detect_marker_in_scene(
    scene_bgr: np.ndarray,
    marker_template: Dict[str, Any]
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    mask = yellow_mask(scene_bgr)
    candidates = find_vertical_yellow_candidates(mask)

    if not candidates:
        return None, 0.0

    best_rect = None
    best_score = -1.0

    template_mask = marker_template["template_mask"]
    template_aspect = marker_template["aspect_ratio"]

    for x, y, w, h in candidates:
        candidate_mask = mask[y:y + h, x:x + w]
        candidate_bgr = scene_bgr[y:y + h, x:x + w]

        similarity = compare_candidate_to_template(candidate_mask, template_mask)

        aspect = h / max(w, 1)
        aspect_score = 1.0 - min(abs(aspect - template_aspect) / max(template_aspect, 1e-6), 1.0)

        black_score = score_black_features(candidate_bgr)

        # gewicht op vorm + zwart patroon + template
        final_score = (similarity * 0.45) + (aspect_score * 0.15) + (black_score * 0.40)

        if final_score > best_score:
            best_score = final_score
            best_rect = (x, y, w, h)

    if best_score < 0.35:
        return None, best_score

    return best_rect, best_score


def detect_window(scene_bgr: np.ndarray, marker_rect: Optional[Tuple[int, int, int, int]]) -> Optional[Dict[str, Any]]:
    gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Combineer threshold + edge voor betere raamkandidaten
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    edges = cv2.Canny(blur, 50, 150)
    combined = cv2.bitwise_or(thresh, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.dilate(combined, kernel, iterations=1)
    combined = cv2.erode(combined, kernel, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = scene_bgr.shape[:2]
    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (img_h * img_w) * 0.01:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        if rect_area < 5000:
            continue

        # niet tegen rand van foto
        if x < 5 or y < 5 or x + w > img_w - 5 or y + h > img_h - 5:
            continue

        # geen overlap met marker
        if marker_rect is not None:
            mx, my, mw, mh = marker_rect
            overlap = not (x + w < mx or mx + mw < x or y + h < my or my + mh < y)
            if overlap:
                continue

        aspect = w / max(h, 1)
        if not (0.35 <= aspect <= 3.5):
            continue

        rectangularity = area / max(rect_area, 1)
        if rectangularity < 0.35:
            continue

        # donkere zone binnenin helpt vaak voor glas/raam
        roi = gray[y:y + h, x:x + w]
        mean_intensity = float(np.mean(roi))
        darkness_score = 1.0 - min(mean_intensity / 255.0, 1.0)

        # raamkader vaak duidelijke rand
        perimeter = cv2.arcLength(cnt, True)
        compactness = 4 * math.pi * area / max(perimeter * perimeter, 1.0)
        compactness = max(0.0, min(compactness, 1.0))

        # score: groot + redelijk rechthoekig + beetje donker + compact
        score = (
            (rect_area / (img_h * img_w)) * 2.0
            + rectangularity * 1.8
            + darkness_score * 0.8
            + compactness * 0.4
        )

        if score > best_score:
            best_score = score
            best = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "score": round(score, 3),
                "rectangularity": round(rectangularity, 3),
                "mean_intensity": round(mean_intensity, 1)
            }

    return best


def annotate_image(
    img: np.ndarray,
    marker_rect: Optional[Tuple[int, int, int, int]],
    window_data: Optional[Dict[str, Any]],
    scale_cm_per_px: Optional[float]
) -> np.ndarray:
    out = img.copy()

    if marker_rect is not None:
        x, y, w, h = marker_rect
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        radius = int(max(w, h) * 0.75)
        cv2.circle(out, (cx, cy), radius, (0, 255, 255), 4)
        cv2.putText(
            out, "Marker",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
        )

    if window_data is not None:
        x = window_data["x"]
        y = window_data["y"]
        w = window_data["width"]
        h = window_data["height"]

        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 4)

        if scale_cm_per_px is not None:
            width_cm = round(w * scale_cm_per_px, 2)
            height_cm = round(h * scale_cm_per_px, 2)
            area_m2 = round((width_cm * height_cm) / 10000.0, 3)

            # breedte bovenaan
            cv2.putText(
                out, f"{width_cm} cm",
                (x + max(0, int(w / 2) - 55), max(25, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA
            )

            # hoogte links verticaal
            label_img = np.zeros((250, 60, 3), dtype=np.uint8)
            cv2.putText(
                label_img, f"{height_cm} cm",
                (5, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA
            )
            label_img = cv2.rotate(label_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            ly1 = max(0, y + h // 2 - label_img.shape[0] // 2)
            ly2 = min(out.shape[0], ly1 + label_img.shape[0])
            lx1 = max(0, x - 50)
            lx2 = min(out.shape[1], lx1 + label_img.shape[1])

            label_crop = label_img[:ly2 - ly1, :lx2 - lx1]
            gray_crop = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_crop, 1, 255, cv2.THRESH_BINARY)
            roi = out[ly1:ly2, lx1:lx2]
            label_fg = cv2.bitwise_and(label_crop, label_crop, mask=mask)
            bg_mask = cv2.bitwise_not(mask)
            roi_bg = cv2.bitwise_and(roi, roi, mask=bg_mask)
            out[ly1:ly2, lx1:lx2] = cv2.add(roi_bg, label_fg)

            # oppervlakte onderaan
            cv2.putText(
                out, f"Raam 1 • {area_m2} m²",
                (x, min(out.shape[0] - 20, y + h + 28)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA
            )

    return out


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "API draait"}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    marker: UploadFile = File(...)
) -> JSONResponse:
    try:
        image_bytes = await image.read()
        marker_bytes = await marker.read()

        scene_bgr_original = read_upload_to_bgr(image_bytes)
        marker_bgr_original = read_upload_to_bgr(marker_bytes)

        scene_bgr, scene_scale = resize_if_needed(scene_bgr_original, max_width=1400)
        marker_bgr, _ = resize_if_needed(marker_bgr_original, max_width=500)

        marker_template = extract_marker_template(marker_bgr)
        marker_rect, marker_score = detect_marker_in_scene(scene_bgr, marker_template)

        reference_detected = marker_rect is not None
        scale_cm_per_pixel = None
        marker_json = None

        if reference_detected:
            mx, my, mw, mh = marker_rect
            scale_cm_per_pixel = MARKER_REAL_HEIGHT_CM / mh
            marker_json = {
                "x": int(mx),
                "y": int(my),
                "width": int(mw),
                "height": int(mh),
                "match_score": round(marker_score, 3)
            }

        window_data = None
        windows = []
        warning = None

        if reference_detected:
            window_data = detect_window(scene_bgr, marker_rect)
            if window_data is not None:
                width_cm = round(window_data["width"] * scale_cm_per_pixel, 2)
                height_cm = round(window_data["height"] * scale_cm_per_pixel, 2)
                area_cm2 = round(width_cm * height_cm, 2)
                area_m2 = round(area_cm2 / 10000.0, 3)

                windows.append({
                    "x": int(window_data["x"]),
                    "y": int(window_data["y"]),
                    "width": int(window_data["width"]),
                    "height": int(window_data["height"]),
                    "width_cm": width_cm,
                    "height_cm": height_cm,
                    "area_cm2": area_cm2,
                    "area_m2": area_m2,
                    "confidence": window_data["score"]
                })
            else:
                warning = "Marker herkend, maar raam niet betrouwbaar gedetecteerd."
        else:
            warning = "Marker niet herkend."

        annotated = annotate_image(scene_bgr, marker_rect, window_data, scale_cm_per_pixel)
        annotated_b64 = encode_bgr_to_base64(annotated)

        return JSONResponse({
            "reference_detected": reference_detected,
            "scale_cm_per_pixel": round(scale_cm_per_pixel, 4) if scale_cm_per_pixel else None,
            "marker": marker_json,
            "windows": windows,
            "warning": warning,
            "annotated_image": annotated_b64
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
