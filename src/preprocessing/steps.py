"""Individual preprocessing operations. Each returns a *new* image plus a log dict."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps


def to_gray_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"))


def exif_orientation(image: Image.Image) -> tuple[Image.Image, dict]:
    orientation = image.getexif().get(0x0112, 1)
    out = ImageOps.exif_transpose(image) if orientation != 1 else image.copy()
    return out, {"exif_orientation": int(orientation)}


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def detect_document(image: Image.Image, min_area_ratio: float = 0.2,
                    work_side: int = 1000) -> np.ndarray | None:
    """Find the page quadrilateral (photo of a page on a contrasting background).

    Classic contour approach: blur -> Canny -> close -> largest 4-gon covering at least
    ``min_area_ratio`` of the frame. Returns the quad in original pixel coordinates or
    None (then later steps are skipped and the log says so).
    """
    gray = to_gray_array(image)
    scale = work_side / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    scale = min(scale, 1.0)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = small.shape[0] * small.shape[1]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) >= min_area_ratio * frame_area:
            return order_quad(approx) / scale
    return None


def perspective_correct(image: Image.Image, quad: np.ndarray) -> Image.Image:
    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(np.asarray(image.convert("RGB")), matrix, (width, height),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def crop_fraction(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = image.size
    x0, y0, x1, y1 = box
    return image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def estimate_skew(image: Image.Image, max_angle: float = 5.0, step: float = 0.2,
                  work_side: int = 1200) -> float:
    """Skew angle (degrees, counter-clockwise positive) by projection-profile variance.

    Text lines produce sharp horizontal ink profiles when the page is level; we pick
    the rotation in [-max_angle, max_angle] maximising the variance of row sums.
    """
    gray = to_gray_array(image)
    scale = min(1.0, work_side / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = ink.shape
    center = (w / 2, h / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        m = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rotated.sum(axis=1, dtype=np.float64)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def rotate(image: Image.Image, angle: float) -> Image.Image:
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def resize(image: Image.Image, max_side: int | None = None, min_side: int | None = None) -> tuple[Image.Image, float]:
    w, h = image.size
    scale = 1.0
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
    elif min_side and max(w, h) < min_side:
        scale = min_side / max(w, h)
    if scale == 1.0:
        return image, 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def clahe(image: Image.Image, clip_limit: float = 2.0, tile: int = 8) -> Image.Image:
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB))


def autocontrast(image: Image.Image, cutoff: float = 1.0) -> Image.Image:
    return ImageOps.autocontrast(image.convert("RGB"), cutoff=cutoff)


def binarize(image: Image.Image, block_size: int = 41, c: int = 15) -> Image.Image:
    gray = to_gray_array(image)
    out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    return Image.fromarray(out).convert("RGB")
