import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# project root needs to be importable for the existing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import getSegmentationFigure
from getmeresults import getMeResults, getMeResultsSimple


def center_crop(img: np.ndarray, size: int = 1000) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(size, h, w)
    cy, cx = h // 2, w // 2
    half = side // 2
    y1, x1 = max(0, cy - half), max(0, cx - half)
    y2, x2 = y1 + side, x1 + side
    if y2 > h:
        y2, y1 = h, h - side
    if x2 > w:
        x2, x1 = w, w - side
    return img[y1:y2, x1:x2].copy()


def fig_to_img(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape((h, w, 4))[:, :, :3]
    plt.close(fig)
    return img


def run_preview(img: np.ndarray, mode: str, params: dict, crop_size: int = 1000) -> dict:
    cropped = center_crop(img, crop_size)

    if mode == "simple":
        stats, segmentation, coloring = getMeResultsSimple(cropped, params)
    else:
        stats, segmentation, coloring, debug = getMeResults(cropped, params, return_debug=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    getSegmentationFigure(segmentation, stats, "preview", ax=ax)
    result_img = fig_to_img(fig)

    out = {"stats": stats, "result": result_img, "coloring": coloring}
    if mode == "pro":
        out["debug"] = debug
    return out
