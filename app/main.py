import json
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.processing import run_preview, run_full
from app.serialization import to_png_b64, build_zip

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


def parse_params(mode: str, p: dict) -> dict:
    if mode == "simple":
        return {
            "gamma": float(p.get("gamma", 1.0)),
            "otsu_classes": int(p.get("otsu_classes", 5)),
            "otsu_range": (int(p.get("otsu_range_low", 3)), int(p.get("otsu_range_high", 4))),
            "first_kernel_size": (5, 5),
            "second_kernel_size": (3, 3),
            "cont_mult": float(p.get("cont_mult", 2.5)),
            "ws_ths_factor": float(p.get("ws_ths_factor", 0.025)),
            "ws_gl_vecinity": int(p.get("ws_gl_vecinity", 15)),
        }
    bh = int(p.get("bh_ks", 7))
    bh = bh if bh % 2 == 1 else bh + 1
    return {
        "otsu_classes": int(p.get("otsu_classes", 5)),
        "otsu_range": (int(p.get("otsu_range_low", 3)), int(p.get("otsu_range_high", 4))),
        "first_kernel_size": (5, 5),
        "second_kernel_size": (3, 3),
        "bh_ks": (bh, bh),
        "bhm_iter": int(p.get("bhm_iter", 4)),
        "bhm_mult": int(p.get("bhm_mult", 60)),
        "cont_mult": float(p.get("cont_mult", 2.5)),
        "ws_ths_factor": float(p.get("ws_ths_factor", 0.025)),
        "ws_gl_vecinity": int(p.get("ws_gl_vecinity", 15)),
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/preview", response_class=HTMLResponse)
async def preview(
    request: Request,
    image: UploadFile = File(),
    mode: str = Form(),
    params: str = Form(),
    crop_size: int = Form(1000),
):
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return HTMLResponse("<p class='error'>could not decode image</p>", status_code=422)

    core_params = parse_params(mode, json.loads(params))

    try:
        out = run_preview(img, mode, core_params, crop_size=crop_size)
    except Exception as e:
        return HTMLResponse(f"<p class='error'>processing error: {e}</p>", status_code=500)

    return templates.TemplateResponse(request, "preview.html", {
        "result_b64": to_png_b64(out["result"]),
        "stats": out["stats"],
    })


@app.post("/batch-export")
async def batch_export(
    images: List[UploadFile] = File(),
    mode: str = Form(),
    params: str = Form(),
    export_flags: str = Form(),
):
    p = json.loads(params)
    core_params = parse_params(mode, p)
    flags = json.loads(export_flags)

    results = []
    for i, upload in enumerate(images):
        raw = await upload.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        f = flags[i] if i < len(flags) else {}
        try:
            out = run_full(img, mode, core_params)
            results.append({
                "name": upload.filename,
                "result": out["result"],
                "coloring": out["coloring"],
                "stats": out["stats"],
                "params": core_params,
                "export_result": f.get("result", True),
                "export_coloring": f.get("coloring", False),
                "export_stats": f.get("stats", False),
            })
        except Exception as e:
            logger.error(f"failed to process {upload.filename}: {e}")
            continue

    zip_bytes = build_zip(results)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="batch_results.zip"'},
    )
