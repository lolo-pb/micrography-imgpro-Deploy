import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.processing import run_preview
from app.serialization import to_png_b64

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/preview", response_class=HTMLResponse)
async def preview(
    request: Request,
    image: UploadFile = File(),
    mode: str = Form(),
    params: str = Form(),
):
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return HTMLResponse("<p class='error'>could not decode image</p>", status_code=422)

    p = json.loads(params)

    if mode == "simple":
        core_params = {
            "gamma": float(p.get("gamma", 1.0)),
            "otsu_classes": int(p.get("otsu_classes", 5)),
            "otsu_range": (int(p.get("otsu_range_low", 3)), int(p.get("otsu_range_high", 4))),
            "first_kernel_size": (5, 5),
            "second_kernel_size": (3, 3),
            "cont_mult": float(p.get("cont_mult", 2.5)),
            "ws_ths_factor": float(p.get("ws_ths_factor", 0.025)),
            "ws_gl_vecinity": int(p.get("ws_gl_vecinity", 15)),
        }
    else:
        bh = int(p.get("bh_ks", 7))
        bh = bh if bh % 2 == 1 else bh + 1
        core_params = {
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

    try:
        out = run_preview(img, mode, core_params)
    except Exception as e:
        return HTMLResponse(f"<p class='error'>processing error: {e}</p>", status_code=500)

    return templates.TemplateResponse(request, "preview.html", {
        "result_b64": to_png_b64(out["result"]),
        "stats": out["stats"],
    })
