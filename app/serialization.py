import base64
import io
import os
import zipfile

import cv2
import numpy as np


def to_png_b64(img: np.ndarray) -> str:
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/png;base64,{b64}"


def to_png_bytes(img: np.ndarray) -> bytes:
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def stats_to_csv(name: str, params: dict, stats: dict) -> bytes:
    cols = ["file"]
    vals = [name]
    for k, v in params.items():
        cols.append(f"param_{k}")
        vals.append(str(v))
    for k, v in stats.items():
        cols.append(k)
        vals.append(str(v))
    header = ",".join(cols)
    row = ",".join(vals)
    return f"{header}\n{row}\n".encode("utf-8")


def build_zip(results: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for r in results:
            name = os.path.splitext(r["name"])[0]
            if r.get("export_result") and r.get("result") is not None:
                zf.writestr(f"{name}_result.png", to_png_bytes(r["result"]))
            if r.get("export_coloring") and r.get("coloring") is not None:
                zf.writestr(f"{name}_coloring.png", to_png_bytes(r["coloring"]))
            if r.get("export_stats") and r.get("stats") is not None:
                zf.writestr(f"{name}_stats.csv", stats_to_csv(name, r["params"], r["stats"]))
    return buf.getvalue()
