import os
import zipfile
import io
from typing import Dict, Any, Tuple

import cv2
cv2.setUseOptimized(True)
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Existing pipeline modules 
from getmeresults import getMeResults
import getmepores as gmp
import getmeflashes as gmfl
import getmefibers as gmf
from common import getSegmentationFigure

# ---------- Helpers ----------

def fig_to_img(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape((h, w, 4))[:, :, :3]
    plt.close(fig)
    return img

def png_bytes(img: np.ndarray) -> bytes:
    if img is None: return b""
    ok, buf = cv2.imencode(".png", to_uint8(img))
    return buf.tobytes() if ok else b""

def png_bytes_bgr(img: np.ndarray) -> bytes:
    if img is None: return b""
    img = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""

def name_only_from_source(source: str) -> str:
    base = os.path.basename(source) if source else "image"
    return os.path.splitext(base)[0] or "image"

def to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None: return img
    if img.dtype == np.bool_: return (img.astype(np.uint8) * 255)
    if np.issubdtype(img.dtype, np.floating):
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)

def decode_uploaded_gray(upload) -> np.ndarray:
    upload.seek(0)
    data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError(f"Could not decode {upload.name}")
    return img

def to_rgb_for_display(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 2: return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2RGB)

def normalize_mask_for_display(mask: np.ndarray) -> np.ndarray:
    if mask is None: return mask
    if mask.ndim == 3: return to_rgb_for_display(mask)
    m = mask.astype(np.float32)
    mn, mx = float(np.min(m)), float(np.max(m))
    if mx <= mn: return np.zeros_like(mask, dtype=np.uint8)
    return ((m - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

def as_odd(n: int) -> int:
    return n if (n % 2 == 1) else n + 1

def get_default_params() -> Dict[str, Any]:
    return {
        "otsu_classes": 5, "otsu_range": (0, 4),
        "first_kernel_size": (5, 5), "second_kernel_size": (3, 3),
        "contours_mult": 2.5, "bh_ks": (7, 7), "bhm_iter": 4,
        "bhm_mult": 60, "cont_mult": 2.5, "ws_ths_factor": 0.025,
        "ws_gl_vecinity": 15,
    }

def build_parameters_ui(p: Dict[str, Any], key_suffix: str) -> Dict[str, Any]:
    """Uses key_suffix to ensure sliders are unique per image."""
    st.subheader("Parameters")
    
    o_classes = st.slider("Multi-Otsu Classes", 2, 10, p.get("otsu_classes", 5), key=f"ots_c_{key_suffix}")
    
    # Ensure range is valid for new class count
    curr_range = p.get("otsu_range", (0, 4))
    safe_range = (min(curr_range[0], o_classes-1), min(curr_range[1], o_classes-1))
    
    o_range = st.slider("Class Range", 0, o_classes - 1, safe_range, key=f"ots_r_{key_suffix}")

    with st.expander("Advanced Settings", expanded=False):
        fk = st.slider("first_kernel_size (odd)", 1, 31, p.get("first_kernel_size", (5,5))[0], 2, key=f"fk_{key_suffix}")
        sk = st.slider("second_kernel_size (odd)", 1, 31, p.get("second_kernel_size", (3,3))[0], 2, key=f"sk_{key_suffix}")
        cont_mult = st.slider("cont_mult", 0.1, 10.0, float(p.get("contours_mult", 2.5)), 0.1, key=f"cm_{key_suffix}")
        bh = st.slider("bh_ks (odd)", 1, 61, p.get("bh_ks", (7,7))[0], 2, key=f"bh_{key_suffix}")
        bhm_iter = st.slider("bhm_iter", 1, 20, p.get("bhm_iter", 4), 1, key=f"bmi_{key_suffix}")
        bhm_mult = st.slider("bhm_mult", 1, 300, p.get("bhm_mult", 60), 1, key=f"bmm_{key_suffix}")
        cont_mult_fib = st.slider("cont_mult (fibers)", 0.1, 10.0, float(p.get("cont_mult", 2.5)), 0.1, key=f"cmf_{key_suffix}")
        ws_ths_factor = st.slider("ws_ths_factor", 0.0001, 0.2, float(p.get("ws_ths_factor", 0.025)), 0.0005, format="%.4f", key=f"wsf_{key_suffix}")
        ws_gl_vecinity = st.slider("ws_gl_vecinity", 1, 200, p.get("ws_gl_vecinity", 15), 1, key=f"wsv_{key_suffix}")

    return {
        "otsu_classes": int(o_classes), "otsu_range": o_range,
        "first_kernel_size": (as_odd(fk), as_odd(fk)),
        "second_kernel_size": (as_odd(sk), as_odd(sk)),
        "contours_mult": cont_mult, "bh_ks": (as_odd(bh), as_odd(bh)),
        "bhm_iter": bhm_iter, "bhm_mult": bhm_mult,
        "cont_mult": cont_mult_fib, "ws_ths_factor": ws_ths_factor,
        "ws_gl_vecinity": ws_gl_vecinity,
    }

def run_pipeline(base_img_gray: np.ndarray, mode: str, parameters: Dict[str, Any]):
    outputs: Dict[str, Any] = {}
    try:
        if mode == "All Results":
            stats, segmentation, coloring = getMeResults(base_img_gray, parameters)
            fig, ax = plt.subplots(figsize=(10,6))
            getSegmentationFigure(segmentation, stats, "out", ax=ax)
            outputs["results"] = fig_to_img(fig)
        elif mode == "Fibers":
            binary_mask, _, _ = gmf.getMeFibers(
                base_img_gray, bh_ks=parameters["bh_ks"], bhm_iter=parameters["bhm_iter"],
                bhm_mult=parameters["bhm_mult"], cont_mult=parameters["cont_mult"],
                ws_ths_factor=parameters["ws_ths_factor"], ws_gl_vecinity=parameters["ws_gl_vecinity"],
                otsu_classes=parameters["otsu_classes"], otsu_range=parameters["otsu_range"], 
            )
            outputs["binary_mask"] = binary_mask
        elif mode == "Flashes":
            outputs["mask_flashes"] = gmfl.getMeFlashes(base_img_gray, cont_mult=parameters["cont_mult"])
        elif mode == "Pores":
            mask_bubbles, _ = gmp.getMetPores(base_img_gray, parameters["first_kernel_size"], parameters["second_kernel_size"])
            outputs["mask_bubbles"] = mask_bubbles
    except Exception as e:
        st.error(f"Pipeline error: {e}")
    return outputs

def get_exportable_items(outputs: Dict[str, Any], name_only: str) -> list[Tuple[str, bytes]]:
    items = []
    if not outputs: return items
    mapping = {
        "results": (f"{name_only}_results.png", png_bytes_bgr),
        "binary_mask": (f"{name_only}_fib.png", png_bytes),
        "mask_flashes": (f"{name_only}_flash.png", png_bytes),
        "mask_bubbles": (f"{name_only}_pore.png", png_bytes)
    }
    for key, (fname, func) in mapping.items():
        if key in outputs: items.append((fname, func(outputs[key])))
    return items

# ---------- UI Execution ----------
st.set_page_config(page_title="Batch Image Processor", layout="wide")

if "img_data" not in st.session_state: 
    st.session_state.img_data = {}

with st.sidebar:
    st.header("Upload")
    uploaded_files = st.file_uploader("Select images", type=["png", "jpg", "tif"], accept_multiple_files=True)
    
    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        # Cleanup session state if files removed
        st.session_state.img_data = {k: v for k, v in st.session_state.img_data.items() if k in current_names}
        # Add new files
        for f in uploaded_files:
            if f.name not in st.session_state.img_data:
                st.session_state.img_data[f.name] = {
                    "image": decode_uploaded_gray(f), 
                    "params": get_default_params(),
                    "mode": "All Results", 
                    "outputs": None
                }

    if st.session_state.img_data:
        st.divider()
        st.header("Selection & Tuning")
        active_file = st.selectbox("Pick image to edit:", list(st.session_state.img_data.keys()))
        
        data = st.session_state.img_data[active_file]
        
        # Mode Selection
        modes = ["All Results", "Fibers", "Flashes", "Pores"]
        data["mode"] = st.selectbox("Processing Mode", modes, index=modes.index(data["mode"]), key=f"mode_{active_file}")
        
        # Parameters (Unique keys prevent state bleeding)
        data["params"] = build_parameters_ui(data["params"], active_file)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Preview"):
                data["outputs"] = run_pipeline(data["image"], data["mode"], data["params"])
        with col2:
            if st.button("Apply to All"):
                for k in st.session_state.img_data:
                    st.session_state.img_data[k]["params"] = data["params"].copy()
                    st.session_state.img_data[k]["mode"] = data["mode"]
                st.success("Applied to all!")

        st.divider()
        st.header("Batch Export")
        if st.button("🚀 Process & Download All (ZIP)", type="primary"):
            zip_buffer = io.BytesIO()
            prog = st.progress(0)
            status = st.empty()
            
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx, (fname, d) in enumerate(st.session_state.img_data.items()):
                    status.text(f"Processing {fname}...")
                    out = run_pipeline(d["image"], d["mode"], d["params"])
                    items = get_exportable_items(out, name_only_from_source(fname))
                    for item_name, item_bytes in items:
                        zip_file.writestr(item_name, item_bytes)
                    prog.progress((idx + 1) / len(st.session_state.img_data))
            
            status.text("Done!")
            st.download_button("Download ZIP", zip_buffer.getvalue(), "batch_results.zip", "application/zip")

# ---------- Main Panel ----------
if st.session_state.img_data and 'active_file' in locals():
    data = st.session_state.img_data[active_file]
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(f"Input: {active_file}")
        st.image(to_rgb_for_display(data["image"]), width="stretch")
    with col_r:
        st.subheader("Output Preview")
        if data["outputs"]:
            for k, v in data["outputs"].items():
                if isinstance(v, np.ndarray):
                    st.image(normalize_mask_for_display(v) if "mask" in k or "binary" in k else v, caption=k, width="stretch")
        else:
            st.info("Adjust parameters and click 'Preview'.")
else:
    st.info("Please upload images in the sidebar.")