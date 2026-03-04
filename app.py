# app.py
# Simple UI to pick ONE image and tweak parameters, then run your processing on demand.
#
# Run:
#   pip install streamlit opencv-python numpy
#   streamlit run app.py

import os
import glob
from typing import Dict, Any, Tuple, Optional

import cv2
cv2.setUseOptimized(True)
import numpy as np
import streamlit as st

# Existing pipeline modules 
from getmeresults import getMeResults
import getmepores as gmp
import getmeflashes as gmfl
import getmefibers as gmf

from common import getSegmentationFigure, getColoringFigure
import matplotlib.pyplot as plt

# ---------- Helpers ----------

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def name_only_from_source(source: str) -> str:
    # source can be a path or just a filename
    base = os.path.basename(source) if source else "image"
    return os.path.splitext(base)[0] or "image"

def to_uint8(img: np.ndarray) -> np.ndarray:
    """Make cv2.imwrite happy for common mask formats (bool/0-1 float/etc)."""
    if img is None:
        return img
    if img.dtype == np.bool_:
        return (img.astype(np.uint8) * 255)
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        mx = float(np.nanmax(img)) if img.size else 0.0
        if mx <= 1.0:
            return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.clip(img, 0.0, 255.0).astype(np.uint8)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(img, 0, 255).astype(np.uint8)
    # fallback
    return img.astype(np.uint8, copy=False)

def list_images_in_folder(folder: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    return files


def imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def decode_uploaded_gray(upload) -> np.ndarray:
    data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode uploaded image (unsupported/invalid file).")
    return img


def to_rgb_for_display(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2RGB)
    # assume BGR
    return cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2RGB)


def normalize_mask_for_display(mask: np.ndarray) -> np.ndarray:
    # masks often come as 0/255 already, but normalize safely
    if mask is None:
        return mask
    m = mask
    if m.ndim == 3:
        # if it's already color
        return to_rgb_for_display(m)
    m = m.astype(np.float32)
    mn, mx = float(np.min(m)), float(np.max(m))
    if mx <= mn:
        return np.zeros_like(mask, dtype=np.uint8)
    m = (m - mn) / (mx - mn)
    m = (m * 255.0).clip(0, 255).astype(np.uint8)
    return m


def as_odd(n: int) -> int:
    return n if (n % 2 == 1) else n + 1


def build_parameters_ui() -> Dict[str, Any]:
    st.subheader("Parameters")

    # --- Top: Otsu controls first ---
    st.markdown("**Fibers (Otsu)**")
    o_classes = st.slider("Multi-Otsu Classes", 2, 10, 5)
    o_range = st.slider("Class Range", 0, o_classes - 1, (0, o_classes - 1))

    # --- Collapsible: everything else ---
    with st.expander("More parameters", expanded=False):
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Pores (kernels)**")
            fk = st.slider("first_kernel_size (odd)", 1, 31, 5, 2)
            sk = st.slider("second_kernel_size (odd)", 1, 31, 3, 2)
            first_kernel_size = (as_odd(int(fk)), as_odd(int(fk)))
            second_kernel_size = (as_odd(int(sk)), as_odd(int(sk)))

            st.markdown("**Flashes**")
            cont_mult = st.slider("cont_mult", 0.1, 10.0, 2.5, 0.1)

        with colB:
            st.markdown("**Fibers (blackhat + watershed)**")
            bh = st.slider("bh_ks (odd)", 1, 61, 7, 2)
            bh_ks = (as_odd(int(bh)), as_odd(int(bh)))

            bhm_iter = st.slider("bhm_iter", 1, 20, 4, 1)
            bhm_mult = st.slider("bhm_mult", 1, 300, 60, 1)
            cont_mult_fib = st.slider("cont_mult (fibers)", 0.1, 10.0, 2.5, 0.1)

            ws_ths_factor = st.slider(
                "ws_ths_factor", 0.0001, 0.2, 0.025, 0.0005, format="%.4f"
            )
            ws_gl_vecinity = st.slider("ws_gl_vecinity", 1, 200, 15, 1)

    # If the expander is collapsed on first render, these vars won't exist yet.
    # So give defaults if they weren't set (Streamlit reruns will populate them).
    first_kernel_size = locals().get("first_kernel_size", (5, 5))
    second_kernel_size = locals().get("second_kernel_size", (3, 3))
    cont_mult = float(locals().get("cont_mult", 2.5))
    bh_ks = locals().get("bh_ks", (7, 7))
    bhm_iter = int(locals().get("bhm_iter", 4))
    bhm_mult = int(locals().get("bhm_mult", 60))
    cont_mult_fib = float(locals().get("cont_mult_fib", 2.5))
    ws_ths_factor = float(locals().get("ws_ths_factor", 0.025))
    ws_gl_vecinity = int(locals().get("ws_gl_vecinity", 15))

    params = {
        "otsu_classes": int(o_classes),
        "otsu_range": o_range,

        "first_kernel_size": first_kernel_size,
        "second_kernel_size": second_kernel_size,
        "contours_mult": cont_mult,          # flashes path in your original UI
        "bh_ks": bh_ks,
        "bhm_iter": bhm_iter,
        "bhm_mult": bhm_mult,
        "cont_mult": cont_mult_fib,          # fibers + flashes in your modules
        "ws_ths_factor": ws_ths_factor,
        "ws_gl_vecinity": ws_gl_vecinity,
    }
    return params


def run_pipeline(
    base_img_gray: np.ndarray,
    mode: str,
    parameters: Dict[str, Any],
):
    """
    Returns a dict of outputs to display.
    """
    outputs: Dict[str, Any] = {}

    if mode == "All Results":
        stats, segmentation, coloring = getMeResults(base_img_gray, parameters)
        outputs["segmentation-r"] = segmentation
        outputs["stats-r"] = stats

    elif mode == "All Data":
        stats, segmentation, coloring = getMeResults(base_img_gray, parameters)
        outputs["stats"] = stats
        outputs["segmentation"] = segmentation
        outputs["coloring"] = coloring

    elif mode == "Fibers":
        
        binary_mask, contours_filtered_img, list_masks = gmf.getMeFibers(
            base_img_gray,
            bh_ks=parameters["bh_ks"],
            bhm_iter=parameters["bhm_iter"],
            bhm_mult=parameters["bhm_mult"],
            cont_mult=parameters["cont_mult"],
            ws_ths_factor=parameters["ws_ths_factor"],
            ws_gl_vecinity=parameters["ws_gl_vecinity"],
            otsu_classes=parameters["otsu_classes"],
            otsu_range=parameters["otsu_range"], 
        )
        outputs["binary_mask"] = binary_mask
        outputs["contours_filtered_img"] = contours_filtered_img
        outputs["list_masks_count"] = 0 if list_masks is None else len(list_masks)

    elif mode == "Flashes":
        mask_flashes = gmfl.getMeFlashes(
            base_img_gray,
            cont_mult=parameters["cont_mult"],
        )
        outputs["mask_flashes"] = mask_flashes

    elif mode == "Pores":
        mask_bubbles, undefined_region_mask = gmp.getMetPores(
            base_img_gray,
            first_kernel_size=parameters["first_kernel_size"],
            second_kernel_size=parameters["second_kernel_size"],
        )
        outputs["mask_bubbles"] = mask_bubbles
        outputs["undefined_region_mask"] = undefined_region_mask

    else:
        raise ValueError(f"Unknown mode: {mode}")
    

    return outputs


# ---------- UI ----------
st.set_page_config(page_title="Image Processing Tuner", layout="wide")
st.title("Image Processing Tuner")

with st.sidebar:
    st.header("Input")
    source = st.radio("Image source", ["Upload", "Pick from folder"], horizontal=False)

    folder_default = "preprodata"
    folder = st.text_input("Folder", folder_default, help="Used only when 'Pick from folder' is selected.")

    uploaded = None
    picked_path = None

    if source == "Upload":
        uploaded = st.file_uploader("Upload one image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    else:
        files = list_images_in_folder(folder)
        if len(files) == 0:
            st.warning(f"No images found in: {folder}")
        else:
            picked_path = st.selectbox("Select image", files)

    st.divider()
    st.header("What to run")
    mode = st.radio("Mode", ["All Results", "All Data", "Fibers", "Flashes", "Pores"], index=0)

    st.divider()
    params = build_parameters_ui()

    st.divider()
    run_btn = st.button("Run processing", type="primary")


# Load image
base_gray: Optional[np.ndarray] = None
load_err: Optional[str] = None
img_source_label: Optional[str] = None

try:
    if source == "Upload":
        if uploaded is not None:
            base_gray = decode_uploaded_gray(uploaded)
            img_source_label = uploaded.name
    else:
        if picked_path is not None:
            base_gray = imread_gray(picked_path)
            img_source_label = picked_path
except Exception as e:
    load_err = str(e)

if load_err:
    st.error(load_err)

if base_gray is None:
    st.info("Pick or upload an image to start.")
    st.stop()

# Export
name_only = name_only_from_source(img_source_label or "image")
st.session_state["last_name_only"] = name_only

def export_last_outputs(outputs: Dict[str, Any], name_only: str) -> Tuple[list[str], Optional[str]]:
    """
    Save current outputs to disk using controller.py naming.
    Returns (saved_paths, error_message).
    """
    saved: list[str] = []
    try:
        # Match controller.py folders + names
        if "segmentation" in outputs or "coloring" in outputs:
            out_dir = "processed_results"
            ensure_dir(out_dir)

            if "segmentation" in outputs:
                p = os.path.join(out_dir, f"{name_only}_seg.png")
                cv2.imwrite(p, to_uint8(outputs["segmentation"]))
                saved.append(p)

            if "coloring" in outputs:
                p = os.path.join(out_dir, f"{name_only}_color.png")
                cv2.imwrite(p, to_uint8(outputs["coloring"]))
                saved.append(p)

        if "binary_mask" in outputs:
            out_dir = "processed_fibers"
            ensure_dir(out_dir)
            p = os.path.join(out_dir, f"{name_only}_fib.png")
            cv2.imwrite(p, to_uint8(outputs["binary_mask"]))
            saved.append(p)

        if "mask_flashes" in outputs:
            out_dir = "processed_flashes"
            ensure_dir(out_dir)
            p = os.path.join(out_dir, f"{name_only}_flash.png")
            cv2.imwrite(p, to_uint8(outputs["mask_flashes"]))
            saved.append(p)

        if "mask_bubbles" in outputs:
            out_dir = "processed_pores"
            ensure_dir(out_dir)
            p = os.path.join(out_dir, f"{name_only}_pore.png")
            cv2.imwrite(p, to_uint8(outputs["mask_bubbles"]))
            saved.append(p)

        # Note: controller.py doesn't export stats or undefined_region_mask, so we keep parity.
        return saved, None
    except Exception as e:
        return saved, str(e)

# Display original
left, right = st.columns([1, 1])

with left:
    st.subheader("Original")
    st.image(to_rgb_for_display(base_gray), use_container_width=True)
    st.caption(f"Shape: {base_gray.shape} | dtype: {base_gray.dtype}")

    st.divider()

    export_btn = st.button(
        "Export outputs",
        type="secondary",
        use_container_width=True
    )

    if export_btn:
        outputs = st.session_state.get("last_outputs", None)

        if outputs is None:
            st.warning("Run the pipeline first.")
        else:
            name_only = st.session_state.get("last_name_only", "image")
            saved_paths, err = export_last_outputs(outputs, name_only)

            if err:
                st.error(f"Export failed: {err}")
            elif not saved_paths:
                st.warning("Nothing to export.")
            else:
                st.success("Exported:")
                for p in saved_paths:
                    st.write(p)

# Run + display
if run_btn:
    try:
        outputs = run_pipeline(base_gray, mode, params)
        st.session_state["last_outputs"] = outputs
        st.session_state["last_mode"] = mode
        st.session_state["last_params"] = params
        st.success("Done.")
    except Exception as e:
        st.exception(e)

outputs = st.session_state.get("last_outputs", None)
last_mode = st.session_state.get("last_mode", None)
last_name_only = st.session_state.get("last_name_only", "image")

with right:
    st.subheader("Output")

    if outputs is None:
        st.info("Hit **Run processing** to render the result.")
    else:
        st.caption(f"Last run: {last_mode}")

        if "segmentation-r" in outputs:
            st.markdown("**Segmentation**")

            fig, ax = plt.subplots(figsize=(14,8))
            getSegmentationFigure(
                outputs["segmentation-r"],
                outputs["stats-r"],
                "out",
                ax=ax,

            )
            st.pyplot(fig)

        if "stats" in outputs:
            st.markdown("**Stats**")
            # stats is a dict in your pipeline
            st.json(outputs["stats"])

        if "segmentation" in outputs:
            st.markdown("**Segmentation**")
            st.image(to_rgb_for_display(outputs["segmentation"]), use_container_width=True)

        if "coloring" in outputs:
            st.markdown("**Coloring**")
            st.image(to_rgb_for_display(outputs["coloring"]), use_container_width=True)

        if "binary_mask" in outputs:
            st.markdown("**Fibers mask**")
            st.image(normalize_mask_for_display(outputs["binary_mask"]), use_container_width=True)

        if "contours_filtered_img" in outputs:
            st.markdown("**Contours filtered**")
            st.image(to_rgb_for_display(outputs["contours_filtered_img"]), use_container_width=True)

        if "list_masks_count" in outputs:
            st.caption(f"Fiber sub-masks: {outputs['list_masks_count']}")

        if "mask_flashes" in outputs:
            st.markdown("**Flashes mask**")
            st.image(normalize_mask_for_display(outputs["mask_flashes"]), use_container_width=True)

        if "mask_bubbles" in outputs:
            st.markdown("**Pores mask (bubbles)**")
            st.image(normalize_mask_for_display(outputs["mask_bubbles"]), use_container_width=True)

        if "undefined_region_mask" in outputs:
            st.markdown("**Undefined region mask**")
            st.image(normalize_mask_for_display(outputs["undefined_region_mask"]), use_container_width=True)

st.caption("Tip: tweak parameters on the left and hit **Run processing** again to compare quickly.")