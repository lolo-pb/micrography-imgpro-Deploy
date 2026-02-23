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
import numpy as np
import streamlit as st

# Your existing pipeline modules (must be importable from this folder)
from getmeresults import getMeResults
import getmepores as gmp
import getmeflashes as gmfl
import getmefibers as gmf


# ---------- Helpers ----------
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

        ws_ths_factor = st.slider("ws_ths_factor", 0.0001, 0.2, 0.025, 0.0005, format="%.4f")
        ws_gl_vecinity = st.slider("ws_gl_vecinity", 1, 200, 15, 1)

    # keep both keys your code expects; controller.py uses both contours_mult and cont_mult in defaults
    params = {
        "first_kernel_size": first_kernel_size,
        "second_kernel_size": second_kernel_size,
        "contours_mult": float(cont_mult),
        "bh_ks": bh_ks,
        "bhm_iter": int(bhm_iter),
        "bhm_mult": int(bhm_mult),
        "cont_mult": float(cont_mult_fib),  # used by fibers + flashes in your modules
        "ws_ths_factor": float(ws_ths_factor),
        "ws_gl_vecinity": int(ws_gl_vecinity),
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
    mode = st.radio("Mode", ["All Results", "Fibers", "Flashes", "Pores"], index=0)

    st.divider()
    params = build_parameters_ui()

    st.divider()
    run_btn = st.button("Run processing", type="primary")


# Load image
base_gray: Optional[np.ndarray] = None
load_err: Optional[str] = None

try:
    if source == "Upload":
        if uploaded is not None:
            base_gray = decode_uploaded_gray(uploaded)
    else:
        if picked_path is not None:
            base_gray = imread_gray(picked_path)
except Exception as e:
    load_err = str(e)

if load_err:
    st.error(load_err)

if base_gray is None:
    st.info("Pick or upload an image to start.")
    st.stop()

# Display original
left, right = st.columns([1, 1])

with left:
    st.subheader("Original")
    st.image(to_rgb_for_display(base_gray), use_container_width=True)
    st.caption(f"Shape: {base_gray.shape} | dtype: {base_gray.dtype}")

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

with right:
    st.subheader("Output")

    if outputs is None:
        st.info("Hit **Run processing** to render the result.")
    else:
        st.caption(f"Last run: {last_mode}")

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