# Visual Attention Simulation (Cognitive Vision System) - Streamlit App
# Simplified: Spectral Residual Saliency + Foveation + Scanpath with IoR

import streamlit as st
import numpy as np
import cv2
import json
from typing import List, Tuple

# ============ Utils ============

if "saliency" not in st.session_state:
    st.session_state.saliency = None
if "visited" not in st.session_state:
    st.session_state.visited = None
if "fixations" not in st.session_state:
    st.session_state.fixations = []
if "step" not in st.session_state:
    st.session_state.step = 0
if "img" not in st.session_state:
    st.session_state.img = None

def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img

def load_image(file) -> np.ndarray:
    """Load image from uploaded file (BytesIO) -> BGR (OpenCV)."""
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def resize_keep_aspect(img: np.ndarray, max_side: int = 768) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img.copy()
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ---------- Saliency ----------

def spectral_residual_saliency(img_bgr: np.ndarray, fft_size: int = 256) -> np.ndarray:
    """Compute Spectral Residual saliency map."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1e-6
    small = cv2.resize(gray, (fft_size, fft_size), interpolation=cv2.INTER_AREA)

    F = np.fft.fft2(small)
    A = np.abs(F)
    L = np.log(A + 1e-6)
    L_blur = cv2.blur(L, (3, 3))
    SR = L - L_blur
    sal_small = np.abs(np.fft.ifft2(np.exp(SR + 1j * np.angle(F)))) ** 2
    sal_small = cv2.GaussianBlur(sal_small, (9, 9), 2)

    sal = cv2.resize(sal_small, (w, h), interpolation=cv2.INTER_LINEAR)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
    return sal.astype(np.float32)

def overlay_heatmap(img_bgr: np.ndarray, saliency: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    sal_8u = to_uint8(saliency * 255.0)
    heat = cv2.applyColorMap(sal_8u, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
    return overlay

# ---------- Foveation ----------

def foveate(img_bgr: np.ndarray, fix: Tuple[int, int], radii=(40, 80, 140, 220, 320)) -> np.ndarray:
    """Approximate foveated rendering by blending blurred images across concentric rings."""
    blurs = [img_bgr]
    for k in [5, 9, 13, 21, 31]:
        blurs.append(cv2.GaussianBlur(img_bgr, (k, k), 0))

    H, W = img_bgr.shape[:2]
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    dist = np.sqrt((xs - fix[0]) ** 2 + (ys - fix[1]) ** 2)

    out = np.zeros_like(img_bgr)
    bands = [0] + list(radii) + [1e9]
    for i in range(len(bands) - 1):
        mask = (dist >= bands[i]) & (dist < bands[i + 1])
        out[mask] = blurs[min(i, len(blurs)-1)][mask]
    return out

# ---------- Fixations + IoR ----------

def gaussian_mask(shape: Tuple[int, int], center: Tuple[int, int], sigma: float) -> np.ndarray:
    H, W = shape
    y0, x0 = center[1], center[0]
    ys, xs = np.mgrid[0:H, 0:W]
    dist2 = (xs - x0) ** 2 + (ys - y0) ** 2
    return np.exp(-dist2 / (2 * sigma ** 2)).astype(np.float32)

def next_fixation(saliency: np.ndarray, visited_mask: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
    sal = saliency - visited_mask
    sal = np.clip(sal, 0, None)
    y, x = np.unravel_index(np.argmax(sal), sal.shape)
    return (int(x), int(y)), sal

def apply_ior(visited_mask: np.ndarray, center: Tuple[int, int], ior_sigma: float, ior_alpha: float) -> np.ndarray:
    H, W = visited_mask.shape
    ior = gaussian_mask((H, W), center, ior_sigma)
    return np.maximum(visited_mask, ior_alpha * ior)

def draw_scanpath(img_bgr: np.ndarray, fixations: List[Tuple[int, int]], fovea_r: int) -> np.ndarray:
    out = img_bgr.copy()
    for i, (x, y) in enumerate(fixations, start=1):
        cv2.circle(out, (x, y), fovea_r, (0, 255, 255), 2)
        cv2.circle(out, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(out, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if i > 1:
            x0, y0 = fixations[i - 2]
            cv2.line(out, (x0, y0), (x, y), (0, 255, 0), 2)
    return out

def crop_fovea(img_bgr: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    x, y = center
    x0, x1 = max(0, x - radius), min(W, x + radius)
    y0, y1 = max(0, y - radius), min(H, y + radius)
    return img_bgr[y0:y1, x0:x1]

# ============ Streamlit UI ============

st.set_page_config(page_title="Visual Attention Simulation", layout="wide")

st.title("👀 Visual Attention Simulation (Cognitive Vision System)")
st.markdown("""
**Demo includes:**
- Bottom-up saliency (Spectral Residual)
- Foveated vision (sharp center, blurry periphery)
- Scanpath with Inhibition of Return (IoR)
""")

with st.sidebar:
    st.header("Controls")
    n_fix = st.slider("Number of fixations", 1, 20, 8, 1)
    fovea_radius = st.slider("Fovea radius (px)", 20, 200, 80, 5)
    ior_sigma = st.slider("IoR sigma (px)", 10, 200, 60, 5)
    ior_alpha = st.slider("IoR strength (0-1)", 0.0, 1.0, 0.7, 0.05)
    max_side = st.slider("Max image side (resize)", 256, 1280, 768, 32)
    sal_alpha = st.slider("Heatmap overlay alpha", 0.0, 1.0, 0.45, 0.05)

    st.markdown("---")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def reset_state():
    st.session_state.saliency = None
    st.session_state.visited = None
    st.session_state.fixations = []
    st.session_state.step = 0

col_left, col_mid = st.columns([1,1])

if uploaded is not None:
    img0 = load_image(uploaded)  # BGR
    img0 = resize_keep_aspect(img0, max_side=max_side)
    st.session_state.img = img0
else:
    st.info("Upload an image to begin.")

if st.session_state.img is not None:
    img = st.session_state.img

    run_sal = st.button("🔍 Compute Saliency / Reset")
    step_btn = st.button("➡️ Step (Next Fixation)")
    run_full = st.button("🏁 Run All Fixations")

    if run_sal:
        reset_state()
        sal = spectral_residual_saliency(img)
        st.session_state.saliency = sal
        st.session_state.visited = np.zeros_like(sal, dtype=np.float32)

    if st.session_state.saliency is None:
        with col_left:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
        st.stop()

    sal = st.session_state.saliency
    visited = st.session_state.visited
    fixations = st.session_state.fixations

    # ---- FIXED do_one_step ---- #
    def do_one_step():
        (x, y), sal_eff = next_fixation(sal, visited)
        fixations.append((x, y))
        st.session_state.visited = apply_ior(visited, (x, y), ior_sigma=ior_sigma, ior_alpha=ior_alpha)
        cv2.circle(sal, (x, y), int(1.2 * fovea_radius), 0, -1)
        st.session_state.saliency = sal
        st.session_state.fixations = fixations

    if step_btn and len(fixations) < n_fix:
        do_one_step()

    if run_full:
        while len(st.session_state.fixations) < n_fix:
            do_one_step()

    # Visualizations
    heat_overlay = overlay_heatmap(img, st.session_state.saliency, alpha=sal_alpha)
    scanpath_img = draw_scanpath(heat_overlay, fixations, fovea_r=fovea_radius)

    with col_left:
        st.subheader("Saliency & Scanpath")
        st.image(cv2.cvtColor(scanpath_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_mid:
        st.subheader("Foveated View (Current Fixation)")
        if len(fixations) == 0:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
        else:
            current_fix = fixations[-1]
            fov_img = foveate(img, current_fix,
                              radii=(fovea_radius, int(1.6*fovea_radius),
                                     int(2.2*fovea_radius), int(3.0*fovea_radius)))
            st.image(cv2.cvtColor(fov_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with st.expander("Foveal Crop (for detail)"):
                crop = crop_fovea(img, current_fix, radius=fovea_radius)
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                         caption=f"Foveal crop @ {current_fix}", use_container_width=True)

    st.subheader("Raw Saliency Map")
    sal8 = to_uint8(st.session_state.saliency * 255.0)
    sal_color = cv2.applyColorMap(sal8, cv2.COLORMAP_JET)
    st.image(cv2.cvtColor(sal_color, cv2.COLOR_BGR2RGB), use_container_width=True)

    if len(fixations) > 0:
        scanpath = {"fixations": [{"x": int(x), "y": int(y)} for (x,y) in fixations],
                    "fovea_radius": fovea_radius,
                    "ior_sigma": ior_sigma,
                    "ior_alpha": ior_alpha}
        scan_json = json.dumps(scanpath, indent=2)
        st.download_button("⬇️ Download Scanpath (JSON)",
                           data=scan_json, file_name="scanpath.json", mime="application/json")

import streamlit as st

def show_footer():
    st.markdown("---")
    st.subheader("📘 Explanation of Terms & Parameters")

    with st.expander("🧠 Saliency Map"):
        st.write("""
        A **Saliency Map** highlights the most visually significant regions in an image.
        It is computed based on contrasts in color, intensity, and orientation.
        In human vision, saliency maps explain **where the eyes are most likely to look first**.
        """)

    with st.expander("👀 Fixations"):
        st.write("""
        A **Fixation** is the point where the eye stops and focuses for a short period.
        In the model, fixations are chosen based on the highest saliency values that haven't been visited yet.
        """)

    with st.expander("🔄 Inhibition of Return (IoR)"):
        st.write("""
        After the eyes focus on a point, the brain suppresses attention from that location for a while.
        This prevents the visual system from **repeatedly focusing on the same spot**, encouraging exploration.
        In the algorithm, this is simulated by reducing saliency in the visited regions.
        """)

    with st.expander("⚙️ Parameters"):
        st.write("""
        - **Visited**: Keeps track of the regions (pixels) already fixated on.
        - **Number of Steps**: Defines how many fixations will be made in sequence.
        - **Saliency Efficiency**: A measure of how strongly a given location stands out.
        - **Fixation Radius**: The area around the fixation point that is suppressed (IoR).
        """)

    with st.expander("🧩 Pathfinding Relevance"):
        st.write("""
        These concepts are inspired by **visual attention in cognitive science**.
        By simulating attention shifts (fixations) and ignoring past regions (IoR),
        we model how humans navigate visually in a crowded environment.
        
        This can be applied to **pathfinding in robotics**, **autonomous navigation**, and **AI-based vision systems**.
        """)

    st.markdown("---")
    st.markdown("💡 *This demo simulates how visual attention works in human-like systems, step by step.*")

show_footer()