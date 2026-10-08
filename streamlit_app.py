"""
Streamlit — Fourier-Based Document Demoiréing for Financial OCR.
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from dataclasses import dataclass


@dataclass
class DemoireConfig:
    notch_sigma: float = 4.0
    num_notches: int = 15
    dc_exclusion_radius: int = 25
    peak_percentile: float = 99.0
    nms_radius: int = 12
    bilateral_d: int = 7
    bilateral_sigma_color: float = 40.0
    bilateral_sigma_space: float = 40.0
    adaptive_block_size: int = 31
    adaptive_C: float = 10.0
    verbose: bool = False


def to_grayscale(image):
    if image is None:
        raise ValueError("Input is None.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported shape {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def compute_fft(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    return fshift, np.log1p(np.abs(fshift))


def _local_maxima_mask(arr, radius):
    h, w = arr.shape
    padded = np.full((h + 2 * radius, w + 2 * radius), -np.inf, dtype=arr.dtype)
    padded[radius:radius + h, radius:radius + w] = arr
    is_max = np.ones_like(arr, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[radius + dy: radius + dy + h,
                             radius + dx: radius + dx + w]
            is_max &= arr >= shifted
    return is_max


def detect_noise_peaks(log_mag, cfg):
    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist_sq = (Y - cy) ** 2 + (X - cx) ** 2
    masked = log_mag.copy()
    masked[dist_sq < cfg.dc_exclusion_radius ** 2] = 0.0
    lm = _local_maxima_mask(masked, radius=cfg.nms_radius)
    positives = masked[masked > 0]
    if positives.size == 0:
        return []
    thr = np.percentile(positives, cfg.peak_percentile)
    cand = np.argwhere(lm & (masked > thr))
    if cand.size == 0:
        return []
    vals = masked[cand[:, 0], cand[:, 1]]
    cand = cand[np.argsort(vals)[::-1]]
    peaks = []
    nms_sq = cfg.nms_radius ** 2
    for y, x in cand:
        if y > cy or (y == cy and x < cx):
            continue
        if any((y - sy) ** 2 + (x - sx) ** 2 < nms_sq for sy, sx in peaks):
            continue
        peaks.append((int(y), int(x)))
        if len(peaks) >= cfg.num_notches:
            break
    return peaks


def build_gaussian_notch_mask(shape, peaks, cfg):
    h, w = shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    sigma_sq2 = 2.0 * cfg.notch_sigma ** 2
    mask = np.ones((h, w), dtype=np.float32)
    for (py, px) in peaks:
        for (yy, xx) in ((py, px), (2 * cy - py, 2 * cx - px)):
            D_sq = (Y - yy) ** 2 + (X - xx) ** 2
            mask *= (1.0 - np.exp(-D_sq / sigma_sq2))
    return mask


def inverse_fft(fshift_filtered):
    f_ishift = np.fft.ifftshift(fshift_filtered)
    return np.abs(np.real(np.fft.ifft2(f_ishift))).astype(np.float32)


def postprocess(gray, cfg):
    d = cfg.bilateral_d if cfg.bilateral_d % 2 == 1 else cfg.bilateral_d + 1
    d = max(3, d)
    denoised = cv2.bilateralFilter(gray, d=d,
                                   sigmaColor=cfg.bilateral_sigma_color,
                                   sigmaSpace=cfg.bilateral_sigma_space)
    block = cfg.adaptive_block_size
    if block % 2 == 0:
        block += 1
    block = max(3, block)
    binary = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   blockSize=block, C=cfg.adaptive_C)
    return denoised, binary


def demoire_pipeline(bgr, cfg):
    gray = to_grayscale(bgr)
    fshift, log_mag = compute_fft(gray)
    peaks = detect_noise_peaks(log_mag, cfg)
    mask = build_gaussian_notch_mask(gray.shape, peaks, cfg)
    filtered_complex = fshift * mask
    restored = inverse_fft(filtered_complex)
    filtered_u8 = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    denoised, binary = postprocess(filtered_u8, cfg)
    return {
        "gray": gray,
        "log_mag": log_mag,
        "peaks": peaks,
        "filtered": filtered_u8,
        "denoised": denoised,
        "binary": binary,
    }


# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="Fourier Demoiréing for OCR",
                   page_icon="🔬", layout="wide")

st.title("🔬 Fourier-Based Document Demoiréing for Financial OCR")
st.markdown(
    "Remove periodic moiré artifacts from documents photographed off a "
    "screen. The pipeline detects noise spikes in the 2-D Fourier spectrum, "
    "applies Gaussian notch filters, and outputs an OCR-ready binary image."
)

with st.sidebar:
    st.header("⚙ Parameters")
    notch_sigma = st.slider("Notch σ (pixels)", 1.0, 12.0, 4.0, 0.5)
    num_notches = st.slider("Max notch pairs", 1, 40, 15)
    dc_radius = st.slider("DC exclusion radius", 5, 80, 25)
    percentile = st.slider("Peak percentile", 90.0, 99.9, 99.0, 0.1)
    nms_radius = st.slider("NMS radius", 3, 30, 12)
    bilateral_d = st.slider("Bilateral diameter", 3, 15, 7, 2)
    adaptive_block = st.slider("Adaptive block size", 11, 71, 31, 2)
    adaptive_C = st.slider("Adaptive C", 2.0, 30.0, 10.0, 0.5)

uploaded = st.file_uploader(
    "📤 Upload a moiré document (JPG / PNG)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if "bgr" not in st.session_state:
    st.session_state["bgr"] = None

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    rgb = np.array(pil)
    st.session_state["bgr"] = rgb[:, :, ::-1].copy()

if uploaded is None and st.session_state["bgr"] is None:
    st.info("Upload an image, or click **Use synthetic example** below.")
    if st.button("▶ Use synthetic example"):
        h, w = 700, 900
        canvas = np.full((h, w), 245, dtype=np.uint8)
        rng = np.random.default_rng(7)
        y = 60
        while y < h - 50:
            lw = int(rng.integers(250, w - 150))
            cv2.rectangle(canvas, (70, y), (70 + lw, y + 9), 25, -1)
            y += int(rng.integers(22, 38))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        g = (np.sin(2 * np.pi * (xx / 6.5 + yy / 42.0))
             + np.sin(2 * np.pi * (yy / 7.5 - xx / 58.0)))
        noisy = np.clip(canvas.astype(np.float32) + 28.0 * g, 0, 255).astype(np.uint8)
        st.session_state["bgr"] = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
        st.rerun()

if st.session_state["bgr"] is not None:
    if st.button("🚀 Run demoiréing pipeline", type="primary"):
        cfg = DemoireConfig(
            notch_sigma=notch_sigma,
            num_notches=num_notches,
            dc_exclusion_radius=dc_radius,
            peak_percentile=percentile,
            nms_radius=nms_radius,
            bilateral_d=bilateral_d,
            adaptive_block_size=adaptive_block,
            adaptive_C=adaptive_C,
            verbose=False,
        )
        with st.spinner("Running Fourier notch filtering..."):
            r = demoire_pipeline(st.session_state["bgr"], cfg)

        st.success(f"✅ Detected **{len(r['peaks'])}** notch pairs.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Fourier-Based Document Demoiréing Pipeline",
                     fontsize=14, fontweight="bold")

        axes[0, 0].imshow(r["gray"], cmap="gray")
        axes[0, 0].set_title("(a) Original moiré image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(r["log_mag"], cmap="viridis")
        axes[0, 1].set_title("(b) Spectrum + detected peaks")
        axes[0, 1].axis("off")
        if r["peaks"]:
            hh, ww = r["log_mag"].shape
            ys = [p[0] for p in r["peaks"]]
            xs = [p[1] for p in r["peaks"]]
            axes[0, 1].scatter(xs, ys, s=60, facecolors="none",
                               edgecolors="red", linewidths=1.6)
            axes[0, 1].scatter([ww - 1 - x for x in xs],
                               [hh - 1 - y for y in ys],
                               s=60, facecolors="none",
                               edgecolors="orange", linewidths=1.2)

        axes[1, 0].imshow(r["filtered"], cmap="gray")
        axes[1, 0].set_title("(c) Filtered (after IFFT)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(r["binary"], cmap="gray")
        axes[1, 1].set_title("(d) Binary — ready for OCR")
        axes[1, 1].axis("off")

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        st.pyplot(fig)

        st.subheader("⬇ Download outputs")
        c1, c2, c3 = st.columns(3)
        for col, key, label in [
            (c1, "filtered", "Filtered spatial"),
            (c2, "binary", "Binary OCR-ready"),
            (c3, "gray", "Grayscale input"),
        ]:
            buf = io.BytesIO()
            Image.fromarray(r[key]).save(buf, format="PNG")
            col.download_button(
                label=f"⬇ {label}.png",
                data=buf.getvalue(),
                file_name=f"{key}.png",
                mime="image/png",
            )

st.markdown("---")
st.caption(
    "Source: [GitHub repo]"
    "(https://github.com/KSatwik/Fourier-Based-Document-Demoir-ing-Pipeline-for-Financial-OCR)"
)
