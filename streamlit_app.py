import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from src.demoire import demoire_pipeline, DemoireConfig

st.set_page_config(page_title="Fourier Demoiréing for OCR", page_icon="🔬", layout="wide")
st.title("🔬 Fourier-Based Document Demoiréing for Financial OCR")
st.markdown("Remove periodic moiré artifacts from documents photographed off a screen. The pipeline detects noise spikes in the 2-D Fourier spectrum, applies Gaussian notch filters, and outputs an OCR-ready binary image.")

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

uploaded = st.file_uploader("📤 Upload a moiré document (JPG / PNG)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded is None:
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
        g = (np.sin(2 * np.pi * (xx / 6.5 + yy / 42.0)) + np.sin(2 * np.pi * (yy / 7.5 - xx / 58.0)))
        noisy = np.clip(canvas.astype(np.float32) + 28.0 * g, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
        st.session_state["bgr"] = bgr
    st.stop()

pil = Image.open(uploaded).convert("RGB")
rgb = np.array(pil)
bgr = rgb[:, :, ::-1].copy()
st.session_state["bgr"] = bgr

bgr = st.session_state.get("bgr")

if st.button("🚀 Run demoiréing pipeline", type="primary"):
    if bgr is None:
        st.warning("Please upload an image or use the synthetic example first.")
        st.stop()
    cfg = DemoireConfig(
        notch_sigma=notch_sigma, num_notches=num_notches, dc_exclusion_radius=dc_radius,
        peak_percentile=percentile, nms_radius=nms_radius, bilateral_d=bilateral_d,
        adaptive_block_size=adaptive_block, adaptive_C=adaptive_C, verbose=False,
    )
    with st.spinner("Running Fourier notch filtering..."):
        r = demoire_pipeline(bgr, cfg)
    st.success(f"✅ Detected **{len(r['peaks'])}** notch pairs.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Fourier-Based Document Demoiréing Pipeline", fontsize=14, fontweight="bold")
    axes[0, 0].imshow(r["gray"], cmap="gray"); axes[0, 0].set_title("(a) Original moiré image"); axes[0, 0].axis("off")
    axes[0, 1].imshow(r["log_mag"], cmap="viridis"); axes[0, 1].set_title("(b) Spectrum + detected peaks"); axes[0, 1].axis("off")
    if r["peaks"]:
        h, w = r["log_mag"].shape
        ys = [p[0] for p in r["peaks"]]; xs = [p[1] for p in r["peaks"]]
        axes[0, 1].scatter(xs, ys, s=60, facecolors="none", edgecolors="red", linewidths=1.6)
        axes[0, 1].scatter([w - 1 - x for x in xs], [h - 1 - y for y in ys], s=60, facecolors="none", edgecolors="orange", linewidths=1.2)
    axes[1, 0].imshow(r["filtered"], cmap="gray"); axes[1, 0].set_title("(c) Filtered (after IFFT)"); axes[1, 0].axis("off")
    axes[1, 1].imshow(r["binary"], cmap="gray"); axes[1, 1].set_title("(d) Binary — ready for OCR"); axes[1, 1].axis("off")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    st.pyplot(fig)
    st.subheader("⬇ Download outputs")
    c1, c2, c3 = st.columns(3)
    for col, key, label in [(c1, "filtered", "Filtered spatial"), (c2, "binary", "Binary OCR-ready"), (c3, "gray", "Grayscale input")]:
        buf = io.BytesIO(); Image.fromarray(r[key]).save(buf, format="PNG")
        col.download_button(label=f"⬇ {label}.png", data=buf.getvalue(), file_name=f"{key}.png", mime="image/png")

st.markdown("---")
st.caption("Source: [GitHub repo](https://github.com/KSatwik/Fourier-Based-Document-Demoir-ing-Pipeline-for-Financial-OCR)")
