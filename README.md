# 🔬 Fourier-Based Document Demoiréing Pipeline for Financial OCR

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-orange)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Removing periodic moiré artifacts from screen-photographed financial
> documents so that OCR engines can reliably extract structured text.

---

## 📖 Overview

When a document is photographed off an LCD/LED monitor, the camera's
sensor aliases with the screen's pixel grid, producing **periodic
moiré interference** — light/dark bands that corrupt OCR output. This
project implements a **Fourier-domain notch-filtering pipeline** that
detects those periodic artifacts automatically and removes them, then
post-processes the image into a clean binary document ready for
structural text extraction (Tesseract, PaddleOCR, LayoutLM).

---

## 🎯 Problem → Solution

| Stage | Input | Output |
|-------|-------|--------|
| **Raw** | Moiré-contaminated financial document | — |
| **FFT** | Grayscale → 2-D FFT → fftshift | Centred spectrum |
| **Detect** | Log-magnitude spectrum | Auto-detected peaks |
| **Filter** | Gaussian notch mask | Attenuated spectrum |
| **Reconstruct** | IFFT + \|real\| | Clean spatial image |
| **Binarize** | Adaptive threshold | OCR-ready binary |

---

## 🖼️ Results

### Verification — 4-Panel Comparison

![4-Panel Verification](results/04_verification_4panel.png)

*Panel (a) original moiré image · (b) log-magnitude spectrum with detected
peaks (red = detected, orange = conjugate-symmetric mirrors) · (c) notch-
filtered spatial reconstruction · (d) final binary ready for OCR.*

### Stage-by-Stage

| Input (Moiré) | Filtered | Binary OCR-Ready |
|:---:|:---:|:---:|
| ![](results/01_input_moire.png) | ![](results/02_filtered_spatial.png) | ![](results/03_binary_ocr_ready.png) |

---

## ⚙️ Pipeline

```text
[ Raw Image with Moiré ]
             │
             ▼
┌──────────────────────────┐
│ 1. Frequency Demoiréing  │ ──► FFT ➔ Notch/Band-Stop Filtering ➔ IFFT
└──────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│ 2. Layout & Segmentation │ ──► Deskew, Binarization, ROI Detection
└──────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│  3. Multi-Engine OCR     │ ──► Hybrid Deep-Learning Text Extraction
└──────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│ 4. Semantic Validation   │ ──► Named Entity Recognition (NER) & Schema Rules
└──────────────────────────┘
             │
             ▼
  [ Verified JSON Payload ]
```


## 🧮 Theory in 60 Seconds

A moiré pattern is a **periodic spatial interference** — in the Fourier
domain it appears as **conjugate-symmetric spike pairs** away from DC.
We exploit this:

- **DC term** = document's own low-frequency structure (background, layout).
- **Spikes** = moiré gratings.
- **Notches** = Gaussian attenuation applied at each spike location.

Mathematically, for each detected peak $(u_k, v_k)$ and its mirror
$(u'_k, v'_k)$:

$$
H(u,v) = \prod_k \left[ 1 - \exp\!\left(-\frac{D_k(u,v)^2}{2\sigma^2}\right)\right]
$$

where $D_k(u,v)$ is the Euclidean distance from $(u,v)$ to the $k$-th
peak. Multiply `H` element-wise with the shifted FFT, then invert.

The soft Gaussian profile avoids ringing artifacts that a hard binary
mask would introduce — crucial for preserving thin character strokes.

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/KSatwik/
cd fourier-demoire-ocr
pip install -r requirements.txt
