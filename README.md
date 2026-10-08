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
Application link - https://satwik-document-demoiring-financial-ocr.streamlit.app/
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

<img width="1440" height="855" alt="image" src="https://github.com/user-attachments/assets/8ad12ed9-29f5-4e02-9e48-35ce645577d9" />


*Panel (a) original moiré image · (b) log-magnitude spectrum with detected
peaks (red = detected, orange = conjugate-symmetric mirrors) · (c) notch-
filtered spatial reconstruction · (d) final binary ready for OCR.*

### Stage-by-Stage


<img width="1062" height="463" alt="image" src="https://github.com/user-attachments/assets/79302d77-656d-4cae-8cca-024b0022b4f4" />

<img width="1017" height="431" alt="image" src="https://github.com/user-attachments/assets/b8f855f3-4725-431d-8a6f-727ee721f2ee" />


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
H(u,v) = \prod_k \left[ 1 - \exp\left(-\frac{D_k(u,v)^2}{2\sigma^2}\right)\right]
$$

where $D_k(u,v)$ is the Euclidean distance from $(u,v)$ to the $k$-th
peak. Multiply `H` element-wise with the shifted FFT, then invert.

The soft Gaussian profile avoids ringing artifacts that a hard binary
mask would introduce — crucial for preserving thin character strokes.

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/KSatwik/Fourier-Based-Document-Demoir-ing-Pipeline-for-Financial-OCR
cd Fourier-Based-Document-Demoir-ing-Pipeline-for-Financial-OCR
pip install -r requirements.txt
```
### 2. Sample Images

Don't have a moiré document handy? Use this **Colab sample generator** to create realistic financial-document images replica (invoices, statements,tax forms) with configurable moiré interference — perfect for testing and demos.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eUddi-ZJtdsD8buPNE65GhNreZHOgXvj?usp=sharing)

**The notebook lets you:**
- Control moiré intensity, angle, and frequency
- Adjust page size and text density
- Download the generated PNG directly to your machine

**Typical workflow:**

1. Open the Colab notebook → adjust parameters → **Run all**
2. Download `financial_doc_moire.png` from the Files panel
3. Drag it into the live Streamlit demo, or run locally:
   ```bash
   python src/demoire.py --input financial_doc_moire.png
