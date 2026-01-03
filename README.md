# 🌿 LeafGuard — Plant Disease Detection Web Application

LeafGuard aims to provide a practical, accessible tool to assist gardeners, farmers
and green-thumbed individuals with the early detection of plant diseases. By
enabling users to upload images of leaves and automatically identify common plant
diseases using computer vision techniques, the application can help users promptly
assess plant health, therefore mitigating crop damage and preventing further disease
spread.

## 🚀 Features

**- Upload leaf images via an intuitive user-friendly web interface**

**- Deep learning–based disease classification**

**- Confidence scores and top-3 predictions for interpretability**

**- Low-confidence warnings (threshold = 0.65) to highlight uncertain predictions**

**- End-to-end system (frontend + backend + model inference)**

**- Containerised using Docker**

## 🧩 System Architecture

**React + Tailwind UI -> FastAPI REST API -> PyTorch Model Inference -> JSON Response (class, confidence, top-3)**

_Frontend: React, Tailwind CSS_

_Backend: FastAPI (Python)_

_Model: PyTorch (Baseline CNN + ResNet18 via transfer learning)_

_Deployment: Docker, Render_

## 🧠 Machine Learning Models
### Baseline CNN

**- Custom CNN trained from scratch**

**- Used to establish a performance reference point**

**- Validation accuracy ≈ 85%**


### ResNet18 (Transfer Learning)

**- Pretrained ResNet18 fine-tuned on leaf disease data**

**- Significantly improved performance and confidence stability**

**- Test accuracy ≈ 94%**

**- Faster convergence (≈3 epochs)**


## Key Findings

### What works
Add me

## Live Demo

**Frontend demo (if deployed):
👉 https://leafguard.onrender.com**



A Postman collection is used for regression testing, including:

**- Valid image uploads**

**- Invalid file types**

**- Missing file handling**