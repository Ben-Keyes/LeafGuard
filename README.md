# 🌿 LeafGuard — Plant Disease Detection Web Application

LeafGuard aims to provide a practical, accessible tool to assist gardeners, farmers
and green-thumbed individuals with the early detection of plant diseases. By
enabling users to upload images of leaves and automatically identify common plant
diseases using computer vision techniques, the application can help users promptly
assess plant health, therefore mitigating crop damage and preventing further disease
spread.

## 🚀 Features

- **Upload leaf images via a user-friendly web interface**

- **Deep learning–based disease classification**

- **Confidence scores and top-3 predictions for interpretability**

- **Low-confidence warnings (threshold = 0.50) to highlight uncertain predictions**

- **End-to-end system (frontend + backend + model inference)**

- **Containerised using Docker**

## 🧩 System Architecture

**Frontend (React + Tailwind) → Backend (FastAPI) → Model (PyTorch ResNet50) → JSON Response**

_Frontend: React, Tailwind CSS_

_Backend: FastAPI (Python)_

_Model: PyTorch (ResNet50)_

_Deployment: Docker, Render_

## 🧠 Machine Learning Model

### ResNet50 (Transfer Learning)

- **Pretrained ResNet50 weights fine-tuned on leaf disease data [PlantVillage dataset]**

- **Custom validation dataset**

- **Generalisation Top-1 accuracy ≈ 40%**

- **Generalisation Top-3 accuracy ≈ 57%**

- **PlantVillage Training Accuracy ≈ 99%**

### Datasets

- **Custom Validation Dataset:** [OneDrive link to zipped folder](https://qubstudentcloud-my.sharepoint.com/:u:/g/personal/40365576_ads_qub_ac_uk/IQD6c6Y2VIveRJg0kBLXLU82ASNj4pZHKggWc95ASmHkRU0?e=eZOZcG)

- **PlantVillage Dataset:** [OneDrive Link to zipped folder](https://qubstudentcloud-my.sharepoint.com/:u:/g/personal/40365576_ads_qub_ac_uk/IQDuVi3sjtm9Q7eKS3jXml46AUjLvVIJ1du1QV6Vxw276Fw?e=hOa9cl)

## 🔬 Key Findings

### What works
- Custom validation dataset informed finetuning of model
- Top-3 predictions improve usability despite low Top-1 accuracy
- Confidence thresholds help prevent misleading predictions

### Limitations
- Model struggles with real-world images outside the training domain
- High training accuracy indicates overfitting to PlantVillage dataset

## Live Demo

**Frontend demo (May take a minute to deploy as it is 3rd-party hosted):
👉 https://leafguard.onrender.com**

_NB: Inference times are significantly greater than when run locally_

## Local Run

To build and run the project locally:

```bash
cd Leafguard/frontend
npm run build

cd ..
docker build -t leafguard .

docker run -p 8000:8000 leafguard

```

## 💡 What this does
- `npm run build` → builds the frontend  
- `docker build` → creates a container  
- `docker run` → runs the app on **http://localhost:8000**


## Testing 
A Postman collection is used for regression testing, including:

- **Valid image uploads**

- **Invalid file types**

- **Missing file handling**