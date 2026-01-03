from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import os, io
from torchvision import models
import torch.nn as nn
import json

app = FastAPI()

# CORS dependency
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ML setup
num_classes = 39
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The model we intend to use
model_path = os.path.join(BASE_DIR, "model", "leaf_resnet18_best_layer4_fc.pth")

# Using ResNet18
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
model.to("cpu")

# Transformations (augmenting dataset)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes_path = os.path.join(BASE_DIR, "model", "class_names.json")
with open(classes_path, "r") as f:
    classes = json.load(f)

# Confidence threshold
THRESHOLD = 0.65

# API routing
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        # Top 3 predictions
        topk_probs, topk_indices = torch.topk(probs, k=3, dim=1)

        top3 = [
            {
                "class_index": int(idx.item()),
                "class_name": classes[int(idx.item())],
                "confidence": float(prob.item())
            }
            for idx, prob in zip(topk_indices[0], topk_probs[0])
        ]

    best_conf = top3[0]["confidence"]
    is_confident = best_conf >= THRESHOLD
    # JSON response passed to FE
    return {
        "predicted_class": top3[0]["class_index"],
        "class_name": top3[0]["class_name"],
        "confidence": best_conf,
        "is_confident": is_confident,
        "threshold": THRESHOLD,
        "top3": top3
    }

# Serving the FE
frontend_dir = os.path.join(BASE_DIR, "static", "frontend")

# Adding assets for design of UI
assets_dir = os.path.join(BASE_DIR, "assets")
if os.path.isdir(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
