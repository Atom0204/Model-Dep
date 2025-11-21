import torch
import torch.nn as nn
from torchvision import models, transforms

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

import gdown
import os
import json

# ----------------------------------------------------
# 1. Google Drive Model Download
# ----------------------------------------------------

MODEL_PATH = "best_model.pth"
CLASS_MAP_PATH = "class_mapping.json"

# Put your Google Drive **DIRECT download** link here
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1ZPIDoczoIxYgFQLlZc-Fs6rtUfupd9MS"

def download_model():
    """
    Downloads the model from Google Drive if not present.
    """
    if not os.path.exists(MODEL_PATH):
        print("⏳ Downloading model from Google Drive...")
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded.")

# Download model at startup
download_model()


# ----------------------------------------------------
# 2. Load Class Mapping
# ----------------------------------------------------

with open(CLASS_MAP_PATH, "r") as f:
    class_mapping = json.load(f)

num_classes = len(class_mapping)


# ----------------------------------------------------
# 3. Build Model (ResNet50)
# ----------------------------------------------------

def build_model(num_classes):
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

model = build_model(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print("✅ Model loaded successfully.")


# ----------------------------------------------------
# 4. Preprocessing Transform
# ----------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ----------------------------------------------------
# 5. FastAPI App
# ----------------------------------------------------

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Food Classification API running successfully!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        class_name = class_mapping[str(predicted.item())]

        return {"prediction": class_name}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
