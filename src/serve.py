import io
import json
from pathlib import Path

import torch
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms

from model import get_model

app = Flask(__name__)

CHECKPOINT_PATH = Path("/app/checkpoints/classifier_v1.pt")
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = Path("app/checkpoints/classifier_v1.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

model = None

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def load_model():
    global model
    m = get_model(architecture="resnet18", num_classes=10)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.to(DEVICE)
    m.eval()
    model = m


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({"status": "not_ready"}), 503
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).tolist()

    result = {CLASSES[i]: round(p, 4) for i, p in enumerate(probs)}
    return jsonify(result), 200


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8080)