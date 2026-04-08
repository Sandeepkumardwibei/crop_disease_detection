from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

from model import HybridPlantModel

app = Flask(__name__)

device = torch.device("cpu")
os.makedirs("static", exist_ok=True)

# MODEL
model = HybridPlantModel(num_classes=38)
model.load_state_dict(torch.load("robust_model.pth", map_location=device, mmap=True, weights_only=True))
model.to(device)
model.eval()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# CLASSES
classes = [
'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
'Grape___Black_rot','Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot','Peach___healthy',
'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
'Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# DETAILED DISEASE INFO
disease_details = {
    "Apple___Apple_scab": """Cause:\n- Fungal infection in humid conditions\n- Poor air circulation\n- Wet leaves for long time\n\nPrecautions:\n- Remove infected leaves\n- Use fungicide regularly\n- Maintain proper spacing""",
    "Apple___Black_rot": """Cause:\n- Fungus attacks damaged tissues\n- Warm and moist weather\n- Infected fruits left on tree\n\nPrecautions:\n- Prune infected branches\n- Clean fallen fruits\n- Apply fungicide""",
    "Apple___Cedar_apple_rust": """Cause:\n- Fungus from cedar trees\n- Wind spreads spores\n- Moist weather\n\nPrecautions:\n- Remove nearby cedar plants\n- Use resistant varieties\n- Apply fungicide""",
    "Apple___healthy": """Healthy Leaf 🌿\n- No infection detected\n- Proper plant growth\n\nPrecautions:\n- Maintain watering schedule\n- Ensure sunlight exposure"""
}

def generate_info(disease_name):
    if disease_name in disease_details:
        return disease_details[disease_name]
    name = disease_name.replace("___", " ").replace("_", " ")
    if "healthy" in name.lower():
        return f"""Healthy Leaf 🌿\n\nCause:\n- No disease detected\n- Proper plant condition\n\nPrecautions:\n- Maintain watering\n- Provide sunlight"""
    return f"""Disease: {name}\n\nCause:\n- Fungal or bacterial infection\n- High humidity or poor care\n- Environmental stress\n\nPrecautions:\n- Remove infected leaves\n- Avoid overwatering\n- Use fungicide/pesticide"""

# ROUTE
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    heatmap_path = None
    info = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            img = Image.open(filepath).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)

            prediction_raw = classes[pred.item()]
            prediction = prediction_raw.replace("___", " - ").replace("_", " ")
            info = generate_info(prediction_raw)

            # FOR RENDER FREE TIER: Bypass heatmap generation to prevent 502 Output Timeout & Memory Limits
            heatmap_path = filepath
            image_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        info=info,
        image_path=image_path,
        heatmap_path=heatmap_path
    )

if __name__ == "__main__":
    app.run(debug=True)