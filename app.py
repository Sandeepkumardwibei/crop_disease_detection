from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

from model import HybridPlantModel

app = Flask(__name__)

# Device
device = torch.device("cpu")

# Load model
model = HybridPlantModel(num_classes=38)
model.load_state_dict(torch.load("robust_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Classes 
classes = [
'Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',
'Blueberry___healthy',
'Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy',
'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot',
'Peach___healthy',
'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Raspberry___healthy',
'Soybean___healthy',
'Squash___Powdery_mildew',
'Strawberry___Leaf_scorch',
'Strawberry___healthy',
'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus',
'Tomato___healthy'
]

@app.route("/", methods=["GET", "POST"])
def index():
    
    prediction = None
    image_path = None
    heatmap_path = None
    
    if request.method == "POST":
        
        file = request.files["file"]
        
        if file:
            
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            
            img = Image.open(filepath).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            # Prediction
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            
            prediction = classes[pred.item()]
            
            # Dummy CAM (we'll refine later)
            cam = np.random.rand(224,224)
            
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam),
                cv2.COLORMAP_JET
            )
            
            overlay = heatmap / 255.0
            
            heatmap_path = os.path.join("static", "heatmap.png")
            plt.imsave(heatmap_path, overlay)
            
            image_path = filepath
    
    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        heatmap_path=heatmap_path
    )

if __name__ == "__main__":
    app.run(debug=True)