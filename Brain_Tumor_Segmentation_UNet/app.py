from flask import Flask, render_template, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Load your model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)  # Define the same architecture as used in training
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 4)  # Update this to the number of your classes
)
model.load_state_dict(torch.load("E:\\Transfered from Local Disk F\\All Folders\\T Chow\\Brain Tumour Project\\best_model.pth"))
model = model.to(device)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Flask app
app = Flask(__name__)

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Classification route
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Process the uploaded image
    img = Image.open(file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        class_idx = probabilities.argmax()

    # Class names
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Update this with your classes
    result = {
        'class': class_names[class_idx],
        'confidence': float(probabilities[class_idx])
    }

    # Generate ROC curve
    y_true = [1 if i == class_idx else 0 for i in range(len(class_names))]
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot and save the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()

    # Save the plot to the static folder
    roc_curve_path = os.path.join('static', 'images', 'roc_curve.png')
    os.makedirs(os.path.dirname(roc_curve_path), exist_ok=True)
    plt.savefig(roc_curve_path)
    plt.close()

    # Redirect to results page with the ROC curve
    return render_template('results.html', result=result, roc_curve_url='/static/images/roc_curve.png')

if __name__ == '__main__':
    app.run(debug=True)
