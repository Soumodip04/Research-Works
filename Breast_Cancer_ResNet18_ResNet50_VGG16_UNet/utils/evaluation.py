import os
from PIL import Image
import torch
import numpy as np

def predict_classification(model, input_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()  # Convert logits to probabilities
        predicted_index = probabilities.argmax()  # Get the index of the class with the highest probability
    return predicted_index, probabilities  # Return the index and probabilities


def predict_segmentation(model, input_tensor, file_path, prediction_folder):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor.unsqueeze(0))  # Ensure the input tensor has a batch dimension
        probabilities = torch.softmax(outputs, dim=1)
        predicted_mask = outputs.argmax(1).squeeze().cpu().numpy()
        confidence = probabilities.max().item()  # Confidence score for segmentation


    # Save the predicted mask as an image
    predicted_image_path = os.path.join(prediction_folder, f"predicted_{os.path.basename(file_path)}")
    Image.fromarray((predicted_mask * 255).astype("uint8")).save(predicted_image_path)
    return "Segmentation result saved.", confidence, predicted_image_path  # Return confidence as well
