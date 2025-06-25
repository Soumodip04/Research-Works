import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(image_path, model_name):
    if model_name == "unet":
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    
    # Add a batch dimension
    return tensor.unsqueeze(0)
