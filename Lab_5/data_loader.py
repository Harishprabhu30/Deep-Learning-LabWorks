import os
from PIL import Image
import torch
from torchvision import transforms

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_images(folder_path):
    """
    Load all JPG images from a folder and preprocess them.
    Returns list of (filename, tensor) tuples.
    """
    images = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".jpg"):
            path = os.path.join(folder_path, fname)
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            images.append((fname, tensor))
    return images