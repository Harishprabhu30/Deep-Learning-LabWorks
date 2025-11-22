import torch
from PIL import Image
from torchvision import transforms
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128

# Transformations
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image_path):
    """Load and preprocess an image."""
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def list_images(folder_path):
    """
    Return a list of all JPG images in a folder.
    
    Args:
        folder_path (str): Path to folder containing images.
        
    Returns:
        list of str: Paths to all .jpg files in the folder.
    """
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith('.jpg')]
