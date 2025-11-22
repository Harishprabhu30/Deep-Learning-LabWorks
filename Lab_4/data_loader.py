import torch
from PIL import Image, UnidentifiedImageError, ImageFile
from torchvision import transforms
import os

# Fix for truncated / corrupted JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128

# Transformations
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image_path):
    """Load and preprocess an image safely."""
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, UnidentifiedImageError) as e:
        print(f"\n[ERROR] Could not load image: {image_path}")
        print("[DETAILS]", e)
        raise SystemExit("[STOPPED] Bad image file encountered.")

    # Apply transforms
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def list_images(folder_path):
    """Return list of all jpg images in a folder."""
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.jpg')
    ]
