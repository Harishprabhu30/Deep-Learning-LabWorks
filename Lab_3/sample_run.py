import os
import random
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from models import build_model
from data_loader import create_dataloaders
from evaluate import evaluate_model

MODEL_NAME = "efficientnet"  # choose model
IMG_SIZE = 224
NUM_SAMPLES = 10
SAMPLE_SAVE_DIR = "sample_predictions"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAMPLE_SAVE_DIR, exist_ok=True)

train_loader, val_loader, classes = create_dataloaders(img_size=IMG_SIZE, batch_size=32)

model = build_model(MODEL_NAME).to(DEVICE)
model_path = os.path.join("saved_models", f"{MODEL_NAME}_trained.pth")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print(f"Loaded trained model from {model_path}")

val_dataset = val_loader.dataset
indices = random.sample(range(len(val_dataset)), NUM_SAMPLES)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

print(f"Saving {NUM_SAMPLES} sample predictions to '{SAMPLE_SAVE_DIR}'...")

with torch.no_grad():
    for idx in indices:
        img_path, true_label_idx = val_dataset.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        output = model(input_tensor)
        _, pred_idx = torch.max(output, 1)
        pred_label = classes[pred_idx.item()]
        true_label = classes[true_label_idx]

        plt.figure()
        plt.imshow(img)
        plt.title(f"True: {true_label} | Pred: {pred_label}")
        plt.axis("off")
        img_name = os.path.basename(img_path)
        plt.savefig(os.path.join(SAMPLE_SAVE_DIR, f"{MODEL_NAME}_{img_name}"))
        plt.close()

print("Sample predictions saved successfully.")

# Evaluate on full validation set and save metrics
evaluate_model(model, val_loader, classes, model_name=MODEL_NAME)
