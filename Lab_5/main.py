import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
import torch
from data_loader import load_images
from model import load_model, deep_dream_multiscale_with_jitter
from utils import deprocess

# Parameters
iterations = 20
lr = 0.008
num_octaves = 3
octave_scale = 1.0
layer_names_weights = {
    'inception3a': 0.1,
    'inception4a': 0.2,
    'inception4d': 0.3,
    'inception4e': 0.4
}

# Folder setup
image_folder = "/home/vgtu/Downloads/Harish_Thesis/Deep-Learning-LabWorks/Lab_5/data"
output_folder = "/home/vgtu/Downloads/Harish_Thesis/Deep-Learning-LabWorks/Lab_5/model_images"
dreamed_output_folder = "/home/vgtu/Downloads/Harish_Thesis/Deep-Learning-LabWorks/Lab_5/data/output"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(dreamed_output_folder, exist_ok=True)

# Image resize transform to reduce GPU memory usage
resize_transform = transforms.Resize((512, 512))  # reduce large images to 512x512

# Load model and images
print('Loading Model...')
model = load_model()
print('Model Loaded.')
images = load_images(image_folder)

for fname, input_image in images:
    print(f"\nProcessing {fname}")

    # Resize input image to prevent CUDA OOM
    pil_image = transforms.ToPILImage()(input_image.squeeze(0).cpu())  # convert tensor to PIL
    pil_image = resize_transform(pil_image)
    input_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Generate DeepDream image and loss history
    dreamed_image, loss_history = deep_dream_multiscale_with_jitter(
        model, input_image, iterations, lr, layer_names_weights, num_octaves, octave_scale
    )
    result = deprocess(dreamed_image)

    # Save dreamed image
    dreamed_fname = os.path.join(dreamed_output_folder, f"{os.path.splitext(fname)[0]}_dreamed.jpg")
    dreamed_pil = Image.fromarray((result * 255).astype('uint8'))
    dreamed_pil.save(dreamed_fname)
    print(f"Saved dreamed image as {dreamed_fname}")

    # Plot and save loss graph
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, label="Dream Loss + TV Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {fname}")
    plt.legend()
    plot_fname = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_loss.jpg")
    plt.savefig(plot_fname)
    plt.close()
    print(f"Saved loss plot as {plot_fname}")

    # Free GPU memory after each image to prevent OOM
    torch.cuda.empty_cache()
