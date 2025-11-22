import matplotlib.pyplot as plt
import os
from PIL import Image
from data_loader import load_images
from model import load_model, deep_dream_multiscale_with_jitter
from utils import deprocess

# Parameters
iterations = 50
lr = 0.01
num_octaves = 5
octave_scale = 1.4
layer_names_weights = {'inception4a': 1.0}

# Folder setup
image_folder = "data"
output_folder = "Lab_5/model_images"
dreamed_output_folder = "Lab_5/data/output"
os.makedirs(output_folder, exist_ok=True)

# Load model and images
model = load_model()
images = load_images(image_folder)

for fname, input_image in images:
    print(f"\nProcessing {fname}")
    dreamed_image, loss_history = deep_dream_multiscale_with_jitter(model, input_image, iterations, lr, layer_names_weights, num_octaves, octave_scale)
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
