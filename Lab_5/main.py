import matplotlib.pyplot as plt
from PIL import Image
import os
from data_loader import load_images
from model import load_model, deep_dream_multiscale_with_jitter
from utils import deprocess

# Parameters
iterations = 50
lr = 0.01
num_octaves = 5
octave_scale = 1.4
layer_names_weights = {'inception4a': 1.0}

# Load model
model = load_model()

# Load images
image_folder = "images"
images = load_images(image_folder)

# Loop through images and apply Deep Dream
for fname, input_image in images:
    print(f"\nProcessing {fname}")
    dreamed_image = deep_dream_multiscale_with_jitter(model, input_image, iterations, lr, layer_names_weights, num_octaves, octave_scale)
    result = deprocess(dreamed_image)

    # Display
    plt.figure(figsize=(8,8))
    plt.imshow(result)
    plt.axis('off')
    plt.title(fname)
    plt.show()

    # Save the result
    dreamed_fname = os.path.join(image_folder, f"{os.path.splitext(fname)[0]}_dreamed.jpg")
    dreamed_pil = Image.fromarray((result * 255).astype('uint8'))
    dreamed_pil.save(dreamed_fname)
    print(f"Saved dreamed image as {dreamed_fname}")
