import torch
import os
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from data_loader import image_loader, device, list_images  # list_images lists all JPGs in a folder
from model import get_style_model_and_losses
from utils import imshow
import torch.optim as optim
from losses import ContentLoss, StyleLoss
import matplotlib.pyplot as plt

# ----- USER SET CONTENT IMAGE -----
content_img_path = './data/images/mycontent.jpg'
content_img = image_loader(content_img_path)
input_img_original = content_img.clone()  # Keep a copy to reset for each style

# Normalization values
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Load pretrained VGG19
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

# Output folder inside data/
output_folder = './data/output/'
os.makedirs(output_folder, exist_ok=True)

# Styles folder
style_folder = './data/styles/'
style_images = list_images(style_folder)  # list all jpg files in styles folder

# Loop over all style images
for s_path in style_images:
    style_img = image_loader(s_path)
    input_img = input_img_original.clone()  # Reset input image for each style

    # Build model with loss layers
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img
    )

    # Optimizer
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    # Run style transfer
    num_steps = 300
    style_weight = 1e6
    content_weight = 1

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]} ({os.path.basename(s_path)}): "
                      f"Style Loss {style_score.item():.4f}, Content Loss {content_score.item():.4f}")
            return loss
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0,1)

    # Show result
    plt.figure()
    imshow(input_img, title=f'Output - {os.path.basename(s_path)}')
    plt.show()

    # Save result
    c_name = os.path.splitext(os.path.basename(content_img_path))[0]
    s_name = os.path.splitext(os.path.basename(s_path))[0]
    output_path = os.path.join(output_folder, f'{c_name}_stylized_{s_name}.jpg')
    with torch.no_grad():
        save_image(input_img.clamp(0,1), output_path)
    print(f"Saved stylized image: {output_path}")
