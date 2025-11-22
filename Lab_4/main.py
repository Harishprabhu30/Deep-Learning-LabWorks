import torch
import os
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from data_loader import image_loader, device, list_images
from model import get_style_model_and_losses
from utils import imshow, plot_loss_history
import torch.optim as optim
import matplotlib.pyplot as plt
from losses import ContentLoss, StyleLoss

# ----- FOLDERS -----
content_img_path = './data/images/mycontent.jpg'  # manually changeable
style_folder = './data/styles/'
output_folder = './data/output/'
plot_folder = './Lab_4/model_images/'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

# ----- LOAD CONTENT IMAGE -----
content_img = image_loader(content_img_path)
input_img_original = content_img.clone()  # reset for each style

# ----- NORMALIZATION -----
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# ----- LOAD PRETRAINED VGG19 -----
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

# ----- LIST STYLE IMAGES -----
style_images = list_images(style_folder)

# ----- STYLE TRANSFER PARAMETERS -----
num_steps = 300
style_weight = 1e6
content_weight = 1

# ----- LOOP OVER STYLE IMAGES -----
for s_path in style_images:
    print(f"\nProcessing style image: {os.path.basename(s_path)}")
    
    style_img = image_loader(s_path)
    input_img = input_img_original.clone()
    
    # Build model
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img
    )
    
    # Optimizer
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    # Track losses
    style_loss_history = []
    content_loss_history = []

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
            style_loss_history.append(style_score.item())
            content_loss_history.append(content_score.item())
            
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss {style_score.item():.4f}, Content Loss {content_score.item():.4f}")
            return loss
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0,1)

    # Show result
    plt.figure()
    imshow(input_img, title=f'Output - {os.path.basename(s_path)}')
    plt.show()

    # Save stylized image
    c_name = os.path.splitext(os.path.basename(content_img_path))[0]
    s_name = os.path.splitext(os.path.basename(s_path))[0]
    output_path = os.path.join(output_folder, f'{c_name}_stylized_{s_name}.jpg')
    with torch.no_grad():
        save_image(input_img.clamp(0,1), output_path)
    print(f"Saved stylized image: {output_path}")

    # Save loss plots
    plot_loss_history(style_loss_history, content_loss_history,
                      os.path.join(plot_folder, f'{c_name}_{s_name}_losses.png'))
    print(f"Saved loss plot: {c_name}_{s_name}_losses.png")
