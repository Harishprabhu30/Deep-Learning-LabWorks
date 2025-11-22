import torch
import os
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from data_loader import image_loader, device
from model import get_style_model_and_losses
from utils import imshow
import torch.optim as optim
from losses import ContentLoss, StyleLoss

# Load images
content_img = image_loader('./images/mycontent.jpg')
style_img = image_loader('./styles/mystyle.jpg')
input_img = content_img.clone()  # Or random noise

# Normalization values
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Load pretrained VGG19
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

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
            print(f"Step {run[0]}: Style Loss {style_score.item()}, Content Loss {content_score.item()}")
        return loss
    optimizer.step(closure)

with torch.no_grad():
    input_img.clamp_(0,1)

# Show result
import matplotlib.pyplot as plt
plt.figure()
imshow(input_img, title='Output Image')
plt.show()

# Folder to save output images
output_folder = './output/'
os.makedirs(output_folder, exist_ok=True)

# Save the output image
output_path = os.path.join(output_folder, 'stylized_image.jpg')
# Clamp to [0,1] before saving
with torch.no_grad():
    save_image(input_img.clamp(0,1), output_path)

print(f"Stylized image saved at: {output_path}")
