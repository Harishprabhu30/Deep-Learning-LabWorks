import matplotlib.pyplot as plt
from torchvision import transforms

unloader = transforms.ToPILImage()  # Convert tensor to image

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title: plt.title(title)
    plt.pause(0.001)