import torch
import torch.nn.functional as F

# Deprocess tensor to displayable image
def deprocess(image_tensor):
    image = image_tensor.clone().detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image * std + mean
    image = image.clamp(0,1)
    image = image.squeeze(0).permute(1,2,0)
    return image.numpy()

# Total variation loss
def total_variation_loss(img):
    tv_loss = torch.sum(torch.abs(img[:, :, :-1] - img[:, :, 1:])) + \
              torch.sum(torch.abs(img[:, :-1, :] - img[:, 1:, :]))
    return tv_loss

# Random jitter functions
def add_random_jitter(image, jitter=32):
    b, c, h, w = image.shape
    padded = F.pad(image, (jitter,jitter,jitter,jitter), mode='reflect')
    dx = torch.randint(0, jitter*2, (1,))
    dy = torch.randint(0, jitter*2, (1,))
    jittered = padded[:, :, dy:dy+h, dx:dx+w]
    return jittered, (dx, dy)

def remove_jitter(image, jitter_amounts, orig_shape):
    dx, dy = jitter_amounts
    h, w = orig_shape[-2:]
    return image[:, :, :h, :w]