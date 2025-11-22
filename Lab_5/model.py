import torch
from torch import optim
from torchvision import models
from utils import total_variation_loss, add_random_jitter, remove_jitter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained GoogLeNet
def load_model():
    model = models.googlenet(pretrained=True).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

# Deep Dream with jitter
def deep_dream_with_jitter(model, image, iterations, lr, layer_names_weights, tv_weight=1e-7, jitter=32):
    orig_shape = image.shape
    image = image.clone().requires_grad_(True).to(device)
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook

    for name, module in model.named_modules():
        if name in layer_names_weights:
            hooks.append(module.register_forward_hook(get_activation(name)))

    optimizer = optim.Adam([image], lr=lr)

    for i in range(iterations):
        optimizer.zero_grad()
        activations.clear()
        jittered_image, jitter_amounts = add_random_jitter(image, jitter)
        _ = model(jittered_image)
        losses = []
        for name, weight in layer_names_weights.items():
            if name in activations:
                activation = activations[name] ** 2
                losses.append(-weight * activation.mean())
        dream_loss = sum(losses)
        tv_loss = total_variation_loss(jittered_image[0])
        loss = dream_loss + tv_weight * tv_loss
        loss.backward()
        if image.grad is not None:
            image.grad.data = remove_jitter(image.grad.data, jitter_amounts, orig_shape)
        optimizer.step()
        image.data.clamp_(-1.5, 1.5)
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.4f}")

    for hook in hooks:
        hook.remove()

    return image.detach()

# Multi-scale Deep Dream
def deep_dream_multiscale_with_jitter(model, base_image, iterations, lr, layer_names_weights, num_octaves=5, octave_scale=1.4, jitter=32):
    import torch.nn as nn
    image = base_image.clone()
    octaves = []

    for i in range(num_octaves):
        scale_factor = octave_scale ** (-i)
        size = [int(dim * scale_factor) for dim in image.shape[-2:]]
        octave_image = nn.functional.interpolate(image, size=size, mode='bilinear', align_corners=False)
        octaves.append(octave_image)

    detail = torch.zeros_like(octaves[-1], device=device)

    for octave, octave_image in enumerate(reversed(octaves)):
        if detail.shape != octave_image.shape:
            detail = nn.functional.interpolate(detail, size=octave_image.shape[-2:], mode='bilinear', align_corners=False)
        input_image = octave_image + detail
        dreamed_image = deep_dream_with_jitter(model, input_image, iterations, lr, layer_names_weights,
                                              jitter=max(int(jitter * octave_scale**(-octave)), 8))
        detail = dreamed_image - octave_image

    return dreamed_image