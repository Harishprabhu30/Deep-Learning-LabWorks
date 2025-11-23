import os
import time
import matplotlib.pyplot as plt
import torch
import json
from data_loader import create_dataloaders
from models import build_model
from train import train_model
from evaluate import evaluate_model

# Create folders
os.makedirs("model_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("evaluated_metrics", exist_ok=True)

# File to store training times
training_times_file = os.path.join("evaluated_metrics", "training_times.json")
training_times = {}

print('Loading data...')
train_loader, val_loader, classes = create_dataloaders(img_size=224, batch_size=32)
print('Data loaded.')

for model_name in ["mobilenet", "efficientnet", "nasanet"]:
    print(f"\n===== Training {model_name.upper()} =====")
    model = build_model(model_name)

    start_time = time.time()
    trained_model, history = train_model(model, train_loader, val_loader, epochs=5)
    end_time = time.time()

    elapsed = end_time - start_time
    avg_epoch_time = elapsed / 5
    print(f"{model_name.upper()} trained.")
    print(f"Total training time: {elapsed:.2f} seconds")
    print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

    # Save training time
    training_times[model_name] = {
        "total_time_sec": elapsed,
        "avg_epoch_time_sec": avg_epoch_time
    }
    with open(training_times_file, "w") as f:
        json.dump(training_times, f, indent=4)
    print(f"Training time saved to {training_times_file}")

    # Save trained model weights
    model_save_path = os.path.join("saved_models", f"{model_name}_trained.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

    # Evaluate and save metrics
    evaluate_model(trained_model, val_loader, classes, model_name=model_name)

    # Save loss & accuracy plots
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{model_name.upper()} Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"model_images/{model_name}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.title(f"{model_name.upper()} Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"model_images/{model_name}_accuracy.png")
    plt.close()

    print(f"Plots saved for {model_name.upper()} in 'model_images/'")
