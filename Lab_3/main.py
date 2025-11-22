import time
from data_loader import create_dataloaders
from models import build_model
from train import train_model
from evaluate import evaluate_model

print('Loading data...')
train_loader, val_loader, classes = create_dataloaders(img_size=224, batch_size=32)
print('Data loaded.')

for model_name in ["mobilenet", "efficientnet", "nasanet"]:
    print(f"\n===== Training {model_name.upper()} =====")
    model = build_model(model_name)

    start_time = time.time()
    trained_model = train_model(model, train_loader, val_loader, epochs=5)
    end_time = time.time()

    elapsed = end_time - start_time
    avg_epoch_time = elapsed / 5
    print(f"{model_name.upper()} trained.")
    print(f"Total training time: {elapsed:.2f} seconds")
    print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

    print('Evaluating model...')
    evaluate_model(trained_model, val_loader, classes)
