import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
import json

os.makedirs("evaluated_metrics", exist_ok=True)

def evaluate_model(model, val_loader, classes, model_name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion Matrix:")
    print(cm)

    metrics_file = os.path.join("evaluated_metrics", f"{model_name}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm.tolist()}, f, indent=4)
    print(f"Metrics saved to {metrics_file}")
