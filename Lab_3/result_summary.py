import os
import json
import pandas as pd

METRICS_DIR = "evaluated_metrics"

# Load training times
training_times_file = os.path.join(METRICS_DIR, "training_times.json")
if os.path.exists(training_times_file):
    with open(training_times_file, "r") as f:
        training_times = json.load(f)
else:
    training_times = {}

# List all metrics JSON files
metric_files = [f for f in os.listdir(METRICS_DIR) if f.endswith("_metrics.json")]
metric_files.sort()

summary_data = []

for file in metric_files:
    model_name = file.replace("_metrics.json", "")
    with open(os.path.join(METRICS_DIR, file), "r") as f:
        metrics = json.load(f)

    report = metrics["classification_report"]
    accuracy = report.get("accuracy", None)
    f1_scores = {cls: report[cls]["f1-score"] for cls in report if cls not in ["accuracy", "macro avg", "weighted avg"]}

    # Add training time if available
    total_time = training_times.get(model_name, {}).get("total_time_sec", None)
    avg_epoch_time = training_times.get(model_name, {}).get("avg_epoch_time_sec", None)

    summary_data.append({
        "Model": model_name,
        "Validation Accuracy": accuracy,
        "Total Training Time (s)": total_time,
        "Avg Epoch Time (s)": avg_epoch_time,
        "F1 Scores": f1_scores
    })

df = pd.DataFrame(summary_data)
df.to_csv("model_results_summary.csv", index=False)
print("Summary table saved as 'model_results_summary.csv'")

print("\n=== Model Comparison Summary ===")
print(df)
