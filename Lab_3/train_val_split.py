import os, shutil, random

SOURCE = "E:\\Studies\\Mechanical\\Additional\\Harish\\Deep-Learning-LabWorks\\Lab_3\\data\\train"
VAL = "E:\\Studies\\Mechanical\\Additional\\Harish\\Deep-Learning-LabWorks\\Lab_3\\data\\val"

SPLIT = 0.2  # 20%

os.makedirs(f"{VAL}/cats", exist_ok=True)
os.makedirs(f"{VAL}/dogs", exist_ok=True)

for cls in ["cats", "dogs"]:
    files = os.listdir(f"{SOURCE}/{cls}")
    random.shuffle(files)
    val_count = int(len(files) * SPLIT)
    val_files = files[:val_count]

    for f in val_files:
        shutil.move(f"{SOURCE}/{cls}/{f}", f"{VAL}/{cls}/{f}")

print("Validation split created.")