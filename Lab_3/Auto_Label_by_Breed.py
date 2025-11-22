import os
import shutil

SOURCE = "/Users/harishprabhu/desktop/subjects/sem_3/deep_learning/lab_3/images"     # change to your folder
TARGET = "/Users/harishprabhu/desktop/subjects/sem_3/deep_learning/lab_3/data/train"

# Full cat breed names (lowercase)
cat_breeds = {
    "abyssinian", "bengal", "birman", "bombay",
    "british_shorthair", "egyptian_mau", "maine_coon",
    "persian", "ragdoll", "russian_blue",
    "siamese", "sphynx"
}

# Mapping shortened names → full breed names
breed_alias = {
    "british": "british_shorthair",
    "egyptian": "egyptian_mau",
    "maine": "maine_coon",
    "russian": "russian_blue"
}

detected_breeds = set()
cat_assigned = []
dog_assigned = []

os.makedirs(f"{TARGET}/cats", exist_ok=True)
os.makedirs(f"{TARGET}/dogs", exist_ok=True)

for fname in os.listdir(SOURCE):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Extract breed
    breed_original = fname.split("_")[0]  # e.g. Maine, British
    breed = breed_original.lower()

    # Apply alias mapping
    if breed in breed_alias:
        breed = breed_alias[breed]

    detected_breeds.add(breed_original)

    # Classification
    if breed in cat_breeds:
        shutil.copy(f"{SOURCE}/{fname}", f"{TARGET}/cats/{fname}")
        cat_assigned.append(breed_original)
    else:
        shutil.copy(f"{SOURCE}/{fname}", f"{TARGET}/dogs/{fname}")
        dog_assigned.append(breed_original)

# --- PRINT SUMMARY ---
print("\n=== BREEDS DETECTED IN YOUR FOLDER ===")
for b in sorted(detected_breeds):
    print(" -", b)

print("\n=== BREEDS CLASSIFIED AS CATS ===")
for b in sorted(set(cat_assigned)):
    print(" -", b)

print("\n=== BREEDS CLASSIFIED AS DOGS ===")
for b in sorted(set(dog_assigned)):
    print(" -", b)

print("\n--- DONE ---")
print("Images have been separated into cats/ and dogs/.")
