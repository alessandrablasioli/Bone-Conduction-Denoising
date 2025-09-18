import os
import csv
import torch

project_root = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/testing/mos_bench_project"
data_root = os.path.join(project_root, "data")
output_csv = os.path.join(project_root, "mos_results.csv")


print("Loading SHEET MOS predictor...")
predictor = torch.hub.load("unilight/sheet:v0.1.0", "default", force_reload=True)

device = "cpu"
predictor.model.to(device)

def compute_mos(wav_path):
    try:
        score = predictor.predict(wav_path=wav_path)
        return float(score)
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

results = []
for condition in ["clean", "noisy", "enhanced"]:
    folder = os.path.join(data_root, condition)
    print(f"\nProcessing {condition} files in {folder}...")
    for file in sorted(os.listdir(folder)):
        if file.endswith(".wav"):
            wav_path = os.path.join(folder, file)
            print(f"Predicting MOS for {file}...")
            mos_score = compute_mos(wav_path)
            if mos_score is not None:
                results.append({
                    "file": f"{condition}/{file}",
                    "condition": condition,
                    "MOS_score": round(mos_score, 4)
                })
print(f"\nSaving results to {output_csv}...")
with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["file", "condition", "MOS_score"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Done! MOS results saved.")
