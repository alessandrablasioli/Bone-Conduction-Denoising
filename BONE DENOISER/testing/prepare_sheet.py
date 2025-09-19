# Authored by Alessandra Blasioli
import os
import shutil
import json
import yaml
import soundfile as sf
import librosa

clean_dir = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/data/ac"
noisy_dir = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/inference2/demucsdouble_best.th"
enhanced_dir = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/inference2/demucsdouble_best.th"
output_dir = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/testing/mos_bench_project"

TARGET_SR = 16000

data_folders = {
    "Clean": os.path.join(output_dir, "data", "clean"),
    "Noisy": os.path.join(output_dir, "data", "noisy"),
    "Enhanced": os.path.join(output_dir, "data", "enhanced"),
}
for folder in data_folders.values():
    os.makedirs(folder, exist_ok=True)

def resample_and_save(src_path, dst_path, target_sr=TARGET_SR):
    try:
        audio, sr = librosa.load(src_path, sr=None, mono=True) 
        if sr != target_sr:
            print(f"Resampling {src_path} from {sr} Hz to {target_sr} Hz")
        else:
            print(f"Converting {src_path} to mono {target_sr} Hz")
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sf.write(dst_path, audio_resampled, target_sr)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")

metadata = {"samples": []}

for file in os.listdir(noisy_dir):
    if file.endswith("_noisy.wav"):
        name_parts = file.replace("_noisy.wav", "").split("_")
        base_name = "_".join(name_parts[:3])
        clean_name = f"{base_name}.wav"
        enhanced_name = file.replace("_noisy.wav", "_enhanced.wav")
        clean_path = os.path.join(clean_dir, clean_name)
        noisy_path = os.path.join(noisy_dir, file)
        enhanced_path = os.path.join(enhanced_dir, enhanced_name)

        if not os.path.exists(clean_path):
            print(f"Missing clean file: {clean_path}")
            continue
        if not os.path.exists(enhanced_path):
            print(f"Missing enhanced file: {enhanced_path}")
            continue

        clean_dst = os.path.join(data_folders["Clean"], clean_name)
        noisy_dst = os.path.join(data_folders["Noisy"], file)
        enhanced_dst = os.path.join(data_folders["Enhanced"], enhanced_name)
        resample_and_save(clean_path, clean_dst)
        resample_and_save(noisy_path, noisy_dst)
        resample_and_save(enhanced_path, enhanced_dst)

        metadata["samples"].append({
            "id": base_name,
            "files": {
                "Clean": f"clean/{clean_name}",
                "Noisy": f"noisy/{file}",
                "Enhanced": f"enhanced/{enhanced_name}"
            }
        })

print(f"Processed {len(metadata['samples'])} samples.")

metadata_path = os.path.join(output_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Saved metadata to {metadata_path}")
print("\nMOS-Bench project prepared successfully!")
