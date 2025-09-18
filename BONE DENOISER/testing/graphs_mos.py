import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/testing/mos/mos_bench_project_demucsdouble"
csv_path = os.path.join(project_root, "mos_results.csv")
output_folder = project_root


df = pd.read_csv(csv_path)


sns.set(style="whitegrid")


plt.figure(figsize=(8,6))
sns.boxplot(x="condition", y="MOS_score", data=df, palette=["skyblue","salmon","lightgreen"])
plt.title("MOS Score Distribution by Condition")
plt.ylabel("MOS Score")
plt.ylim(1,5)
plt.savefig(os.path.join(output_folder, "mos_boxplot.png"), dpi=300)
plt.close()


plt.figure(figsize=(10,6))
pivot = df.pivot(index="file", columns="condition", values="MOS_score")

for idx, row in pivot.iterrows():
    plt.plot(["noisy","enhanced","clean"], row.loc[["noisy","enhanced","clean"]], marker="o", color="gray", alpha=0.5)

means = df.groupby("condition")["MOS_score"].mean()
plt.plot(["noisy","enhanced","clean"], means, marker="D", color="red", linewidth=2, label="Mean")

plt.title("Per-sample MOS Progression: Noisy → Enhanced → Clean")
plt.ylabel("MOS Score")
plt.ylim(1,5)
plt.legend()
plt.savefig(os.path.join(output_folder, "mos_scatter_lines.png"), dpi=300)
plt.close()

print(f"Plots saved to:\n- {output_folder}/mos_boxplot.png\n- {output_folder}/mos_scatter_lines.png")
