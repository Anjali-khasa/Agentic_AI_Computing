import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

current_folder = Path(__file__).resolve().parent
df = pd.read_csv(current_folder / "sound_data.csv")

df.columns = df.columns.str.strip()
df["location"] = df["location"].astype(str).str.strip()
df["type"] = df["type"].astype(str).str.strip()
df["zone_type"] = df["zone_type"].astype(str).str.strip()
df["time_block"] = df["time_block"].astype(str).str.strip()
df["decibel_level"] = pd.to_numeric(df["decibel_level"], errors="coerce")
df = df.dropna()

df["location_short"] = df["location"].replace({
    "Outside Retriever Activities Center": "Retriever Center",
    "Albin O. Kuhn Library": "Library",
    "University Center": "Univ Center",
    "ITE Building": "ITE",
    "Commons": "Commons"
})

location_avg = df.groupby("location_short")["decibel_level"].mean().sort_values(ascending=False)
location_order = location_avg.index.tolist()
max_location = location_avg.idxmax()

zone_order = ["Quiet", "Moderate", "Loud"]
zone_avg = df.groupby("zone_type")["decibel_level"].mean().reindex(zone_order)

type_avg = df.groupby("type")["decibel_level"].mean()

pivot = df.pivot_table(
    values="decibel_level",
    index="location_short",
    columns="time_block",
    aggfunc="mean"
).reindex(location_order)

# -------------------------------
# Creating the dashboard
# -------------------------------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

fig.suptitle("Campus Sound Analysis", fontsize=20)

# -------------------------------
# 1. Average sound by location
# -------------------------------
ax1 = fig.add_subplot(gs[0, 0])

colors = ["red" if loc == max_location else "steelblue" for loc in location_avg.index]
ax1.bar(location_avg.index, location_avg.values, color=colors)

for i, v in enumerate(location_avg.values):
    ax1.text(i, v + 0.5, f"{v:.1f}", ha="center")

ax1.text(
    location_avg.index.get_loc(max_location),
    location_avg.max() + 2,
    "Highest",
    ha="center",
    fontsize=10,
    color="red"
)

ax1.set_title("Average Sound Level by Location")
ax1.set_ylabel("Decibel Level (dB)")
ax1.tick_params(axis="x", rotation=20)

# -------------------------------
# 2. Variation by location
# -------------------------------
ax2 = fig.add_subplot(gs[0, 1])

boxplot_data = [df[df["location_short"] == loc]["decibel_level"] for loc in location_order]
ax2.boxplot(boxplot_data, labels=location_order)
ax2.set_title("Variation in Sound Levels Across Locations")
ax2.set_ylabel("Decibel Level (dB)")
ax2.tick_params(axis="x", rotation=20)

# -------------------------------
# 3. Time-based comparison table
# -------------------------------
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis("off")

table = ax3.table(
    cellText=pivot.round(1).values,
    rowLabels=pivot.index,
    colLabels=pivot.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.25, 1.5)

ax3.set_title("Time-based Sound Comparison (Morning vs Afternoon)", pad=15)

# -------------------------------
# 4. Comparison by type and zone
# -------------------------------
ax4 = fig.add_subplot(gs[1, 1])

x_type = [0, 1]
x_zone = [3, 4, 5]

ax4.bar(x_type, type_avg.values, width=0.5, color="steelblue", label="Location Type")
ax4.bar(x_zone, zone_avg.values, width=0.5, color="darkorange", label="Zone Type")

for i, v in zip(x_type, type_avg.values):
    ax4.text(i, v + 0.5, f"{v:.1f}", ha="center")

for i, v in zip(x_zone, zone_avg.values):
    ax4.text(i, v + 0.5, f"{v:.1f}", ha="center")

ax4.set_xticks(x_type + x_zone)
ax4.set_xticklabels(list(type_avg.index) + zone_order)
ax4.set_title("Sound Comparison by Type and Zone")
ax4.set_ylabel("Decibel Level (dB)")
ax4.legend()

# -------------------------------
# Insight text
# -------------------------------
fig.text(
    0.5,
    0.02,
    "Outdoor areas and Loud zones show higher sound levels, while the Library remains one of the quietest locations.",
    ha="center",
    fontsize=11
)

# -------------------------------
# Final layout
# -------------------------------
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(current_folder / "final_clean_dashboard.png", dpi=300, bbox_inches="tight")
plt.show()