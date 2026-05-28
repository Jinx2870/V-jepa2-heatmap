import matplotlib.pyplot as plt

# ---------------- Font ----------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]  # 或 ["DejaVu Sans"]
plt.rcParams["font.size"] = 10
# ---------------- Data ----------------
phases = [
    "Phase 1: Project Initiation",
    "Phase 2: Data Preparation & Gaze Extraction",
    "Phase 3: Model Adaptation & Training",
    "Phase 4: Evaluation & Analysis",
    "Phase 5: Integration Insights for Group Project",
    "Phase 6: Thesis Defense Preparation",
]

start_times = [1, 2, 2, 3, 4, 8]
durations   = [1, 1, 3, 3, 4, 3]

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Muted academic colors
colors = [
    "#4C72B0",
    "#55A868",
    "#8172B2",
    "#64B5CD",
    "#8C8C8C",
    "#C44E52",
]

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(12, 5))

ax.barh(
    phases,
    durations,
    left=start_times,
    height=0.5,              # 👈 扁一点
    color=colors,
    edgecolor="0.3",
    linewidth=0.4
)

# X-axis
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.set_xlim(1, 12)
ax.set_xlabel("Timeline (2026)")

# Title
ax.set_title("Research Project Milestones (2026)", pad=14)

# Y-axis
ax.invert_yaxis()
ax.tick_params(axis="y", pad=6)

# Grid & Spines
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("milestones_timeline.png", dpi=300, bbox_inches="tight")
plt.show()