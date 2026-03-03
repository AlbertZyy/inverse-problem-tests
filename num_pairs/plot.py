from matplotlib import pyplot as plt


data = {
    "nograd": [0.0481, 0.0401, 0.0427, 0.0437, 0.0420, 0.0428, 0.0448, 0.0399],
    "single": [0.0380, 0.0344, 0.0356, 0.0368, 0.0355, 0.0358, 0.0379, 0.0369],
    "multi":  [0.0380, 0.0353, 0.0367, 0.0384, 0.0367, 0.0378, 0.0392, 0.0361],
} # 0.0378

# please draw the plot of the data above, requirements:
# 1. x-axis with label "Number of data pairs (channels)"
# 2. y-axis with label "Validation Loss"
# 3. title "Validation Loss vs Number of Data Pairs"
# 4. legend for each line: "$\gamma$=0", "Single-$\gamma$", "Multi-$\gamma$"
# 5. font size suitable for journal publication
# 6. grid on

plt.figure(figsize=(8, 6))
for key, values in data.items():
    if key == "nograd":
        label = "$\gamma$=0"
    elif key == "single":
        label = "Single-$\gamma$"
    else:
        label = "Multi-$\gamma$"
    plt.plot(range(1, len(values) + 1), values, marker='o', label=label)
plt.xlabel("Number of Data Pairs (Channels)", fontsize=14)
plt.ylabel("Validation Loss", fontsize=14)
plt.title("Validation Loss vs Number of Data Pairs", fontsize=16)
plt.legend(fontsize=12)
plt.grid()
plt.xticks(range(1, len(values) + 1))
plt.tight_layout()
plt.savefig("num_pairs/validation_loss_vs_data_pairs.png", dpi=300)
plt.show()