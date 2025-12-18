import matplotlib.pyplot as plt
import numpy as np

# Correct the add_bars function to avoid the error by making sure that 'data' is a list of appropriate length
def add_bars(position, data, label, color, hatch):
    # Ensure that 'data' is a list with the correct length before trying to plot
    if not isinstance(data, list) or len(data) != len(position):
        raise ValueError("Data should be a list with the same length as 'position'")
    return plt.bar(position, data, bar_width, label=label, color=color, hatch=hatch, alpha=0.7, edgecolor='black')

# Define the data
categories = ['C10→A5', 'C10→W5', 'C10→D5', 'A10→C5', 'W10→C5', 'D10→C5']
base = [96.16, 100.00, 98.53, 91.26, 93.15, 93.32]
spw = [96.27, 100.00, 98.49, 92.79, 93.25, 94.35]
csd = [96.63, 100.00, 100.00, 92.86, 93.20, 94.16]
dcaw = [96.98, 100.00, 100.00, 93.40, 93.92, 94.62]
average = [95.40, 95.83, 96.18, 96.49]

bar_width = 0.2  # Width of the bars
index = np.arange(len(categories) + 1)  # Include space for 'Average' category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colors
bar_patterns = ['/', '\\', 'x', '-']  # Patterns

# Update the dpi setting and create the figure
plt.figure(figsize=(10, 8), dpi=600)

# Plot the bars
for i, (data, label, color, hatch) in enumerate(zip([base, spw, csd, dcaw], ['Base', '+SPW', '+CSD', 'DCAW'], colors, bar_patterns)):
    position = index + i * bar_width
    data_with_average = data + [average[i]]  # Adding the average to the data list
    add_bars(position, data_with_average, label, color, hatch)

# Customize the plot with titles, labels, and grid
plt.xticks(index + 1.5 * bar_width, categories + ['Average'], fontsize=20, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=26, fontweight='bold')
plt.title('Ablation Studies of the DCAW Framework on Caltech-Office Dataset', fontsize=28, fontweight='bold')
plt.ylim(90, 100)
plt.legend(prop={'size': 22})
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.tight_layout()

# Save the plot with a more aesthetic look and high-resolution
plt.savefig('/root/BCD_PDA/Vis/high_res_bar_chart.png', dpi=600)

# Show the plot
plt.show()
