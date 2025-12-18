import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
# Load the data
data = pd.read_excel('/root/autodl-tmp/BCD_PDA/office_31_log/visual/fig6/fig6_acc_test_data.xlsx')

# Setting the deeper colors for the lines
# Setting a color scheme similar to the provided chart but with distinct colors
color_scheme = ['#34A853', '#4285F4']  # Green and blue colors from the chart's color scheme

# Finding the first highest value for each series
first_max_D_to_W = data['D->W'].idxmax()
first_max_W_to_D = data['W->D'].idxmax()

# Re-plotting with the new color scheme and larger markers for the first highest values
fig, ax = plt.subplots(figsize=(16, 10))
# Plotting each series with the new color scheme
ax.plot(data['iter'].values, data['D->W'].values, marker='s', label='D->W', color=color_scheme[0])
ax.plot(data['iter'].values, data['W->D'].values, marker='^', label='W->D', color=color_scheme[1])

# Marking the first highest value with larger markers
ax.plot(data['iter'][first_max_D_to_W], data['D->W'][first_max_D_to_W], 'o', color=color_scheme[0], markersize=12)
ax.plot(data['iter'][first_max_W_to_D], data['W->D'][first_max_W_to_D], 'o', color=color_scheme[1], markersize=12)


# Adding labels and title with bold font
ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy over Iterations for D->W and W->D', fontsize=14, fontweight='bold')

# Adding the legend
ax.legend()

# Setting grid
ax.grid(True)

# Save the plot with high dpi
plt.savefig('/root/autodl-tmp/BCD_PDA/office_31_log/visual/fig6/accuracy_plot_updated.png', dpi=1200)

# Show the plot
plt.show()
