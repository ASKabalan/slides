import matplotlib.pyplot as plt
import numpy as np

# Data for the first plot
grid_sizes = np.array([512, 1024, 2048, 4096, 8192])
memory_needed = (grid_sizes**3 * 8 * 3) / 1024**3

# Data for the second plot
release_years = np.array([2014, 2016, 2018, 2020, 2022, 2024])
gpu_memory = np.array([12, 16, 32, 80, 80, 96])
gpu_labels = ['K80', 'P100', 'V100', 'A100', 'H100', 'B100']

# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# First plot
ax1.plot(grid_sizes,
         memory_needed,
         marker='o',
         color='violet',
         label='Memory Needed')
ax1.set_title('Simulation Sizes', color='white')
ax1.set_xlabel('Grid Size', color='white')
ax1.set_ylabel('Memory Needed (GB)', color='white')
ax1.patch.set_alpha(0.0)
ax1.grid(False)
ax1.spines['top'].set_color('white')
ax1.spines['bottom'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.spines['right'].set_color('white')
ax1.tick_params(axis='y',
                which='both',
                labelleft=True,
                labelright=False,
                colors='white')
ax1.tick_params(axis='x', colors='white')
legend1 = ax1.legend(loc='lower right',
                     fontsize='small',
                     facecolor='black',
                     edgecolor='white')
for text in legend1.get_texts():
    text.set_color('white')

# Second plot
ax2.plot(release_years,
         gpu_memory,
         marker='o',
         color='lime',
         label='GPU Memory')
for i, label in enumerate(gpu_labels):
    ax2.text(release_years[i],
             gpu_memory[i] + 2,
             label,
             fontsize=9,
             color='white',
             ha='right')  # Adjusted position

ax2.set_title('GPU Sizes from Kepler to Hopper', color='white')
ax2.set_xlabel('Release Year', color='white')
ax2.set_ylabel('Memory (GB)', color='white')
ax2.patch.set_alpha(0.0)
ax2.grid(False)
ax2.spines['top'].set_color('white')
ax2.spines['bottom'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.spines['right'].set_color('white')
ax2.tick_params(axis='y',
                which='both',
                labelleft=True,
                labelright=False,
                colors='white')
ax2.tick_params(axis='x', colors='white')
legend2 = ax2.legend(loc='lower right',
                     fontsize='small',
                     facecolor='black',
                     edgecolor='white')
for text in legend2.get_texts():
    text.set_color('white')

# Save the figure with a transparent background
plt.savefig('simulation_gpu_sizes.png', dpi=300, transparent=True)

# Show the plot
plt.show()
