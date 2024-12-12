

import matplotlib.pyplot as plt
import numpy as np

data = [
    [3, 3, 3, 2, 2, 2, 2, 2, 2, 2],
    [4, 4, 4, 4, 4, 4, 2, 2, 2, 3],
]



# Count occurrences of 2, 3, and 4 in each group
values = [2, 3, 4]
group1_counts = [data[0].count(val) for val in values]
group2_counts = [data[1].count(val) for val in values]

# Convert counts to ratios
group1_ratios = [count / len(data[0]) for count in group1_counts] + [0.0]
group2_ratios = [count / len(data[1]) for count in group2_counts] + [0.0]

print(group1_ratios)

# Plotting
x = np.arange(len(values) + 1)  # Positions for groups
width = 0.35  # Width of bars

fig, ax = plt.subplots()

# Bar plots for each group
bar1 = ax.bar(x - width/2, group1_ratios, width, label='pi_b=Gaussian', color='blue')
bar2 = ax.bar(x + width/2, group2_ratios, width, label='pi_b=NF', color='orange')


plt.xticks(range(len(group1_ratios)), ['1.99 < r ≤ 2.00', '2.99 < r ≤ 3.00', '3.99 < r ≤ 4.00', 'others'])


# Customize plot
ax.set_xlabel('Return')
ax.set_ylabel('Frequency')
# ax.set_xticks(x)
# ax.set_xticklabels(values)  # Set tick labels to 2, 3, 4
ax.legend()

# Show plot
plt.tight_layout()

plt.savefig('histogram.pdf')
plt.savefig('histogram.eps')