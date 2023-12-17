import matplotlib.pyplot as plt

# Data for the first scatter plot
x1 = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]

# Data for the second scatter plot
x2 = [2, 3, 4, 5, 6]
y2 = [4, 6, 8, 10, 12]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

# Create the first scatter plot
ax1.scatter(x1, y1, s=100, c='red', alpha=0.5)

# Create the second scatter plot
ax2.scatter(x2, y2, s=100, c='blue', alpha=0.5)

# Show the plots
plt.show()