import matplotlib.pyplot as plt

# Create some data for the scatter plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create the scatter plot
plt.scatter(x, y, s=100, c='red', alpha=0.5)

# Add a title and axis labels
plt.title('Scatter Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()