import numpy as np
import matplotlib.pyplot as plt

# Generate a sample dataset: percentages of fat in some food items
np.random.seed(0) # set seed for reproducibility
fat_percentage = np.random.normal(25, 5, 1000) # mean 25%, std 10%, 1000 samples

# Create histogram
plt.figure(figsize=(8, 6))
plt.hist(fat_percentage, bins=15, color='skyblue', edgecolor='black')
plt.title('Histogram of % Fat in Food items')
plt.xlabel('% Fat')
plt.ylabel('Frequency')
plt.show()