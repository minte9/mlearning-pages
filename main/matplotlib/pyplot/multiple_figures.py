""" Matplotlib Multiple Figures
https://matplotlib.org/stable/tutorials/introductory/pyplot.html
"""
import matplotlib.pyplot as plt

# Create the first figure
fig1 = plt.figure()

# Plot on the first figure
plt.plot([1, 2, 3], [4, 5, 6])

# Create the second figure
fig2 = plt.figure()

# Plot on the second figure
plt.plot([4, 5, 6], [1, 2, 3])

# Show both figures
plt.show()