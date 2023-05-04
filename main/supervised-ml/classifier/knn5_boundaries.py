"""Vizualize KNN Decision Regions
Create a function to plot scatter graph - plotFruitFigure()
Create a function to plot KNN's decision regions - plotKNN()
Figure 1 - Visualize the Dataset
Figure 2 - Visualize KNN Decision Regions
Figure 3 - Visualize learned decision regions of a model
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Training dataset
df = pd.DataFrame({

  'height': [
    3.91, 7.09, 10.48, 9.21, 7.95, 7.62, 7.95, 4.69, 7.50, 7.11, 
    4.15, 7.29, 8.49, 7.44, 7.86, 3.93, 4.40, 5.5, 8.10, 8.69
  ], 

  'width': [
     5.76, 7.69, 7.32, 7.20, 5.90, 7.51, 5.32, 6.19, 5.99, 7.02, 
     5.60, 8.38, 6.52, 7.89, 7.60, 6.12, 5.90, 4.5, 6.15, 5.82
  ],

  'fruit': [
    'Mandarin', 'Apple', 'Lemon', 'Lemon', 'Lemon', 'Apple', 'Mandarin', 
    'Mandarin', 'Lemon', 'Apple', 'Mandarin', 'Apple', 'Lemon', 'Apple', 
    'Apple', 'Apple', 'Mandarin', 'Lemon', 'Lemon', 'Lemon'
  ]
})
print(df)

def plotFruitFigure():

    # Define variables for graph
    H1, W1 = df.height[df.fruit == 'Apple'], df.width[df.fruit == 'Apple']
    H2, W2 = df.height[df.fruit == 'Mandarin'], df.width[df.fruit == 'Mandarin']
    H3, W3 = df.height[df.fruit == 'Lemon'], df.width[df.fruit == 'Lemon']

    # Initialize the graph
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot defined variables on it
    ax.plot(H1, W1, 'o', color='r', label='apple')
    ax.plot(H2, W2, 'o', color='g', label='mandarin')
    ax.plot(H3, W3, 'o', color='b', label='lemon')

    # Show legend and configure graph's size
    plt.legend()
    plt.ylim(3, 10) 
    plt.xlim(3, 11)
    plt.title('KNN Fruits Scatter Plot')
    plt.xlabel("height")
    plt.ylabel("width") 
    
def plotKNN(
    n_neighbors=int, 
            plot_data=True,
            plot_height=None,
            plot_width=None,
            plot_labels=None):
    
    # Turn categorical target variable into numerical to make a graph
    X = df[['height', 'width']].values
    y_encoded = df["fruit"].astype('category').cat.codes #encoded y

    # Create color maps for graph
    cmap_light = ListedColormap(['pink', 'lightblue', 'lightgreen'])
    cmap_bold = ['green', 'red', 'blue']

    # Initialize the model and train it with the dataset
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X, y_encoded)

    # Plot the decision boundary (assign a color to each point)
    x_min, x_max = X[:, 0].min() - 3, X[:, 0].max() + 3
    y_min, y_max = X[:, 1].min() - 3, X[:, 1].max() + 3

    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    
    if plot_data==True:
        # Plot also the dataset observations
        sns.scatterplot(x=plot_height, 
                        y=plot_width,#y=X[:, 1], 
                        hue=plot_labels,#df.fruit.values,
                        palette=cmap_bold, 
                        alpha=1.0, 
                        edgecolor="black")

    # Configure the graph
    plt.ylim(3, 10) 
    plt.xlim(3, 11)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(f'KNN Fruits Classifier, n={n_neighbors}')
    plt.xlabel("height")
    plt.ylabel("width")  


plotFruitFigure()
plt.figure()

plotKNN(n_neighbors=1, 
        plot_height=df['height'], 
        plot_width=df['width'], 
        plot_labels=df.fruit.values)
plt.figure()

plotKNN(n_neighbors=5, plot_data=False)
plt.show()