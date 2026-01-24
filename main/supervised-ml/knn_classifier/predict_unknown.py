# PREDICT UNKNOWN - KNN
# ---------------------
# Concept:
#   - We provide training dataset ponts (features) and label (target).
#   - We train the model (with k=3 nearest neighbors constrain).
#   - We are able to predict the label (y) for a new (unknown) data point.
# ------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

# Training dataset
# ----------------
X = [[0,0],
     [1,1],
     [2,2],
     [3,3]]
y = [0, 1, 0, 1]

# Train the model
# ---------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Make predictions for unknown
# ----------------------------
x_unknown = [1,2]
y_pred = knn.predict([x_unknown])

# Output results
# --------------
print(x_unknown)  # [1, 2]
print(y_pred)     # [0]