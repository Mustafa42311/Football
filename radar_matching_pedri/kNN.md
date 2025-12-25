# K-Nearest Neighbors (KNN) Algorithm

K-Nearest Neighbors (KNN) is a non-parametric, supervised learning algorithm used for both classification and regression tasks. It is often referred to as a "lazy learner" because it does not learn a discriminative function from the training data during a training phase. Instead, it memorizes the entire dataset and performs computations only when a prediction is required.

The core intuition behind KNN is the proximity hypothesis: similar data points likely exist in close proximity to one another in an n-dimensional feature space.

## Measuring Similarity

To determine which data points are "nearest" to a new query point, the algorithm must quantify similarity. It does this by calculating the geometric distance between points in the feature space.

The most common metric used is Euclidean Distance. For two points, p and q, in an n-dimensional space, the distance is calculated as:
$$d(p,q)=∑n​(qi​−pi​)2​$$

Other distance metrics include:
- Manhattan Distance: The sum of absolute differences (useful for high-dimensional grid-like data).
- Minkowski Distance: A generalized form of both Euclidean and Manhattan distances.

## The Role of 'K'

The variable k represents the number of nearest neighbors the algorithm considers when making a prediction. It is the most critical hyperparameter in the model.
Small k (e.g., k=1 or k=3): The model becomes sensitive to noise and outliers. This can lead to overfitting, where the decision boundaries are too jagged and complex.
Large k: The model becomes overly smoothed. It may include neighbors from other classes simply because they are abundant in the dataset, leading to underfitting and high bias.

## Feature Scaling

One of the most critical prerequisites for KNN is Feature Scaling. Since the algorithm relies on distance calculations, features with large magnitudes (e.g., Salary: 50,000 – 100,000) will dominate features with small magnitudes (e.g., Age: 20 – 60), even if the smaller feature is more relevant.

To prevent this bias, data is typically normalized using Min-Max Scaling or Standardization (Z-score) to bring all features into the same range (usually 0 to 1).
$$\frac{X-X{min}}{X{max}-X{min}}$$

## Prediction Mechanism
Once the nearest neighbors are identified, the algorithm makes a prediction based on the task.
The algorithm calculates the Average (Mean) of the values of the k nearest neighbors.
Example: To predict a house price, it finds the 5 most similar houses and averages their selling prices.
