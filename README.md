K-Means
===
kmeans is a fast BSD licensed k-means implementation(with better initialization, 
known as kmeans++) written in pure Go. It supports various distance functions out
of the box for convenience and experimentation. It has large coverage for tests.
The algorithm is tested on Iris dataset and distance functions have full test coverage.


### Distance Functions
It supports various distance functions:

- LP norms (manhattan, euclidean distances including)
- SquaredEuclideanDistance
- Minkowski Distance
- Weighted Minkowski Distance
- Chebyshev Distance
- Hamming Distance
- Bray Curtis Distance
- CanberraDistance
