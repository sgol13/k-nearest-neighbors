# K-nearest neighbors
This program is an implementation of a [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). The algorithm is applied to an exemplary dataset (medical data concerning cancer types) to test its performance in practice.

In the listing below you can observe the algorithms performance for various combinations of metrics and K values. Numbers indicate the percentage of wrong test results. Hence, the lower the number, the more accurate the classifier is.

```
                K = 1   K = 3   K = 5   K = 7   K = 9   K = 11  K = 13  K = 15  K = 17
 Euclidean       1.75%   1.46%   1.46%   1.46%   1.17%   1.17%   1.17%   1.17%   1.46%
 Manhattan       1.75%   1.17%   1.46%   1.46%   1.46%   1.17%   1.17%   1.17%   1.17%
 Chebyshev       3.22%   2.92%   2.34%   1.75%   1.75%   1.75%   1.46%   1.17%   1.17%
 Railway        23.68%  23.68%  23.68%  23.68%  23.68%  23.68%  23.68%  23.68%  23.68%
 Hamming        13.45%   9.65%   9.06%   9.06%   9.06%   7.31%   7.02%   7.02%   7.89%
 Correlation    23.39%  22.81%  22.81%  22.81%  23.98%  23.98%  23.98%  23.98%  23.39%

```

## Compilation
```
git clone https://github.com/sgol13/k-nearest-neighbors.git
cd k-nearest-neighbors
g++ -o classifier main.cpp
./classifier
```

## License
This project is under MIT [license](LICENSE).
