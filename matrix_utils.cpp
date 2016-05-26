#include "matrix_utils.hpp"
using namespace std;


DenseMatrix::DenseMatrix(int h, int w): Matrix(h, w), data(h) {
    for (auto row : data) row.resize(w);
}


DenseMatrix::DenseMatrix(int h, int w, MatrixGenerator gen, int seed): DenseMatrix(h, w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            at(i, j) = gen(seed, i, j);
        }
    }
}


int firstIdxForProcess(int size, int parts, int rank) {
    rank %= parts;
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) * rank + (rank > numSmaller ? rank - numSmaller : 0);
}


int elemsForProcess(int size, int parts, int rank) {
    rank %= parts;
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) + (rank > numSmaller ? 1 : 0);
}
