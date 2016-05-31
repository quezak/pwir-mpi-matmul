#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP
#include <iostream>
#include <string>
#include <vector>

#include "matrix.hpp"

using std::string;
using std::ostream;
using std::vector;

/// Return the index of first row/column owned by rank-th process when dividing matrix of a given
/// size into almost-equal parts. If rank > parts, it is taken modulo parts.
int firstIdxForProcess(int size, int parts, int rank);


/// Return number of elements owned by rank-th process when dividing matrix of a given
/// size into almost-equal parts. If rank > parts, it is taken modulo parts.
int idxsForProcess(int size, int parts, int rank);


/// Return maximum number of elements owned by any process. In current implementation that's the
/// last one.
int maxIdxsForProcess(int size, int parts);


/// Read a sparse matrix from file, return true if successful
bool readSparseMatrix(const string &filename, SparseMatrix &m);


/// Gather and show a dense matrix. Assumes it's divided equally between processes.
/// @param m dense matrix column block.
void gatherAndShow(DenseMatrix &m);


template<class T>
ostream& operator<< (ostream& output, const vector<T> &v) {
    bool first = true;
    for (const T &t : v) {
        if (first) { output << "["; first = false; }
        else { output << ", "; }
        output << t;
    }
    output << "]" << std::endl;
    return output;
}


/// Split a sparse matrix into parts and distribute it among processes.
/// @param m a sparse matrix, nonempty only for scatter root
/// @return i-th column block (as sparse submatrix) of matrix m for process i
SparseMatrix splitAndScatter(const SparseMatrix &m, vector<int> &nnzs);


DenseMatrix generateBFragment();


#endif  // MATRIX_UTILS_HPP
