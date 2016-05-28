#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP
#include <string>

#include "matrix.hpp"

using std::string;

/// Return the index of first row/column owned by rank-th process when dividing matrix of a given
/// size into almost-equal parts. If rank > parts, it is taken modulo parts.
int firstIdxForProcess(int size, int parts, int rank);


/// Return number of elements owned by rank-th process when dividing matrix of a given
/// size into almost-equal parts. If rank > parts, it is taken modulo parts.
int idxsForProcess(int size, int parts, int rank);


/// Read a sparse matrix from file, return true if successful
bool readSparseMatrix(const string &filename, SparseMatrix &m);


/// Gather and show a dense matrix. Assumes it's divided equally between processes.
void gatherAndShow(DenseMatrix &m);

#endif  // MATRIX_UTILS_HPP
