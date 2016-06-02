#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>

#include "matrix.hpp"

using std::string;
using std::ostream;
using std::vector;


/// Return size of i-th matrix part (in rows/cols), either before replication (p parts) or after (p/c parts)
int partSize(bool repl, int i);


/// Return first index of i-th matrix part (in rows/cols), either before replication (p parts) or after (p/c parts)
int partStart(bool repl, int i);
int partEnd(bool repl, int i);


void initPartSizes();


/// Read a sparse matrix from file, return true if successful
bool readSparseMatrix(const string &filename, SparseMatrix &m);


/// Gather and show a dense matrix. Assumes it's divided equally between processes.
/// @param m dense matrix column block.
void gatherAndShow(DenseMatrix &m);
void gatherAndShow(DenseMatrix &m, int parts, MPI::Intracomm &comm);


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


/// @param m rank-th block(row|col) of matrix A divided into p parts
/// @return the appropriate part of matrix A divided and replicated into p/c parts
SparseMatrix replicateA(const SparseMatrix &m, vector<int> &nnzs);


DenseMatrix generateBFragment();


#endif  // MATRIX_UTILS_HPP
