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


/// Return size, start (inclusive) and end (exclusive) of i-th matrix part, in rows/cols,
/// either before replication (p parts) or after (p/c parts).
/// (1) for dividing A matrix in whichever algorithm is active
int partASize(bool repl, int i);
int partAStart(bool repl, int i);
int partAEnd(bool repl, int i);
/// (2) for dividing B matrix in innerABC algorithm
int innerBPartSize(bool repl, int i);
int innerBPartStart(bool repl, int i);
int innerBPartEnd(bool repl, int i);
int innerAWhichReplGroup(int i);
int innerBWhichReplGroup(int i);


void initPartSizes();


/// Read a sparse matrix from file, return true if successful
bool readSparseMatrix(const string &filename, SparseMatrix &m);


/// Gather and show a dense matrix. Assumes it's divided equally between processes.
/// @param m dense matrix column block.
DenseMatrix gatherAndShow(const DenseMatrix &m);


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
SparseMatrix splitAndScatter(SparseMatrix &m, vector<int> &nnzs);


/// @param m rank-th block(row|col) of matrix A divided into p parts
/// @return the appropriate part of matrix A divided and replicated into p/c parts
void replicateA(SparseMatrix &m, vector<int> &nnzs);


DenseMatrix generateBFragment();


int reduceGeElems(const DenseMatrix &m_part, double bound);


#endif  // MATRIX_UTILS_HPP
