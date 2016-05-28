#ifndef MATMUL_INNER_HPP
#define MATMUL_INNER_HPP

#include "matrix.hpp"


DenseMatrix matmulInnerABC(SparseMatrix &A, DenseMatrix &B,
        int exponent, int replication);


DenseMatrix matmulColumnA(SparseMatrix &A, DenseMatrix &B,
        int exponent, int replication);


#endif  // MATMUL_INNER_HPP
