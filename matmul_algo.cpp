#include "matmul_algo.hpp"

#include "utils.hpp"


DenseMatrix matmulInnerABC(SparseMatrix &A, DenseMatrix &B,
        int exponent, int replication) {
    throw ShouldNotBeCalled("innerABC not yet implemented");
}


static void mulColA(const SparseMatrix &A, const DenseMatrix &B, DenseMatrix &C);
static SparseMatrix rotateColA(const SparseMatrix &A);

DenseMatrix matmulColumnA(SparseMatrix &A, DenseMatrix &B,
        int exponent, int replication) {
    DenseMatrix C(B.height, B.width);
    // FIXME add replication
    for (int iter = 0; iter < exponent; ++iter) {
        mulColA(A, B, C);
        if (iter != exponent-1) {
            B = C;
            C = DenseMatrix(B.height, B.width);
            A = rotateColA(A);
        }
    }
    return C;
}


static void mulColA(const SparseMatrix &A, const DenseMatrix &B, DenseMatrix &C) {
    // For each sparse submatrix element we have, add value to all C fields in its row
    for (int row = 0; row < A.height; ++row) {
        for (int i = A.ia[row]; i < A.ia[row+1]; ++i) {
            // element == A.a[i]
            // A col == B row == A.ja[i]
            for (int bcol = 0; bcol < B.width; ++bcol) {
                C[row][bcol] += A.a[i] * B[A.ja[i]][bcol];
            }
        }
    }
}


static SparseMatrix rotateColA(const SparseMatrix &A) {
    return A;
}
