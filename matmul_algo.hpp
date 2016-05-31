#ifndef MATMUL_INNER_HPP
#define MATMUL_INNER_HPP
#include <vector>

#include "matrix.hpp"
using std::vector;


class Multiplicator {
private:
    vector<double> recv_a_v, send_a_v;
    vector<int> recv_ij_v, send_ij_v;
    int c;  // replication
    int p;  // num_processes
    int n;  // main matrix size
    int rank = Flags::NOT_SET;
    int part_id = Flags::NOT_SET;

    void mulColA();
    void rotateColA();

public:
    SparseMatrix A;  // A matrix part for each process
    DenseMatrix B;  // B matrix part
    DenseMatrix C;  // result part
    vector<int> nnzs;

    Multiplicator(SparseMatrix &_A, DenseMatrix &_B, vector<int> &nnzs);

    DenseMatrix matmulInnerABC(int exponent, int replication);

    DenseMatrix matmulColumnA(int exponent, int replication);

};


#endif  // MATMUL_INNER_HPP
